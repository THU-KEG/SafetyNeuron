
import os
import csv
import argparse
import random
from collections import Counter, defaultdict
from functools import partial

import pandas as pd
import numpy as np
import torch
import datasets
from transformers import PreTrainedTokenizerBase

from peft.peft_model import PeftModel
from src.utils import get_act_name, seed_torch
from eval.utils import generate_completions_and_scores, dynamic_import_function, load_hf_score_lm_and_tokenizer, load_hooked_lm_and_tokenizer, load_hf_lm_and_tokenizer


def layer_patch_hook(value, hook, neurons, patched_values):
    """ 
    replace specific neurons activation(given by **neurons**) with **patched_values**
    """
    try:
        if not isinstance(patched_values, torch.Tensor):
            patched_values = torch.tensor(patched_values)
        patched_values = patched_values.to(value)
        value[..., neurons] = patched_values
    except Exception as e:
        print(f'Error in hook {hook}', e)
    return value

def perturb_hook(activation, hook, neuron, value):
    """ 
    add noise to specific neurons activation
    """
    try:
        activation[...,neuron] += (torch.randn_like(value) * 3 * value).to(activation.device)
    except Exception as e:
        print(f'Error in hook {hook}', e)
    return activation

def is_same_tokenizer(
    tokenizer: PreTrainedTokenizerBase,
    other_tokenizer: PreTrainedTokenizerBase,
) -> bool:
    """Check if two tokenizers are the same."""
    return tokenizer is other_tokenizer or (
        tokenizer.__class__ == other_tokenizer.__class__
        and tokenizer.get_vocab() == other_tokenizer.get_vocab()
    )

# set operation for tensors
def tensor_intersect(a, b):
    b_set = set([(x, y) for x, y in b.tolist()])
    a_list = a.tolist()
    return torch.tensor([[x, y] for x, y in a_list if (x, y) in b_set])
def tensor_substract(a, b):
    b_set = set([(x, y) for x, y in b.tolist()])
    a_list = a.tolist()
    return torch.tensor([[x, y] for x, y in a_list if (x, y) not in b_set])

def get_save_name(args):
    save_name = 'vanilla'
    if args.patch_mean:
        save_name = 'patch_mean'
    elif args.patch_zero:
        save_name = 'patch_zero'
    elif args.add_noise:
        save_name = 'add_noise'
    elif args.guided_generation:
        save_name = f'guided_by_{os.path.basename(args.blue_peft_path[-1])}' if args.blue_peft_path is not None else f'guided_by_{os.path.basename(args.model_name_or_path)}'
    if args.index_path:
        save_name += f'_idx_{os.path.basename(args.index_path).rsplit(".", 1)[0]}'
        if args.value_path and args.index_path != args.value_path:
            save_name += f'_value_{os.path.basename(args.value_path).rsplit(".", 1)[0]}'
    if args.ignore_index_path:
        save_name += f'_sub_{os.path.basename(args.ignore_index_path).rsplit(".", 1)[0]}' 
    if args.intersect_index_path:
        save_name += f'_intersect_{os.path.basename(args.intersect_index_path).rsplit(".", 1)[0]}' 
    if args.generation_startswith != '':
        save_name += f'_startswith_{args.generation_startswith}'
    if args.use_random_neurons:
        save_name += f'_random_neurons'
    if args.sliding_window is not None:
        save_name += f'_window_{args.sliding_window}'
    return save_name
    
def main(args):
    seed_torch()
    os.makedirs(args.save_dir, exist_ok=True)

    print("loading data and model...")
    eval_data = datasets.load_dataset('json', data_files=args.dataset)["train"]
    prompts = []
    chat_formatting_function = dynamic_import_function(args.chat_formatting_function) if args.use_chat_format else None
    for example in eval_data:
        prompt = example["prompt"]
        if args.use_chat_format:
            messages = [{"role": "user", "content": prompt}]
            prompt = chat_formatting_function(messages, add_bos=False)
        prompts.append(prompt + args.generation_startswith)

    prompts = random.sample(prompts[-3*args.num_samples:], args.num_samples)
    if args.model_name_or_path is not None:
        if args.hf_model:
            red_model, tokenizer = load_hf_lm_and_tokenizer(
                model_name_or_path=args.model_name_or_path, 
                tokenizer_name_or_path=args.tokenizer_name_or_path,
                load_in_8bit=args.load_in_8bit, 
                device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
            )
            if args.red_peft_path:
                for red_peft_path in args.red_peft_path:
                    red_model = PeftModel.from_pretrained(red_model, red_peft_path)
                    red_model = red_model.merge_and_unload()
                    print(f"Load LoRA module from {red_peft_path}.")
        else:
            red_model, tokenizer = load_hooked_lm_and_tokenizer(
                model_name_or_path=args.model_name_or_path,
                tokenizer_name_or_path=args.tokenizer_name_or_path if args.tokenizer_name_or_path is not None else args.model_name_or_path,
                load_in_8bit=args.load_in_8bit,
                device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
                peft_name_or_path=args.red_peft_path
            )

        if args.cost_model_name_or_path:
            cost_model, cost_tokenizer = load_hf_score_lm_and_tokenizer(
                model_name_or_path=args.cost_model_name_or_path,
                tokenizer_name_or_path=args.cost_tokenizer_name_or_path if args.cost_tokenizer_name_or_path is not None else args.cost_model_name_or_path,
                load_in_8bit=args.load_in_8bit,
                device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
            )
        else:
            cost_model, cost_tokenizer = None, None
            
        if args.reward_model_name_or_path:
            reward_model, reward_tokenizer = load_hf_score_lm_and_tokenizer(
                model_name_or_path=args.reward_model_name_or_path,
                tokenizer_name_or_path=args.reward_tokenizer_name_or_path if args.reward_tokenizer_name_or_path is not None else args.reward_model_name_or_path,
                load_in_8bit=args.load_in_8bit,
                device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
            )
        else:
            reward_model, reward_tokenizer = None, None
            
        if is_same_tokenizer(tokenizer, cost_tokenizer):
            cost_tokenizer = tokenizer
        if is_same_tokenizer(tokenizer, reward_tokenizer):
            reward_tokenizer = tokenizer

        stop_id_sequences = None
        red_outputs, red_cost_scores, red_reward_scores, _ = generate_completions_and_scores(
            model=red_model,
            tokenizer=tokenizer,
            cost_model=cost_model,
            cost_tokenizer=cost_tokenizer,
            reward_model=reward_model,
            reward_tokenizer=reward_tokenizer,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.eval_batch_size if args.eval_batch_size else 1,
            do_sample=False,
            stop_id_sequences=stop_id_sequences
        )
        
        is_manipulated = any([args.add_noise, args.patch_mean, args.patch_zero, args.patch_flip, args.guided_generation])
        save_name = get_save_name(args)
        red_name = os.path.basename(args.red_peft_path[-1]) if args.red_peft_path else os.path.basename(args.model_name_or_path)
        # evaluate red model only
        if not args.blue_peft_path and not is_manipulated: 
            columns = [
                'Prompt',
                f'{red_name}',
                'Cost/Reward'
            ]    
            table = []
            for i in range(len(prompts)):
                row = (prompts[i], red_outputs[i], red_cost_scores[i])
                table.append(row)
            table_output_dir = os.path.join(
                args.save_dir,
                red_name
            )
            os.makedirs(table_output_dir, exist_ok=True)
            output_file = os.path.join(table_output_dir, f'{save_name}_table.csv')
            with open(output_file, mode='w', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(columns)
                writer.writerows(table)
                
            print('The following analysis is under the preference of the cost/reward model.',)
            cost_red = np.asarray([row[2] for row in table])
            print(f'Average cost/reward of {red_name}: {cost_red.mean()}')
            os._exit(0)

        if not args.guided_generation and args.blue_peft_path:
            del red_model
            torch.cuda.empty_cache()

        if args.blue_peft_path:
            blue_model, tokenizer = load_hooked_lm_and_tokenizer(
                model_name_or_path=args.model_name_or_path,
                tokenizer_name_or_path=args.tokenizer_name_or_path if args.tokenizer_name_or_path is not None else args.model_name_or_path,
                load_in_8bit=args.load_in_8bit,
                device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
                peft_name_or_path=args.blue_peft_path
            )
        else:
            blue_model = red_model
            
        x, blue_cost, blue_reward, red_cost, red_reward, blue_cost_std, red_cost_std, blue_reward_std, red_reward_std= [], [], [], [], [], [], [], [], []
        hook_fn = lambda v, h: v
        blue_name = os.path.basename(args.blue_peft_path[-1]) if args.blue_peft_path else os.path.basename(args.model_name_or_path)
        
        if is_manipulated:
            if args.index_path:
                _, index, base_mean, base_std, peft_mean, peft_std = torch.load(args.index_path)
                if args.value_path and args.value_path != args.index_path:
                    _, _, base_mean, base_std, peft_mean, peft_std = torch.load(args.value_path)
            if args.ignore_index_path:
                _, ignore_index, *_ = torch.load(args.ignore_index_path)
            if args.intersect_index_path:
                _, intersect_index, *_ = torch.load(args.intersect_index_path)
        else:
            index = []
            
        for topk in args.topk_ablate:
            x.append(topk)
            if args.sliding_window is not None:
                topk_index = index[topk:topk+args.sliding_window]
            else:
                topk_index = index[:topk]
            if args.ignore_index_path:
                topk_index = tensor_substract(topk_index, ignore_index[:topk_index.shape[0]])
            if args.intersect_index_path:
                topk_index = tensor_intersect(topk_index, intersect_index[:topk_index.shape[0]])
            if args.use_random_neurons and topk > 0:
                counts = Counter(topk_index[:, 0].tolist())
                print('Number of neuron each layer: ', counts)
                topk_index = []
                for layer, num in counts.items():
                    neurons = random.sample(range(blue_model.config.intermediate_size), num)
                    topk_index += [[layer, neuron] for neuron in neurons]
                topk_index = torch.tensor(topk_index)
                counts = Counter(topk_index[:, 0].tolist())
                print('Number of neuron each layer after sample: ', counts)

            if not args.guided_generation: # normal activation patching
                layers = defaultdict(list)
                for layer, idx in topk_index:
                    layers[layer.item()].append(idx)
                for layer, neurons in layers.items():
                    neurons = torch.tensor(neurons)
                    if args.patch_mean:
                        if args.blue_peft_path:
                            hook_fn = partial(layer_patch_hook, neurons=neurons, patched_values=peft_mean[layer, neurons])
                        else:
                            hook_fn = partial(layer_patch_hook, neurons=neurons, patched_values=base_mean[layer, neurons])                           
                    elif args.add_noise:
                        if args.blue_peft_path:
                            hook_fn = partial(perturb_hook, neuron=neurons, value=peft_std[layer, neurons])
                        else:
                            hook_fn = partial(perturb_hook, neuron=neurons, value=base_std[layer, neurons])
                    elif args.patch_zero:
                        hook_fn = partial(layer_patch_hook, neurons=neurons, patched_values=0)

                    blue_model.add_perma_hook(name=get_act_name('post', layer), hook=hook_fn)
            
                print(f"running with {len(topk_index)} neurons got ablated")
                blue_outputs, blue_cost_scores, blue_reward_scores, _ = generate_completions_and_scores(
                    model=blue_model,
                    tokenizer=tokenizer,
                    cost_model=cost_model,
                    cost_tokenizer=cost_tokenizer,
                    reward_model=reward_model,
                    reward_tokenizer=reward_tokenizer,
                    prompts=prompts,
                    max_new_tokens=args.max_new_tokens,
                    batch_size=args.eval_batch_size if args.eval_batch_size else 1,
                    do_sample=False,
                    stop_id_sequences=stop_id_sequences
                )
                
                blue_model.reset_hooks(including_permanent=True)

            else: # dynamic activation patching
                print(f"running with {len(topk_index)} neurons replaced by {blue_name} model")
                hook_fn = layer_patch_hook
                blue_outputs, blue_cost_scores, blue_reward_scores, _ = generate_completions_and_scores(
                    model=red_model,
                    tokenizer=tokenizer,
                    cost_model=cost_model,
                    cost_tokenizer=cost_tokenizer,
                    reward_model=reward_model,
                    reward_tokenizer=reward_tokenizer,
                    prompts=prompts,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    batch_size=args.eval_batch_size if args.eval_batch_size else 1,
                    guided_model=blue_model, # provide guided model as generation parameter
                    index=topk_index,
                    hook_fn=hook_fn,
                    stop_id_sequences=stop_id_sequences
                )
                  
            folder_name = f'{red_name}_vs_{blue_name}' if red_name != blue_name else red_name
            columns = [
                'Prompt',
                f'{red_name}',
                'Cost',
                'Reward',
                f'{blue_name}',
                'Cost',
                'Reward',
            ]    

            table = []
            for i in range(len(prompts)):
                row = (prompts[i], red_outputs[i], red_cost_scores[i], red_reward_scores[i], blue_outputs[i], blue_cost_scores[i], blue_reward_scores[i])
                table.append(row)

            print('The following analysis is under the preference of the cost/reward model.',)
            cost_red = np.asarray([row[2] for row in table])
            cost_blue = np.asarray([row[5] for row in table])
            reward_red = np.asarray([row[3] for row in table])
            reward_blue = np.asarray([row[6] for row in table])
            print(f'Average cost of {red_name}: {cost_red.mean()}')
            print(f'Average reward of {red_name}: {reward_red.mean()}')
            print(f'Average cost of {blue_name}: {cost_blue.mean()}')
            print(f'Average reward of {blue_name}: {reward_blue.mean()}')

            table_output_dir = os.path.join(
                args.save_dir,
                folder_name
            )
            os.makedirs(table_output_dir, exist_ok=True)
            output_file = os.path.join(table_output_dir, f'top{topk}_{save_name}_table.csv')
            with open(output_file, mode='w', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(columns)
                writer.writerows(table)
                
            red_cost.append(cost_red.mean())
            blue_cost.append(cost_blue.mean())
            red_reward.append(reward_red.mean())
            blue_reward.append(reward_blue.mean())
            red_cost_std.append(cost_red.std())
            blue_cost_std.append(cost_blue.std())
            red_reward_std.append(reward_red.std())
            blue_reward_std.append(reward_blue.std())
            if not is_manipulated:
                break

        try:
            if is_manipulated:
                df = pd.DataFrame({'topk': x,
                                   f'{red_name}_cost_mean': red_cost,
                                   f'{blue_name}_cost_mean': blue_cost,
                                   f'{red_name}_cost_std': red_cost_std,
                                   f'{blue_name}_cost_std': blue_cost_std,
                                   f'{red_name}_reward_mean': red_reward,
                                   f'{blue_name}_reward_mean': blue_reward,
                                   f'{red_name}_reward_std': red_reward_std,
                                   f'{blue_name}_reward_std': blue_reward_std
                                   }
                                  )
                df.to_csv(os.path.join(table_output_dir, f'{save_name}.csv'))
        except Exception as e:
            print(e)

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to evaluate.",
    )
    parser.add_argument(
        "--topk_ablate",
        type=int,
        default=1,
        nargs='+',
        help="Number of top different neurons to ablate.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Dataset to evaluate.",
    )
    parser.add_argument(
        "--save_dir",
        type=str, 
        default="results/alpaca_farm")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="If specified, we will load the model to generate the predictions.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="If specified, we will load the tokenizer from here.",
    )
    parser.add_argument(
        "--cost_model_name_or_path",
        type=str,
        default=None,
        help="If specified, we will load the model to generate cost scores for predictions.",
    )
    parser.add_argument(
        "--cost_tokenizer_name_or_path",
        type=str,
        default=None,
        help="If specified, we will load the tokenizer from here.",
    )
    parser.add_argument(
        "--reward_model_name_or_path",
        type=str,
        default=None,
        help="If specified, we will load the model to generate cost scores for predictions.",
    )
    parser.add_argument(
        "--reward_tokenizer_name_or_path",
        type=str,
        default=None,
        help="If specified, we will load the tokenizer from here.",
    )
    parser.add_argument(
        "--openai_engine",
        type=str,
        default=None,
        help="If specified, we will use the OpenAI API to generate the predictions.",
    )
    parser.add_argument(
        "--eval_batch_size", 
        type=int, 
        default=1, 
        help="Batch size for evaluation."
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=256, 
        help="Batch size for evaluation."
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8bit mode, which will reduce memory and speed up inference.",
    )
    parser.add_argument(
        "--gptq",
        action="store_true",
        help="If given, we're evaluating a 4-bit quantized GPTQ model.",
    )
    parser.add_argument(
        "--use_chat_format", 
        action="store_true", 
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--chat_formatting_function", 
        type=str, 
        default="eval.templates.create_prompt_with_tulu_chat_format", 
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="If given, we will use vLLM to generate the predictions - much faster.",
    )
    parser.add_argument(
        "--red_peft_path", 
        nargs='+',
        default=None, 
        help="The folder contains lora checkpoint saved with PeftModel.save_pretrained()."
    )
    parser.add_argument(
        "--blue_peft_path", 
        nargs='+',
        default=None, 
        help="The folder contains lora checkpoint saved with PeftModel.save_pretrained()."
    )
    parser.add_argument(
        "--index_path", 
        type=str, 
        default=None, 
        help="The folder contains lora checkpoint saved with PeftModel.save_pretrained()."
    )
    parser.add_argument(
        "--ignore_index_path", 
        type=str, 
        default=None, 
        help="The folder contains lora checkpoint saved with PeftModel.save_pretrained()."
    )
    parser.add_argument(
        "--intersect_index_path", 
        type=str, 
        default=None, 
        help="The folder contains lora checkpoint saved with PeftModel.save_pretrained()."
    )
    parser.add_argument(
        "--value_path", 
        type=str, 
        default=None, 
        help="The folder contains lora checkpoint saved with PeftModel.save_pretrained()."
    )
    parser.add_argument(
        "--use_random_neurons", 
        action="store_true",
        help="Use random neurons instead of found."
    )
    parser.add_argument(
        "--add_noise", 
        action="store_true",
        help="Add random Gaussian noise to selected neurons."
    )
    parser.add_argument(
        "--patch_mean", 
        action="store_true",
        help="Patch with neuron activation mean."
    )
    parser.add_argument(
        "--patch_zero", 
        action="store_true",
        help="Patch with 0."
    )
    parser.add_argument(
        "--patch_flip", 
        action="store_true",
        help="Flip neuron activation."
    )
    parser.add_argument(
        "--guided_generation", 
        action="store_true",
        help="Guided generation using aligned model activation."
    )
    parser.add_argument(
        "--generation_startswith", 
        type=str, 
        default='', 
        help="Generate completion start with given string."
    )
    parser.add_argument(
        "--sliding_window", 
        type=int, 
        default=None, 
        help="If specified, using the neurons ranked between topk to topk+sliding_window."
    )
    parser.add_argument(
        "--hf_model", 
        action="store_true",
        help="Use implementation from huggingface transformers."
    )

    args = parser.parse_args()

    # model_name_or_path and openai_engine cannot be both None or both not None.
    assert (args.model_name_or_path is None) != (args.openai_engine is None), "Either model_name_or_path or openai_engine should be specified."
    main(args)