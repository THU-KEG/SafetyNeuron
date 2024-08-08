import os
import random
import argparse
from collections import defaultdict, Counter
from functools import partial

from tqdm import tqdm
import torch
import datasets

from eval.utils import load_hooked_lm_and_tokenizer
from src.utils import get_act_name
from src.utils import seed_torch

Activation_cache = defaultdict(list)

def layer_patch_hook(value, hook, neurons, patched_values):
    try:
        if not isinstance(patched_values, torch.Tensor):
            patched_values = torch.tensor(patched_values)
        patched_values = patched_values.to(value)
        value[..., neurons] = patched_values
    except Exception as e:
        print(f'Error in hook {hook}', e)
    return value

def main(args):
    seed_torch()

    test = datasets.load_dataset('parquet', data_files={'train': '/data1/cjh/Alignment/data/eval/wikitext-2/test-00000-of-00001.parquet'})['train']
    breakpoint()
        
    if args.model_name_or_path is not None:
        red_model, tokenizer = load_hooked_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            tokenizer_name_or_path=args.tokenizer_name_or_path if args.tokenizer_name_or_path is not None else args.model_name_or_path,
            load_in_8bit=args.load_in_8bit,
            device_map="auto",
            peft_name_or_path=args.red_peft_path
        )
        blue_model, _ = load_hooked_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            tokenizer_name_or_path=args.tokenizer_name_or_path if args.tokenizer_name_or_path is not None else args.model_name_or_path,
            load_in_8bit=args.load_in_8bit,
            device_map="auto",
            peft_name_or_path=args.blue_peft_path
        )
    # breakpoint()
    max_length = 4096
    stride = 4096
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    red_name = os.path.basename(args.red_peft_path[-1]) if args.red_peft_path else os.path.basename(args.model_name_or_path)
    blue_name = os.path.basename(args.blue_peft_path[-1]) if args.blue_peft_path else os.path.basename(args.model_name_or_path)

    names_filter = lambda name: name.endswith('hook_post')
    if args.index_path:
        _, index, *_ = torch.load(args.index_path)
    for topk in args.topk_ablate:
        print(f"running with {topk} neurons patched")
        if args.sliding_window is not None:
            topk_index = index[topk-args.sliding_window:topk]
        else:
            topk_index = index[:topk]
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
        layers = defaultdict(list)
        for layer, idx in topk_index:
            layers[layer.item()].append(idx)

        red_nlls, blue_nlls, blue_patch_red_nlls = [], [], []
        prev_end_loc = 0
        for begin_loc in tqdm(range(0, seq_len, stride)):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(blue_model.device)
            if blue_model.config.model_type == 'gemma':
                input_ids[:, 0] = 2 # bos token is necessary for gemma to get a valid ppl
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs, cache = blue_model.run_with_cache(input_ids=input_ids, labels=target_ids, names_filter=names_filter)
                blue_nlls.append(outputs.loss)

                for layer, neurons in layers.items():
                    layer_cache = cache['post', layer]
                    neurons = torch.tensor(neurons) # pass tensor as parameter rather than list of tensor will speed up significantly
                    partial_hook_fn = partial(layer_patch_hook, neurons=neurons, patched_values=layer_cache[..., neurons])
                    red_model.add_perma_hook(name=get_act_name('post', layer), hook=partial_hook_fn)
                del cache
                torch.cuda.empty_cache()
                outputs = red_model(input_ids=input_ids, labels=target_ids)
                blue_patch_red_nlls.append(outputs.loss)
                
                red_model.reset_hooks(including_permanent=True)
                outputs = red_model(input_ids=input_ids, labels=target_ids)
                red_nlls.append(outputs.loss)
                
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        blue_ppl = torch.exp(torch.stack(blue_nlls).mean())
        red_ppl = torch.exp(torch.stack(red_nlls).mean())
        blue_patch_red_ppl = torch.exp(torch.stack(blue_patch_red_nlls).mean())
        print(f'{red_name} PPL = ', red_ppl.item())
        print(f'{blue_name} PPL = ', blue_ppl.item())
        print(f'{blue_name} patch {red_name} PPL = ', blue_patch_red_ppl.item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--topk_ablate",
        type=int,
        default=1,
        nargs='+',
        help="Number of top different neurons to ablate.",
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
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8bit mode, which will reduce memory and speed up inference.",
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
        "--use_random_neurons", 
        action="store_true",
        help="Use random neurons instead of found."
    )
    parser.add_argument(
        "--sliding_window", 
        type=int, 
        default=None, 
        help="If specified, using the neurons ranked between topk to topk+sliding_window."
    )
    args = parser.parse_args()

    main(args)