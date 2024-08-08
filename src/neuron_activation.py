import os
import argparse
import json
from collections import Counter
import random

import torch
import datasets

from src.activation_processor import NeuronActivation
from src.utils import seed_torch
from eval.templates import create_prompt_with_tulu_chat_format

def main(args):
    eval_data = datasets.load_dataset('json', data_files=args.dataset)["train"]["prompt"]
    if args.num_samples < 0:
        args.num_samples = len(eval_data)
    print(f"using {args.num_samples} samples")

    prompts = []
    for example in eval_data[:args.num_samples]:
        prompt = example
        messages = [{"role": "user", "content": prompt}]
        prompt = create_prompt_with_tulu_chat_format(messages, add_bos=False)
        prompts.append(prompt)

    _, index, *_ = torch.load(args.index_path)
    config = json.load(open(os.path.join(args.model_name_or_path, 'config.json'), 'r'))
    indexes = []
    save_names = []
    for topk in args.topk:
        topk_index = index[:topk]
        if args.use_random_neurons:
            for seed in args.seed:
                seed_torch(seed)
                if args.last_layer_neurons:
                    topk_index = torch.tensor([[config['num_hidden_layers']-1, neuron] for neuron in random.sample(range(config['intermediate_size']), topk)])
                elif args.random_neurons_everywhere:
                    topk_index = torch.tensor([[neuron // config['intermediate_size'], neuron % config['intermediate_size']] for neuron in random.sample(range(config['intermediate_size'] * config['num_hidden_layers']), topk)])
                else:
                    counts = Counter(topk_index[:, 0].tolist())
                    topk_index = []
                    for layer, num in counts.items():
                        neurons = random.sample(range(config['intermediate_size']), num)
                        topk_index += [[layer, neuron] for neuron in neurons]
                    topk_index = torch.tensor(topk_index)
                indexes.append(topk_index)
                save_names.append((topk, seed))
        else:
            indexes.append(topk_index)
            save_names.append(topk)
    
    neuron_activation = NeuronActivation(
        args.model_name_or_path,
        args.cost_model_name_or_path,
        args.peft_path,
        indexes,
        batchsize=args.eval_batch_size,
        device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto"
    )
    activation, _, target = neuron_activation.create_dataset(prompts)
    if args.use_random_neurons:
        for i, (topk, seed) in enumerate(save_names):
            torch.save((activation[i], target), f"{args.output_dir}/{args.output_filename}_top{topk}_seed{seed}.pt")
    else:
        for i, topk in enumerate(save_names):
            torch.save((activation[i], target), f"{args.output_dir}/{args.output_filename}_top{topk}.pt")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to evaluate.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Dataset to evaluate.",
    )
    parser.add_argument(
        "--output_dir",
        type=str, 
        default="hooked_llama/data"
    )
    parser.add_argument(
        "--output_filename",
        type=str, 
        default="activation"
    )
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
        "--peft_path", 
        type=str, 
        nargs='+',
        default=None, 
        help="The folder contains peft checkpoint saved with PeftModel.save_pretrained()."
    )
    parser.add_argument(
        "--eval_batch_size", 
        type=int, 
        default=1, 
        help="Batch size for evaluation."
    )
    parser.add_argument(
        "--index_path", 
        type=str, 
        default=None, 
        help="path to safety neuron ranks."
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=1,
        nargs='+',
        help="Number of top neurons to cache.",
    )
    parser.add_argument(
        "--use_random_neurons", 
        action="store_true",
        help="Use random neurons instead of found."
    )
    parser.add_argument(
        "--random_neurons_everywhere", 
        action="store_true",
        help="Use random neurons instead of found."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        nargs='+',
        help="Random seed.",
    )
    parser.add_argument(
        "--last_layer_neurons", 
        action="store_true",
        help="Use random neurons in last layer."
    )

    args = parser.parse_args()
    main(args)

