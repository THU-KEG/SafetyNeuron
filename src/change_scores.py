import os
import argparse

import torch
import datasets

from src.utils import seed_torch
from src.activation_processor import ActivationContrasting
from eval.templates import create_prompt_with_tulu_chat_format


def main(args):
    
    seed_torch(42)
    eval_data = datasets.load_dataset('json', data_files=args.dataset)["train"]["prompt"]
    if args.num_samples > 0:
        eval_data = eval_data[:args.num_samples]
        
    prompts = []
    for example in eval_data:
        prompt = example.strip() 
        messages = [{"role": "user", "content": prompt}]
        prompt = create_prompt_with_tulu_chat_format(messages, add_bos=False)
        prompts.append(prompt+args.generation_startswith)

    names_filter = lambda name: name.endswith('hook_post') and '31' not in name
    ac = ActivationContrasting(
        args.model_name_or_path,
        args.first_peft_path,
        args.second_peft_path,
        batchsize=args.eval_batch_size,
        max_new_tokens=args.max_new_tokens,
        device_map='balanced_low_0'
    )
    change_scores, neuron_ranks, first_mean, first_std, second_mean, second_std = ac.compute_change_scores(prompts, names_filter, args.token_type)
    output_dir = os.path.dirname(args.output_file)
    os.makedirs(output_dir, exist_ok=True)
    torch.save((change_scores.cpu(), neuron_ranks.cpu(), first_mean.cpu(), first_std.cpu(), second_mean.cpu(), second_std.cpu()), args.output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute change scores via generation-time activation contrasting.")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Max new tokens in generation.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=-1,
        help="Number of samples to evaluate.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Dataset to evaluate.",
    )
    parser.add_argument(
        "--output_file",
        type=str, 
        default="../data/default.pt"
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
        "--first_peft_path", 
        nargs='+', 
        default=None, 
        help="The folder contains peft checkpoint saved with PeftModel.save_pretrained()."
    )
    parser.add_argument(
        "--second_peft_path", 
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
        "--token_type", 
        type=str, 
        default='completion', 
        choices=['prompt', 'prompt_last', 'completion'],
        help="Compute change scores from which token position."
    )
    parser.add_argument(
        "--generation_startswith", 
        type=str, 
        default='', 
        help="Generation start with given prefix."
    )

    args = parser.parse_args()
    main(args)

