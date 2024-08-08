import os
import argparse
import json
from collections import defaultdict
import random
from tqdm import tqdm
import torch
import datasets
from sklearn.linear_model import LogisticRegression

from src.utils import seed_torch
from src.activation_processor import NeuronActivation
from eval.templates import create_prompt_with_tulu_chat_format

def get_clf(peft, dataset, topk=1500):
    data_train = f'/data1/cjh/Alignment/output/activations/harmless_prediction/{dataset}_5000_{peft}_top{topk}.pt'
    X_train, y_train = torch.load(data_train)
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    return clf

def main(args):
    seed_torch()
    dataset_map = {'jailbreak_llms': 'jailbreak_llms', 'hh_rlhf_harmless': 'hh_harmless', 'BeaverTails': 'BeaverTails', 'harmbench': 'harmbench', 'red_team_attempts': 'red_team_attempts'}
    clf = get_clf(os.path.basename(args.peft_path[-1]), dataset_map[os.path.basename(os.path.dirname(args.train_dataset))], args.topk[-1])
    _, index, *_ = torch.load(args.index_path)
    try:
        with open(os.path.join(args.output_dir, args.output_filename), 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        results = defaultdict(list)
        
    for eval_dataset in args.eval_datasets:
        eval_data = datasets.load_dataset('json', data_files=eval_dataset)["train"]["prompt"]
        if args.num_samples < 0:
            args.num_samples = len(eval_data)
        print(f"using {args.num_samples} samples for {os.path.basename(os.path.dirname(eval_dataset))}")
        prompts = []
        for example in eval_data:
            messages = [{"role": "user", "content": example}]
            prompt = create_prompt_with_tulu_chat_format(messages, add_bos=False)
            prompts.append(prompt)
        prompts = random.sample(prompts, args.num_samples)
        neuron_activation = NeuronActivation(
            args.model_name_or_path,
            args.cost_model_name_or_path,
            args.peft_path,
            [index[:args.topk[-1]]],
            batchsize=args.eval_batch_size,
            device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto"
        )
        activation, cost_scores, _ = neuron_activation.create_dataset(prompts)
        try:
            preds = clf.predict(activation[0])
            guard_cost_scores = torch.masked_select(torch.tensor(cost_scores), ~torch.tensor(preds, dtype=torch.bool))
        except Exception as e:
            print(e)
        
        results['Model'].append(os.path.basename(args.model_name_or_path))
        results['Type'].append('DPO' if 'dpo' in args.peft_path[-1] else 'SFT')
        results['Dataset'].append(os.path.basename(os.path.dirname(eval_dataset)))
        results['mean'].append(cost_scores.mean().item())
        results['std'].append(cost_scores.std().item())

        results['Model'].append(os.path.basename(args.model_name_or_path))
        results['Type'].append('DPO+Guard' if 'dpo' in args.peft_path[-1] else 'SFT+Guard')
        results['Dataset'].append(os.path.basename(os.path.dirname(eval_dataset)))
        results['mean'].append(guard_cost_scores.mean().item())
        results['std'].append(guard_cost_scores.std().item())
        
    with open(os.path.join(args.output_dir, args.output_filename), 'w') as f:
        results = json.dump(results, f)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to evaluate.",
    )
    parser.add_argument(
        "--train_dataset",
        type=str,
        default="",
        help="Dataset to train classifier.",
    )
    parser.add_argument(
        "--eval_datasets",
        type=str,
        default="",
        nargs='+',
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

    args = parser.parse_args()
    main(args)