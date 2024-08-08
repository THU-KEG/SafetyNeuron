#!/usr/bin/env python
# coding=utf-8
'''
This script is used to reformat the downloaded datasets into the format that can be used by the model.
Here we use jsonl for the converted data. Each line in the jsonl file is a json object formatted as follows:
{
    "dataset": "dataset_name",
    "id": "unique_id",
    "messages": [
        {"role": "system", "content": "message_text"}, # optional
        {"role": "user", "content": "message_text"},
        {"role": "assistant", "content": "message_text"},
        {"role": "user", "content": "message_text"},
        {"role": "assistant", "content": "message_text"},
        ...
    ],
}
'''
import json
import random
import re
import os
import argparse
import pandas as pd
import jsonlines

from instruction_encode_templates import encode_instruction_example, encode_few_shot_example

def convert_to_messages(text):
    messages = []
    for pair in text.split('Human:')[1:]:
        temp = pair.split('Assistant:')
        try:
            user, assist = temp[0], temp[1]
        except:
            user, assist = temp[0], ' '
        messages.append({
            "role": "user",
            "content": user
        })
        messages.append({
            "role": "assistant",
            "content": assist
        })
        break
    return messages

def create_prompt_with_tulu_chat_format(messages, bos="<s>", eos="</s>", add_bos=False):
    formatted_text = ""
    for message in messages:
        if message["role"] == "system":
            formatted_text += "<|system|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "user":
            formatted_text += "<|user|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "assistant":
            formatted_text += "<|assistant|>\n" + message["content"].strip() + "\n"
        else:
            raise ValueError(
                "Tulu chat template only supports 'system', 'user' and 'assistant' roles. Invalid role: {}.".format(message["role"])
                )
    formatted_text = bos + formatted_text if add_bos else formatted_text
    return formatted_text

def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = "<|assistant|>\n"
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[len("<|user|>\n"): search_term_idx]

def convert_hh_rlhf_harmless_data(data_dir, output_dir, num_examples=None):
    os.makedirs(output_dir, exist_ok=True)
    for split in ['train', 'test']:
        examples = []
        with open(os.path.join(data_dir, f"{split}.jsonl"), "r") as fin:
            for line in fin:
                examples.append(json.loads(line))
        if num_examples:
            examples = random.sample(examples, k=num_examples)
        output_path = os.path.join(output_dir, f"hh_harmless_data_{split}.jsonl")
        with open(output_path, "w") as fout:
            for idx, example in enumerate(examples):
                chosen = convert_to_messages(example["chosen"])
                formatted_chosen = create_prompt_with_tulu_chat_format(chosen)
                prompt = extract_anthropic_prompt(formatted_chosen)
                fout.write(json.dumps({
                    "dataset": "hh_harmless",
                    "id": f"harmless_{idx}",
                    "prompt": prompt,
                }) + "\n")

def convert_self_instruct_data(data_dir, output_dir, number_examples=None):
    os.makedirs(output_dir, exist_ok=True)
    examples = []
    with open(os.path.join(data_dir, "all_instances_82K.jsonl"), "r") as fin:
        for line in fin:
            examples.append(json.loads(line))
    if number_examples:
        examples = random.sample(examples, k=number_examples)
    output_path = os.path.join(output_dir, "self_instruct_data.jsonl")
    with open(output_path, "w") as fout:
        for idx, example in enumerate(examples):
            encoded_example = encode_instruction_example(
                instruction=example["instruction"], 
                input=example["input"], 
                output=example["output"],
                random_template=True,
                eos_token=None
            )
            fout.write(json.dumps({
                "dataset": "self_instruct",
                "id": f"self_instruct_{idx}",
                "prompt": encoded_example["prompt"],
                "completion": encoded_example["completion"] 
            }) + "\n")

def convert_hh_rlhf_helpful_data(data_dir, output_dir, num_examples=None):
    os.makedirs(output_dir, exist_ok=True)
    for split in ['test']:
        examples = []
        with open(os.path.join(data_dir, f"{split}.jsonl"), "r") as fin:
            for line in fin:
                examples.append(json.loads(line))
        if num_examples:
            examples = random.sample(examples, k=num_examples)
        output_path = os.path.join(output_dir, f"hh_helpful_data_{split}.jsonl")
        with open(output_path, "w") as fout:
            for idx, example in enumerate(examples):
                chosen = convert_to_messages(example["chosen"])
                formatted_chosen = create_prompt_with_tulu_chat_format(chosen)
                prompt = extract_anthropic_prompt(formatted_chosen)
                fout.write(json.dumps({
                    "dataset": "hh_helpful",
                    "id": f"helpful_{idx}",
                    "prompt": prompt,
                }) + "\n")

def convert_real_toxicity_prompts_data(data_dir, output_dir, num_examples=30000):
    os.makedirs(output_dir, exist_ok=True)
    examples = []
    with open(os.path.join(data_dir, "prompts.jsonl"), "r") as fin:
        for line in fin:
            examples.append(json.loads(line))
    if num_examples:
        examples = random.sample(examples, k=num_examples)
    output_path = os.path.join(output_dir, f"real_toxicity_prompts.jsonl")
    with open(output_path, "w") as fout:
        for idx, example in enumerate(examples):
            prompt = example['prompt']['text']
            fout.write(json.dumps({
                "dataset": "real_toxicity_prompts",
                "id": f"real_toxicity_prompts_{idx}",
                "prompt": prompt,
            }) + "\n")
                
def convert_red_team_attempts_data(data_dir, output_dir, num_examples=None):
    os.makedirs(output_dir, exist_ok=True)
    examples = []
    with open(os.path.join(data_dir, "red_team_attempts.jsonl"), "r") as fin:
        examples = json.load(fin)
    if num_examples:
        examples = random.sample(examples, k=num_examples)
    output_path = os.path.join(output_dir, f"red_team_attempts.jsonl")
    with open(output_path, "w") as fout:
        for idx, example in enumerate(examples):
            chosen = convert_to_messages(example["transcript"])
            formatted_chosen = create_prompt_with_tulu_chat_format(chosen)
            prompt = extract_anthropic_prompt(formatted_chosen)
            fout.write(json.dumps({
                "dataset": "red_team_attempts",
                "id": f"red_team_attempts_{idx}",
                "prompt": prompt,
            }) + "\n")

def convert_harmbench_data(data_dir, output_dir, num_examples=None):
    os.makedirs(output_dir, exist_ok=True)
    examples = []
    df = pd.read_csv(os.path.join(data_dir, "harmbench_behaviors_text_all.csv"))
    for example in df['Behavior']:
        examples.append(example)
    if num_examples:
        examples = random.sample(examples, k=num_examples)
    output_path = os.path.join(output_dir, f"harmbench.jsonl")
    with open(output_path, "w") as fout:
        for idx, example in enumerate(examples):
            fout.write(json.dumps({
                "dataset": "harmbench",
                "id": f"harmbench_{idx}",
                "prompt": example,
            }) + "\n")

def convert_jailbreak_llms_data(data_dir, output_dir, num_examples=None):
    os.makedirs(output_dir, exist_ok=True)
    examples = []
    df = pd.read_csv(os.path.join(data_dir, "questions.csv"))
    for example in df['question']:
        examples.append(example)
    if num_examples:
        examples = random.sample(examples, k=num_examples)
    output_path = os.path.join(output_dir, f"jailbreak_llms.jsonl")
    with open(output_path, "w") as fout:
        for idx, example in enumerate(examples):
            fout.write(json.dumps({
                "dataset": "jailbreak_llms",
                "id": f"jailbreak_llms_{idx}",
                "prompt": example,
            }) + "\n")

def convert_lima_data(data_dir, output_dir, num_examples=None):
    os.makedirs(output_dir, exist_ok=True)
    examples = []
    with open(os.path.join(data_dir, "train.jsonl"), "r") as fin:
        for line in fin:
            examples.append(json.loads(line))
    if num_examples:
        examples = random.sample(examples, k=num_examples)
    output_path = os.path.join(output_dir, f"lima.jsonl")
    with open(output_path, "w") as fout:
        for idx, example in enumerate(examples):
            fout.write(json.dumps({
                "dataset": "lima",
                "id": f"lima_{idx}",
                "prompt": example['conversations'][0],
            }) + "\n")        
            
            
if __name__ == "__main__":
    # all supported datasets    
    supported_datasets = []
    all_funcs = [func_name for func_name in globals() if callable(globals()[func_name])]
    for func_name in all_funcs:
        if re.match(r"convert_.+_data", func_name):
            supported_datasets.append(func_name[8:-5])

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--raw_data_dir", 
        type=str, 
        default="data/downloads"
    )
    arg_parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/processed"
    )
    arg_parser.add_argument(
        "--dataset", 
        type=str, 
        nargs="+",
        choices=supported_datasets+["tulu_v1", "tulu_v2"],
        default=supported_datasets+["tulu_v1", "tulu_v2"]
    )
    arg_parser.add_argument(
        "--seed", 
        type=int, 
        default=42
    )
    args = arg_parser.parse_args()
    random.seed(args.seed)

    # get the subfolder names in raw_data_dir
    subfolders = [f for f in os.listdir(args.raw_data_dir) if os.path.isdir(os.path.join(args.raw_data_dir, f))]

    for dataset in args.dataset:
        print(f"Processing {dataset} data with default configurations...")
        globals()[f"convert_{dataset}_data"](os.path.join(args.raw_data_dir, dataset), os.path.join(args.output_dir, dataset))