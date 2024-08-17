import argparse
import os
import re
import json
import tqdm
import glob
import torch
import random
import evaluate
from eval.utils import (
    load_hf_lm_and_tokenizer,
    load_hooked_lm_and_tokenizer,
    generate_completions,
    query_openai_chat_model,
    dynamic_import_function,
)

exact_match = evaluate.load("exact_match")
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
    random.seed(42)

    all_tasks = {}
    task_files = glob.glob(os.path.join(args.data_dir, "bbh", "*.json"))
    for task_file in tqdm.tqdm(task_files, desc="Loading tasks"):
        with open(task_file, "r") as f:
            task_name = os.path.basename(task_file).split(".")[0]
            all_tasks[task_name] = json.load(f)["examples"]
            if args.max_num_examples_per_task:
                all_tasks[task_name] = random.sample(all_tasks[task_name], args.max_num_examples_per_task)

    all_prompts = {}
    cot_prompt_files = glob.glob(os.path.join(args.data_dir, "cot-prompts", "*.txt"))
    for cot_prompt_file in tqdm.tqdm(cot_prompt_files, desc="Loading prompts"):
        with open(cot_prompt_file, "r") as f:
            task_name = os.path.basename(cot_prompt_file).split(".")[0]
            task_prompt = "".join(f.readlines()[2:])
            if args.no_cot:
                prompt_fields = task_prompt.split("\n\n")
                new_prompt_fields = []
                for prompt_field in prompt_fields:
                    if prompt_field.startswith("Q:"):
                        assert "So the answer is" in prompt_field, f"`So the answer is` not found in prompt field of {task_name}.txt."
                        assert "\nA:" in prompt_field, "`\nA:` not found in prompt field."
                        answer = prompt_field.split("So the answer is")[-1].strip()
                        question = prompt_field.split("\nA:")[0].strip()
                        new_prompt_fields.append(question + "\nA: " + answer)
                    else:
                        new_prompt_fields.append(prompt_field)
                task_prompt = "\n\n".join(new_prompt_fields)
            all_prompts[task_name] = task_prompt

    assert set(all_tasks.keys()) == set(all_prompts.keys()), "task names in task data and task prompts are not the same."

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "predictions"), exist_ok=True)

    # Load model if not using OpenAI API
    if args.model_name_or_path:
        load_fn = load_hooked_lm_and_tokenizer if args.hooked else load_hf_lm_and_tokenizer
        print("Loading model and tokenizer with huggingface...")
        model, tokenizer = load_fn(
            model_name_or_path=args.model_name_or_path, 
            tokenizer_name_or_path=args.tokenizer_name_or_path, 
            load_in_8bit=args.load_in_8bit, 
            device_map="auto",
            use_fast_tokenizer=not args.use_slow_tokenizer,
            peft_name_or_path=args.red_peft_path
        )
        if args.blue_peft_path and args.index_path:
            guided_model, _ = load_fn(
                model_name_or_path=args.model_name_or_path, 
                tokenizer_name_or_path=args.tokenizer_name_or_path, 
                load_in_8bit=args.load_in_8bit, 
                device_map="auto",
                use_fast_tokenizer=not args.use_slow_tokenizer,
                peft_name_or_path=args.blue_peft_path
            )
            _, index, *_ = torch.load(args.index_path)
            topk_index = index[:20000]
        else:
            guided_model = None
            topk_index = None
    performance = {}
    for task_name in tqdm.tqdm(all_tasks.keys(), desc="Evaluating"):
        task_examples = all_tasks[task_name]
        task_prompt = all_prompts[task_name]
        if args.model_name_or_path:
            # prepare prompts
            prompts = []
            chat_formatting_function = dynamic_import_function(args.chat_formatting_function) if args.use_chat_format else None
            for example in task_examples:
                prompt = task_prompt.strip() + "\n\nQ: " + example["input"]
                if args.use_chat_format:
                    messages = [{"role": "user", "content": prompt}]
                    prompt = chat_formatting_function(messages, add_bos=False)
                    if prompt[-1] in ["\n", " "]:
                        prompt += "A:"
                    else:
                        prompt += " A:"
                else:
                    prompt += "\nA:"
                prompts.append(prompt)

            stop_sequence = tokenizer.encode("\n\n", add_special_tokens=False)[-2:] # get the last token because the tokenizer may add space tokens at the start.
            outputs = generate_completions(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=128,
                batch_size=args.eval_batch_size if args.eval_batch_size else 1,
                stop_id_sequences=[[stop_sequence]],
                do_sample=False,
                guided_model=guided_model,
                index=topk_index,
                hook_fn=layer_patch_hook    
            )
        else:
            instances = []
            for i, example in enumerate(task_examples):
                prompt = task_prompt.strip() + "\n\nQ: " + example["input"] + "\nA:"
                instances.append({
                    "id": example["id"] if "id" in example else i,
                    "prompt": prompt,
                })
            results = query_openai_chat_model(
                engine=args.openai_engine,
                instances=instances,
                batch_size=args.eval_batch_size if args.eval_batch_size else 10,
                output_path=os.path.join(args.save_dir, "predictions", f"{task_name}_openai_prediction_cache.jsonl"),
            )
            outputs = [result["output"] for result in results]

        targets = [example["target"] for example in task_examples]
        predictions = []
        for example, output in zip(task_examples, outputs):
            example["raw_output"] = output
            
            # extract the first answer after `So the answer is` and before the next period.
            # if there is no such answer, we will just use the raw output.
            extracted_answer = re.search(r"So the answer is (.*?)\.", output)
            if extracted_answer:
                prediction = extracted_answer.group(1).strip()
            else:
                # only keep the first part of the output - this is mainly for vanilla language models.
                output = output.strip().split("\n\n")[0].strip()
                prediction = output.strip()

            example["prediction"] = prediction
            predictions.append(prediction)
        
        with open(os.path.join(args.save_dir, "predictions", f"{task_name}.jsonl"), "w") as fout:
            for example in task_examples:
                fout.write(json.dumps(example) + "\n")        

        assert len(predictions) == len(targets), "number of predictions and targets are not the same."
        performance[task_name] = exact_match.compute(predictions=predictions, references=targets, ignore_case=True, ignore_punctuation=True)["exact_match"]

        print(f"Task {task_name} - EM: {performance[task_name]}")

    # save the performance
    with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
        performance["average_exact_match"] = sum(performance.values()) / len(performance)
        print(f"Average EM: {performance['average_exact_match']}")
        json.dump(performance, fout, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="data/bbh"
    )
    parser.add_argument(
        "--save_dir", 
        type=str, 
        default="results/bbh"
    )
    parser.add_argument(
        "--model_name_or_path", 
        type=str, 
        default=None, 
        help="if specified, we will load the model to generate the predictions."
    )
    parser.add_argument(
        "--tokenizer_name_or_path", 
        type=str, 
        default=None, 
        help="if specified, we will load the tokenizer from here."
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )
    parser.add_argument(
        "--openai_engine", 
        type=str, 
        default=None, 
        help="if specified, we will use the OpenAI API to generate the predictions."
    )
    parser.add_argument(
        "--no_cot", 
        action="store_true", 
        help="if specified, chain of thoughts will be removed from the prompts."
    )
    parser.add_argument(
        "--max_num_examples_per_task", 
        type=int, 
        default=None, 
        help="maximum number of examples to evaluate per task."
    )
    parser.add_argument(
        "--eval_batch_size", 
        type=int, 
        default=1, 
        help="batch size for evaluation."
    )
    parser.add_argument(
        "--load_in_8bit", 
        action="store_true", 
        help="load model in 8bit mode, which will reduce memory and speed up inference."
    )
    parser.add_argument(
        "--gptq", 
        action="store_true", 
        help="If given, we're evaluating a 4-bit quantized GPTQ model."
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true", 
        help="If given, we will use the vllm library, which will likely increase the inference throughput."
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
        "--hooked", 
        action="store_true", 
        help="If given, we will use the hooked model."
    )
    args = parser.parse_args()

    # model_name_or_path and openai_engine cannot be both None or both not None.
    assert (args.model_name_or_path is None) != (args.openai_engine is None), "Either model_name_or_path or openai_engine should be specified."
    main(args)
