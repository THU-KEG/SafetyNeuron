from collections import Counter
import json
import argparse
from tqdm import tqdm
import torch
from eval.utils import load_hf_lm_and_tokenizer



def main(args):

    model, tokenizer = load_hf_lm_and_tokenizer(
        model_name_or_path=args.model_name_or_path, 
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        load_in_8bit=args.load_in_8bit, 
        device_map="auto",
        use_fast_tokenizer=not args.use_slow_tokenizer,
    )

    words = {}
    if args.index_path:
        _, index, *_ = torch.load(args.index_path)
    for layer, idx in tqdm(index[:args.topk_neuron]):
        weight = model.get_submodule(f'model.layers.{layer}.mlp.down_proj').weight[:, idx]
        breakpoint()
        logit = model.lm_head(model.model.norm(weight))
        word_ids = logit.topk(k=args.topk_token)[1]
        print(f'{layer}.{idx}', word_ids)
        words[f'{layer}.{idx}'] = tokenizer.batch_decode(word_ids)

    # word_freq = Counter(words)
    # print(word_freq.most_common(100))
    with open('./results/top_tokens.json', 'w') as f:
        json.dump(words, f, indent=4)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
        "--index_path", 
        type=str, 
        default=None, 
        help="The folder contains lora checkpoint saved with PeftModel.save_pretrained()."
    )
    parser.add_argument(
        "--topk_neuron", 
        type=int, 
        default=5000, 
        help="The folder contains lora checkpoint saved with PeftModel.save_pretrained()."
    )
    parser.add_argument(
        "--topk_token", 
        type=int, 
        default=5000, 
        help="The folder contains lora checkpoint saved with PeftModel.save_pretrained()."
    )
    args = parser.parse_args()

    assert args.model_name_or_path is not None, "model_name_or_path should be specified."
    main(args)
