import torch
import tqdm
import json
import time
import asyncio
import os
from importlib import import_module
from collections import defaultdict
from functools import partial
from transformers import StoppingCriteria

from training.finetune import encode_with_prompt_completion_format
from eval.dispatch_openai_requests import dispatch_openai_chat_requesets, dispatch_openai_prompt_requesets
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft.peft_model import PeftModel
from src.models.HookedLlama import HookedLlamaForCausalLM
from src.models.HookedMistral import HookedMistralForCausalLM
from src.models.HookedGemma import HookedGemmaForCausalLM
from src.utils import get_act_name
from eval.arena.models.llama_modelling import LlamaModelForScore
from eval.arena.models.modeling_llama_rm import LlamaRewardModel

AUTO_MODEL_MAPPING = {
    'llama': HookedLlamaForCausalLM,
    'mistral': HookedMistralForCausalLM,
    'gemma': HookedGemmaForCausalLM
}


class KeyWordsCriteria(StoppingCriteria):
    def __init__(self, stop_id_sequences):
        assert isinstance(stop_id_sequences[0], list), "stop_id_sequences should be a list of list of ids"
        self.stop_sequences = stop_id_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            sequence_should_be_stopped = False
            for stop_sequence in self.stop_sequences:
                if input_ids[i][-len(stop_sequence):].tolist() == stop_sequence:
                    sequence_should_be_stopped = True
                    break
            sequences_should_be_stopped.append(sequence_should_be_stopped)
        return torch.tensor(sequences_should_be_stopped, device=input_ids.device)
    
    
@torch.no_grad()
def generate_completions(model, tokenizer, prompts, batch_size=1, stop_id_sequences=None, add_special_tokens=True, disable_tqdm=False, **generation_kwargs):
    generations = []
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Generating Completions")

    num_return_sequences = generation_kwargs.get("num_return_sequences", 1)
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=add_special_tokens)
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        if model.device.type == "cuda":
            batch_input_ids = batch_input_ids.cuda()
            attention_mask = attention_mask.cuda()

        try:
            batch_outputs = model.generate(
                input_ids=batch_input_ids,
                attention_mask=attention_mask,
                stopping_criteria=[KeyWordsCriteria(stop_id_sequences)] if stop_id_sequences else None,
                **generation_kwargs
            )
        
            # the stopping criteria is applied at batch level, so if other examples are not stopped, the entire batch will continue to generate.
            # so some outputs still have the stop sequence, which we need to remove.
            if stop_id_sequences:
                for output_idx in range(batch_outputs.shape[0]):
                    for token_idx in range(batch_input_ids.shape[1], batch_outputs.shape[1]):
                        if any(batch_outputs[output_idx, token_idx: token_idx+len(stop_sequence)].tolist() == stop_sequence for stop_sequence in stop_id_sequences):
                            batch_outputs[output_idx, token_idx:] = tokenizer.pad_token_id
                            break

            # remove the prompt from the output
            # we need to re-encode the prompt because we need to make sure the special tokens are treated the same way as in the outputs.
            # we changed our previous way of truncating the output token ids dicrectly because some tokenizer (e.g., llama) won't add space token before the first token.
            # space is important for some tasks (e.g., code completion).
            batch_outputs = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
            batch_prompts = tokenizer.batch_decode(batch_input_ids, skip_special_tokens=True)
            # duplicate the prompts to match the number of return sequences
            batch_prompts = [prompt for prompt in batch_prompts for _ in range(num_return_sequences)]
            batch_generations = [
                output[len(prompt):] for prompt, output in zip(batch_prompts, batch_outputs)
            ]
        except Exception as e:
            print("Error when generating completions for batch:")
            print(batch_prompts)
            print("Error message:")
            print(e)
            print("Use empty string as the completion.")
            batch_generations = [""] * len(batch_prompts) * num_return_sequences

        generations += batch_generations

        # for prompt, generation in zip(batch_prompts, batch_generations):
        #     print("========")
        #     print(prompt)
        #     print("--------")
        #     print(generation)

        if not disable_tqdm:
            progress.update(len(batch_prompts)//num_return_sequences)

    assert len(generations) == len(prompts) * num_return_sequences, "number of generations should be equal to number of prompts * num_return_sequences"
    return generations

@torch.no_grad()
def generate_completions_and_scores(model, tokenizer, prompts, reward_model=None, reward_tokenizer=None, cost_model=None, cost_tokenizer=None, batch_size=1, stop_id_sequences=None, add_special_tokens=True, disable_tqdm=False, return_mask=False, **generation_kwargs):
    
    def get_reward(model, input_ids, attention_mask):
        rewards = []
        if isinstance(model, LlamaRewardModel):
            rewards = model(input_ids=input_ids, attention_mask=attention_mask).tolist()
        if isinstance(model, LlamaModelForScore):
            rewards = model(input_ids=input_ids, attention_mask=attention_mask).end_scores.squeeze(dim=-1).tolist()
        return rewards
        
    generations = []
    cost_scores = []
    reward_scores = []
    masks = []
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Generating Completions")

    num_return_sequences = generation_kwargs.get("num_return_sequences", 1)
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=add_special_tokens)
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        if model.device.type == "cuda":
            batch_input_ids = batch_input_ids.to(model.device)
            attention_mask = attention_mask.to(model.device)

        try:
            # breakpoint()
            batch_outputs = model.generate(
                input_ids=batch_input_ids,
                attention_mask=attention_mask,
                stopping_criteria=[KeyWordsCriteria(stop_id_sequences)] if stop_id_sequences else None,
                **generation_kwargs
            )
            if return_mask:
                masks.append(batch_outputs[:, batch_input_ids.shape[1]:] != 0)
            # the stopping criteria is applied at batch level, so if other examples are not stopped, the entire batch will continue to generate.
            # so some outputs still have the stop sequence, which we need to remove.
            if stop_id_sequences:
                for output_idx in range(batch_outputs.shape[0]):
                    for token_idx in range(batch_input_ids.shape[1], batch_outputs.shape[1]):
                        if any(batch_outputs[output_idx, token_idx: token_idx+len(stop_sequence)].tolist() == stop_sequence for stop_sequence in stop_id_sequences):
                            batch_outputs[output_idx, token_idx:] = tokenizer.pad_token_id
                            break
            
            # remove the prompt from the output
            # we need to re-encode the prompt because we need to make sure the special tokens are treated the same way as in the outputs.
            # we changed our previous way of truncating the output token ids dicrectly because some tokenizer (e.g., llama) won't add space token before the first token.
            # space is important for some tasks (e.g., code completion).
            batch_outputs = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
            batch_outputs = ['<|user|>'.join(output.split('<|user|>')[:2]) for output in batch_outputs]
            batch_prompts = tokenizer.batch_decode(batch_input_ids, skip_special_tokens=True)
            # duplicate the prompts to match the number of return sequences
            batch_prompts = [prompt for prompt in batch_prompts for _ in range(num_return_sequences)]
            batch_generations = [
                output[len(prompt):] for prompt, output in zip(batch_prompts, batch_outputs)
            ]

            # score model output with reward and cost models
            if cost_model and cost_tokenizer:
                cost_tokenized_prompts = cost_tokenizer(batch_outputs, padding="longest", return_tensors="pt", add_special_tokens=add_special_tokens)
                cost_batch_input_ids = cost_tokenized_prompts.input_ids
                cost_attention_mask = cost_tokenized_prompts.attention_mask     
                if cost_model.device.type == "cuda":
                    cost_batch_input_ids = cost_batch_input_ids.to(cost_model.device)
                    cost_attention_mask = cost_attention_mask.to(cost_model.device)
                cost_score = get_reward(cost_model, cost_batch_input_ids, attention_mask=cost_attention_mask)
            if reward_model and reward_tokenizer:
                reward_tokenized_prompts = reward_tokenizer(batch_outputs, padding="longest", return_tensors="pt", add_special_tokens=add_special_tokens)
                reward_batch_input_ids = reward_tokenized_prompts.input_ids
                reward_attention_mask = reward_tokenized_prompts.attention_mask     
                if reward_model.device.type == "cuda":
                    reward_batch_input_ids = reward_batch_input_ids.to(reward_model.device)
                    reward_attention_mask = reward_attention_mask.to(reward_model.device)
                reward_score = get_reward(reward_model, reward_batch_input_ids, attention_mask=reward_attention_mask)

        except Exception as e:
            print("Error when generating completions for batch:")
            print(batch_prompts)
            print("Error message:")
            print(e)
            print("Use empty string as the completion.")
            batch_generations = [""] * len(batch_prompts) * num_return_sequences
            cost_scores = [0] * len(batch_prompts) * num_return_sequences
            reward_scores = [0] * len(batch_prompts) * num_return_sequences

        generations += batch_generations
        if cost_model:
            cost_scores += cost_score
        if reward_model:
            reward_scores += reward_score

        if not disable_tqdm:
            progress.update(len(batch_prompts)//num_return_sequences)

    assert len(generations) == len(prompts) * num_return_sequences, "number of generations should be equal to number of prompts * num_return_sequences"
    if return_mask:
        masks = torch.cat(masks, 0).detach().cpu()
    return generations, cost_scores, reward_scores, masks

@torch.no_grad()
def generate_completions_and_masks(model, tokenizer, prompts, batch_size=1, add_special_tokens=True, disable_tqdm=False, **generation_kwargs):
    outputs = []
    attention_masks = []
    gather_masks = []
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Generating Completions")

    num_return_sequences = generation_kwargs.get("num_return_sequences", 1)
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=add_special_tokens)
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        if model.device.type == "cuda":
            batch_input_ids = batch_input_ids.to(model.device)
            attention_mask = attention_mask.to(model.device)

        try:
            batch_outputs_ids = model.generate(
                input_ids=batch_input_ids,
                attention_mask=attention_mask,
                **generation_kwargs
            )

            # remove the prompt from the output
            # we need to re-encode the prompt because we need to make sure the special tokens are treated the same way as in the outputs.
            # we changed our previous way of truncating the output token ids dicrectly because some tokenizer (e.g., llama) won't add space token before the first token.
            # space is important for some tasks (e.g., code completion).
            batch_outputs = tokenizer.batch_decode(batch_outputs_ids, skip_special_tokens=True)
            batch_prompts = tokenizer.batch_decode(batch_input_ids, skip_special_tokens=True)
            # duplicate the prompts to match the number of return sequences
            batch_prompts = [prompt for prompt in batch_prompts for _ in range(num_return_sequences)]
            batch_generations = [
                output[len(prompt):] for prompt, output in zip(batch_prompts, batch_outputs)
            ]
            # breakpoint()
            batch_ids = []
            batch_attention_mask = []
            batch_gather_mask = []
            max_length = -1
            for prompt, output in zip(batch_prompts, batch_generations):
                prompt_ids = tokenizer(prompt, add_special_tokens=add_special_tokens).input_ids
                output_ids = tokenizer(output, add_special_tokens=False).input_ids
                ids = prompt_ids + output_ids
                max_length = max(len(ids), max_length)
                batch_ids.append(ids)
                batch_attention_mask.append([1]*len(ids))
                batch_gather_mask.append([1]*len(output_ids))
                
            batch_ids = [[tokenizer.pad_token_id]*(max_length-len(ids)) + ids for ids in batch_ids]
            batch_attention_mask = [[0]*(max_length-len(mask)) + mask for mask in batch_attention_mask]
            batch_gather_mask = [[0]*(max_length-len(mask)) + mask for mask in batch_gather_mask]
            
        except Exception as e:
            print("Error when generating completions for batch:")
            print(batch_prompts)
            print("Error message:")
            print(e)
            print("Use empty string as the completion.")
            batch_ids = batch_prompts * num_return_sequences

        outputs.append(torch.tensor(batch_ids))
        attention_masks.append(torch.tensor(batch_attention_mask))
        gather_masks.append(torch.tensor(batch_gather_mask))

        if not disable_tqdm:
            progress.update(len(batch_prompts)//num_return_sequences)

    # tokenized_generations = tokenizer(generations, padding="longest", return_tensors="pt", add_special_tokens=False)
    # assert len(outputs) == len(prompts) * num_return_sequences, "number of generations should be equal to number of prompts * num_return_sequences"
    return outputs, attention_masks, gather_masks

@torch.no_grad()
def get_next_word_predictions(model, tokenizer, prompts, candidate_token_ids=None, batch_size=1, return_token_predictions=False, add_special_tokens=True, disable_tqdm=False):
    predictions, probs = [], []
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Getting Predictions")

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i: i+batch_size]
        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=add_special_tokens)
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        device = model.device
        batch_input_ids = batch_input_ids.to(device)
        attention_mask = attention_mask.to(device)
        batch_logits = model(input_ids=batch_input_ids, attention_mask=attention_mask).logits[:, -1, :]
        batch_probs = torch.softmax(batch_logits, dim=-1)
        if candidate_token_ids is not None:
            batch_probs = batch_probs[:, candidate_token_ids]
        batch_prediction_indices = torch.argmax(batch_probs, dim=-1)
        if return_token_predictions:
            if candidate_token_ids is not None:
                candidate_tokens = tokenizer.convert_ids_to_tokens(candidate_token_ids)
                batch_predictions = [candidate_tokens[idx] for idx in batch_prediction_indices]
            else:
                batch_predictions = tokenizer.convert_ids_to_tokens(batch_prediction_indices)
            predictions += batch_predictions
        else:
            predictions += batch_prediction_indices.tolist()
        probs += batch_probs.tolist()
        # breakpoint()
        if not disable_tqdm:
            progress.update(len(batch_prompts))

    assert len(predictions) == len(prompts), "number of predictions should be equal to number of prompts"
    return predictions, probs

@torch.no_grad()
def get_next_word_predictions_with_guidance(model, tokenizer, prompts, candidate_token_ids=None, batch_size=1, return_token_predictions=False, add_special_tokens=True, disable_tqdm=False, guided_model=None, index=None, hook_fn=None):
    
    def add_guided_activation_hooks(model, input_ids, attention_mask, guided_model, layers, hook_fn):
        device = guided_model.device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        _, cache = guided_model.run_with_cache(input_ids, attention_mask, names_filter=lambda name: name.endswith('hook_post'))
        for layer, neurons in layers.items():
            layer_cache = cache['post', layer]
            neurons = torch.tensor(neurons) # pass tensor as parameter rather than list of tensor will speed up significantly
            partial_hook_fn = partial(hook_fn, neurons=neurons, patched_values=layer_cache[..., neurons])
            model.add_perma_hook(name=get_act_name('post', layer), hook=partial_hook_fn)
        return model
        
    if guided_model:
        assert index is not None
        layers = defaultdict(list)
        for layer, idx in index:
            layers[layer.item()].append(idx)
    
    predictions, probs = [], []
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Getting Predictions")

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i: i+batch_size]
        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=add_special_tokens)
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        device = model.device
        batch_input_ids = batch_input_ids.to(device)
        attention_mask = attention_mask.to(device)
        if guided_model:
            model = add_guided_activation_hooks(model, batch_input_ids, attention_mask, guided_model, layers, hook_fn)
        batch_logits = model(input_ids=batch_input_ids, attention_mask=attention_mask).logits[:, -1, :]
        if guided_model:
            model.reset_hooks(including_permanent=True)
        batch_probs = torch.softmax(batch_logits, dim=-1)
        if candidate_token_ids is not None:
            batch_probs = batch_probs[:, candidate_token_ids]
        batch_prediction_indices = torch.argmax(batch_probs, dim=-1)
        if return_token_predictions:
            if candidate_token_ids is not None:
                candidate_tokens = tokenizer.convert_ids_to_tokens(candidate_token_ids)
                batch_predictions = [candidate_tokens[idx] for idx in batch_prediction_indices]
            else:
                batch_predictions = tokenizer.convert_ids_to_tokens(batch_prediction_indices)
            predictions += batch_predictions
        else:
            predictions += batch_prediction_indices.tolist()
        probs += batch_probs.tolist()
        # breakpoint()
        if not disable_tqdm:
            progress.update(len(batch_prompts))

    assert len(predictions) == len(prompts), "number of predictions should be equal to number of prompts"
    return predictions, probs

@torch.no_grad()
def score_completions(model, tokenizer, scoring_examples, disable_tqdm=False):
    '''
    Each scoring example is a dict, which contains the following keys:
    - prompt: the prompt to score
    - completions: a list of completions to score
    '''
    
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(scoring_examples), desc="Scoring Completions")

    # unroll the scoring examples
    unrolled_examples = []
    for scoring_example in scoring_examples:
        prompt = scoring_example["prompt"]
        for completion in scoring_example["completions"]:
            unrolled_examples.append({
                "prompt": prompt,
                "completion": completion
            })

    scores = []
    # currently we don't support batching, because we want to directly use the loss returned by the model to score each completion.
    for unrolled_example in unrolled_examples:
        encoded_example = encode_with_prompt_completion_format(unrolled_example, tokenizer, max_seq_length=None)
        # unsqueeze the batch dimension
        for key, value in encoded_example.items():
            encoded_example[key] = value.unsqueeze(0)
        if model.device.type == "cuda":
            encoded_example = {
                key: value.cuda() for key, value in encoded_example.items()
            }
        outputs = model(**encoded_example)
        loss = outputs.loss
        scores.append(-loss.item())
        if not disable_tqdm:
            progress.update(1)

    # roll up the scores
    rolled_up_scores = {}
    for unrolled_example, score in zip(unrolled_examples, scores):
        prompt = unrolled_example["prompt"]
        completion = unrolled_example["completion"]
        if prompt not in rolled_up_scores:
            rolled_up_scores[prompt] = {}
        rolled_up_scores[prompt][completion] = score

    return rolled_up_scores


def load_hooked_lm_and_tokenizer(
        model_name_or_path, 
        peft_name_or_path=None,
        tokenizer_name_or_path=None, 
        device_map="auto", 
        torch_dtype="auto",
        load_in_8bit=False, 
        convert_to_half=False,
        use_fast_tokenizer=True,
        padding_side="left"
    ):
    
    config = json.load(open(os.path.join(model_name_or_path, 'config.json')))
    model_cls = AUTO_MODEL_MAPPING[config['model_type']]
    
    if load_in_8bit:
        hook_model = model_cls.from_pretrained(
            model_name_or_path, 
            device_map=device_map, 
            load_in_8bit=True
        )
    else:
        if device_map:
            hook_model = model_cls.from_pretrained(model_name_or_path, device_map=device_map, torch_dtype=torch_dtype)
        else:
            hook_model = model_cls.from_pretrained(model_name_or_path, torch_dtype=torch_dtype)
            if torch.cuda.is_available():
                hook_model = hook_model.cuda()
        if convert_to_half:
            hook_model = hook_model.half()

    if not tokenizer_name_or_path:
        tokenizer_name_or_path = model_name_or_path
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=use_fast_tokenizer)
    except:
        # some tokenizers (e.g., GPTNeoXTokenizer) don't have the slow or fast version, so we just roll back to the default one
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    # set padding side to left for batch generation
    tokenizer.padding_side = padding_side
    # set pad token to eos token if pad token is not set (as is the case for llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if peft_name_or_path:
        if not isinstance(peft_name_or_path, list):
            peft_name_or_path = [peft_name_or_path]
        for path in peft_name_or_path:
            peft_config = json.load(open(os.path.join(path, "adapter_config.json")))
            setattr(hook_model, "peft_type", peft_config["peft_type"])
            print(f"Load {peft_config['peft_type']} from {path}.")
            if peft_config["peft_type"] == "PROMPT_TUNING":
                hook_model.set_prompt_embeddings(path)
            elif peft_config["peft_type"] == "IA3":
                hook_model.set_ia3(path)
            elif peft_config["peft_type"] == "LORA":
                hook_model.set_lora(path, peft_config)
            # model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch_dtype)
            # peft_model = PeftModel.from_pretrained(model=model, model_id=peft_name_or_path)
            # model = peft_model.merge_and_unload(progressbar=True)
            # hook_model.from_hf_model(model)
        
    hook_model.eval()
    print(f'Load {model_cls.__name__} successfully!')
    return hook_model, tokenizer

def load_hf_lm_and_tokenizer(
        model_name_or_path, 
        peft_name_or_path=None,
        tokenizer_name_or_path=None, 
        device_map="auto", 
        torch_dtype="auto",
        load_in_8bit=False, 
        convert_to_half=False,
        gptq_model=False,
        use_fast_tokenizer=True,
        padding_side="left",
    ):
    
    from transformers import AutoModelForCausalLM, AutoTokenizer, OPTForCausalLM, GPTNeoXForCausalLM

    if gptq_model:
        from auto_gptq import AutoGPTQForCausalLM
        model_wrapper = AutoGPTQForCausalLM.from_quantized(
            model_name_or_path, device="cuda:0", use_triton=True
        )
        model = model_wrapper.model  
    elif load_in_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            device_map=device_map, 
            load_in_8bit=True
        )
    else:
        if device_map:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map=device_map, torch_dtype=torch_dtype)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch_dtype)
            if torch.cuda.is_available():
                model = model.cuda()
        if convert_to_half:
            model = model.half()
    model.eval()

    if not tokenizer_name_or_path:
        tokenizer_name_or_path = model_name_or_path
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=use_fast_tokenizer)
    except:
        # some tokenizers (e.g., GPTNeoXTokenizer) don't have the slow or fast version, so we just roll back to the default one
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    # set padding side to left for batch generation
    tokenizer.padding_side = padding_side
    # set pad token to eos token if pad token is not set (as is the case for llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    for path in peft_name_or_path:
        model = PeftModel.from_pretrained(model=model, model_id=path)

    # for OPT and Pythia models, we need to set tokenizer.model_max_length to model.config.max_position_embeddings 
    # to avoid wrong embedding index.    
    if isinstance(model, GPTNeoXForCausalLM) or isinstance(model, OPTForCausalLM):
        tokenizer.model_max_length = model.config.max_position_embeddings
        print("Set tokenizer.model_max_length to model.config.max_position_embeddings: {}".format(model.config.max_position_embeddings))
        
    return model, tokenizer

def load_hf_score_lm_and_tokenizer(
        model_name_or_path, 
        tokenizer_name_or_path=None, 
        device_map="auto", 
        torch_dtype="auto",
        load_in_8bit=False, 
        convert_to_half=False,
        use_fast_tokenizer=True,
        padding_side="right",
    ):
    
    from transformers import AutoTokenizer
    from eval.arena.models.llama_modelling import LlamaModelForScore
    from eval.arena.models.modeling_llama_rm import LlamaRewardModel

    model_cls = LlamaRewardModel if 'ultra' in model_name_or_path.lower() else LlamaModelForScore

    if load_in_8bit:
        model = model_cls.from_pretrained(
            model_name_or_path, 
            device_map=device_map, 
            load_in_8bit=True
        )
    else:
        if device_map:
            model = model_cls.from_pretrained(model_name_or_path, device_map=device_map, torch_dtype=torch_dtype)
        else:
            model = model_cls.from_pretrained(model_name_or_path, torch_dtype=torch_dtype)
            if torch.cuda.is_available():
                model = model.cuda()
        if convert_to_half:
            model = model.half()
    model.eval()

    if not tokenizer_name_or_path:
        tokenizer_name_or_path = model_name_or_path
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=use_fast_tokenizer)
    except:
        # some tokenizers (e.g., GPTNeoXTokenizer) don't have the slow or fast version, so we just roll back to the default one
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    # set padding side to left for batch generation
    tokenizer.padding_side = padding_side
    # set pad token to eos token if pad token is not set (as is the case for llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    return model, tokenizer

def query_openai_chat_model(engine, instances, output_path=None, batch_size=10, retry_limit=5, reuse_existing_outputs=True, **completion_kwargs):
    '''
    Query OpenAI chat model and save the results to output_path.
    `instances` is a list of dictionaries, each dictionary contains a key "prompt" and a key "id".
    '''
    existing_data = {}
    if reuse_existing_outputs and output_path is not None and os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                instance = json.loads(line)
                existing_data[instance["id"]] = instance

    # by default, we use temperature 0.0 to get the most likely completion.
    if "temperature" not in completion_kwargs:
        completion_kwargs["temperature"] = 0.0

    results = []
    if output_path is not None:
        fout = open(output_path, "w")

    retry_count = 0
    progress_bar = tqdm.tqdm(total=len(instances))
    for i in range(0, len(instances), batch_size):
        batch = instances[i:i+batch_size]
        if all([x["id"] in existing_data for x in batch]):
            results.extend([existing_data[x["id"]] for x in batch])
            if output_path is not None:
                for instance in batch:
                    fout.write(json.dumps(existing_data[instance["id"]]) + "\n")
                    fout.flush()
            progress_bar.update(batch_size)
            continue
        messages_list = []
        for instance in batch:
            messages = [{"role": "user", "content": instance["prompt"]}]
            messages_list.append(messages)
        while retry_count < retry_limit:
            try:
                outputs = asyncio.run(
                    dispatch_openai_chat_requesets(
                    messages_list=messages_list,
                    model=engine,
                    **completion_kwargs,
                ))
                retry_count = 0
                break
            except Exception as e:
                retry_count += 1
                print(f"Error while requesting OpenAI API.")
                print(e)
                print(f"Sleep for {30*retry_count} seconds.")
                time.sleep(30*retry_count)
                print(f"Retry for the {retry_count} time.")
        if retry_count == retry_limit:
            raise RuntimeError(f"Failed to get response from OpenAI API after {retry_limit} retries.")
        assert len(outputs) == len(batch)
        for instance, output in zip(batch, outputs):
            # breakpoint()
            instance["output"] = output.choices[0].message.content
            # instance["response_metadata"] = output
            results.append(instance)
            if output_path is not None:
                fout.write(json.dumps(instance) + "\n")
                fout.flush()
        progress_bar.update(batch_size)
    return results
 

def query_openai_model(engine, instances, output_path=None, batch_size=10, retry_limit=5, reuse_existing_outputs=True, **completion_kwargs):
    '''
    Query OpenAI chat model and save the results to output_path.
    `instances` is a list of dictionaries, each dictionary contains a key "prompt" and a key "id".
    '''
    existing_data = {}
    if reuse_existing_outputs and output_path is not None and os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                instance = json.loads(line)
                existing_data[instance["id"]] = instance

    # by default, we use temperature 0.0 to get the most likely completion.
    if "temperature" not in completion_kwargs:
        completion_kwargs["temperature"] = 0.0

    results = []
    if output_path is not None:
        fout = open(output_path, "w")

    retry_count = 0
    progress_bar = tqdm.tqdm(total=len(instances))
    for i in range(0, len(instances), batch_size):
        batch = instances[i:i+batch_size]
        if all([x["id"] in existing_data for x in batch]):
            results.extend([existing_data[x["id"]] for x in batch])
            if output_path is not None:
                for instance in batch:
                    fout.write(json.dumps(existing_data[instance["id"]]) + "\n")
                    fout.flush()
            progress_bar.update(batch_size)
            continue
        messages_list = []
        for instance in batch:
            messages = instance["prompt"]
            messages_list.append(messages)
        while retry_count < retry_limit:
            try:
                outputs = asyncio.run(
                    dispatch_openai_prompt_requesets(
                    prompt_list=messages_list,
                    model=engine,
                    **completion_kwargs,
                ))
                retry_count = 0
                break
            except Exception as e:
                retry_count += 1
                print(f"Error while requesting OpenAI API.")
                print(e)
                print(f"Sleep for {30*retry_count} seconds.")
                time.sleep(30*retry_count)
                print(f"Retry for the {retry_count} time.")
        if retry_count == retry_limit:
            raise RuntimeError(f"Failed to get response from OpenAI API after {retry_limit} retries.")
        assert len(outputs) == len(batch)
        for instance, output in zip(batch, outputs):
            instance[f"output"] = output["choices"][0]["text"]
            instance["response_metadata"] = output
            results.append(instance)
            if output_path is not None:
                fout.write(json.dumps(instance) + "\n")
                fout.flush()
        progress_bar.update(batch_size)
    return results


def dynamic_import_function(function_path):
    '''
    Dynamically import a function from a path string (e.g., "module.submodule.my_function")
    '''
    module_path, function_name = function_path.rsplit(".", 1)
    module = import_module(module_path)
    function = getattr(module, function_name)
    return function
 