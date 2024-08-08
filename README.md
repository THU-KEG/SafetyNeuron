# SafetyNeuron
Data and code for the paper: Finding Safety Neurons in Large Language Models


## Installation

``` bash
git clone https://github.com/THU-KEG/SafetyNeuron.git
cd SafetyNeuron
```

## Data Preparation
The datasets used in our experiments can be downloaded with scripts in `scripts/data`. You can also add your datasets by converting the format to a unified format. Details can be found in `scripts/data/reformat_datasets.py`.


## Model Alignment

Our code to fine-tune model is stored in the `src/training` folder. You can also use the following scripts to obtain the models we used in paper.

``` bash
bash scripts/training/finetune_lora_with_accelerate.sh
bash scripts/training/dpo.sh
```

## Finding Safety Neurons

### HookedModel Class

Our `HookedPreTrainedModel` in `src/models/HookedModelBase.py` inherits both from `TransformerLens` and huggingface `transformers`, supporting methods such as `model.run_with_cache()`, `model.generate()`. You can also add your models following the implementation in `src/models/HookedLlama.py`

### Compute Change Scores

Our implementation of *Generation-Time Activation Contrasting* is in `src/activation_processor.py`. You can use the following script to compute and save the *change scores* and *neuron ranks*.

``` bash
bash scripts/safety_neuron/get_change_scores.sh
```
The meaning of important arguments
- `--first_peft_path`: The dir containing peft checkpoint. If not provided, we will use the base model.
- `--second_peft_path`: The same as before, and we use generation based on the second model.
- `--token_type`: Which token position to compare neuron activation. Support **full prompt**, **last token of prompt**, **completion**.

### Dynamic Activation Patching

We implement *Dynamic Activation Patching* by overwriting the `generate()` method of `transformers` models in `src/models/HookedModelBase.py` (currently we only implement the greedy seach decoding, the other sampling strategies are similar).

You can perform *Dynamic Activation Patching* by adding 3 extra arguments to `model.generate()`
- `--guided_model`: The model whose activations are used for patching.
- `--index`: The neurons we want to patch, obtained in previous step.
- `--hook_fn`: The hook function actually performs patching.


### Evaluation

Code of this part is stored in the `src/eval` folder. 

You can use the following script to evaluate the results of dynamic activation patching

```bash
bash scripts/eval/arena.sh
```
Here are some important arguments in the script
- `--guided_generation`: If not specified, only evaluate as usual LLMs.
- `--cost_model_name_or_path`: The model used to evaluate the safety of responses.
- `--topk_ablate`: The number of neurons we want to intervene. 
- `--red_peft_path`: The model being patched. If not provided, we will use the base model.
- `--blue_peft_path`: The model used for patching. 
- `--index_path`: Safety neurons index.

## LLM Safeguard

Use `src/neuron_activation.py` to create the training datasets. Use `src/predict_before_gen.py` to evalute the performance of trained safeguard.

## Other Experiment

- `src/ppl.py`: Compute the perplexity after dynamic activation patching.
- `src/neuron2word.py`: Project the neuron weights to vocabulary space.
- `src/training_free_neurons.py`: . 
- `eval/*/run_eval.py`: Evaluate the general capabilities of patched model.





# Cite
If you find our code useful, we will sincerely appreciate it and encourage you to cite the following article:

```bibtex
@article{chen2024finding,
  title={Finding Safety Neurons in Large Language Models},
  author={Chen, Jianhui and Wang, Xiaozhi and Yao, Zijun and Bai, Yushi and Hou, Lei and Li, Juanzi},
  journal={arXiv preprint arXiv:2406.14144},
  year={2024}
}
 ```