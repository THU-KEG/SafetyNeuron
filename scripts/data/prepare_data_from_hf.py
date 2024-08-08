from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer

def prepare_reward_bench():
    dataset = load_dataset('allenai/reward-bench', cache_dir='/data1/cjh/.cache')['filtered']

    reasoning_subset = ['hep-python', 'hep-go', 'hep-cpp', 'hep-js', 'hep-rust', 'hep-java', 'math-prm']
    safety_subset = ['refusals-dangerous', 'refusals-offensive', 'xstest-should-refuse', 'xstest-should-respond', 'do not answer']

    safety_dataset = dataset.filter(lambda ex: ex["subset"] in safety_subset)
    safety_dataset.to_json('/data1/cjh/Alignment/data/raw_train/reward_bench/safety.jsonl')

    reasoning_dataset = dataset.filter(lambda ex: ex["subset"] in reasoning_subset)
    reasoning_dataset.to_json('/data1/cjh/Alignment/data/raw_train/reward_bench/reasoning.jsonl')

def prepare_shp():
    # Function to sample 2000 examples from each domain
    def sample_by_domain(dataset_split):
        df = dataset_split.to_pandas()
        sampled_df = df.groupby('domain').apply(lambda x: x.sample(n=2000, replace=False if len(x) >= 2000 else True)).reset_index(drop=True)
        return Dataset.from_pandas(sampled_df)

    # Define the batch filtering function
    def filter_long_history_batch(batch):
        tokenized_histories = tokenizer(batch['history'], truncation=False, padding=False)
        token_lengths = [len(tokens) for tokens in tokenized_histories['input_ids']]
        return [length <= 256 for length in token_lengths]
    
    dataset = load_dataset("stanfordnlp/shp")
    # Apply the filter to each split in the dataset in batch mode
    tokenizer = AutoTokenizer.from_pretrained('/data2/Llama-2-7b-hf', use_fast=True)
    dataset = dataset.filter(filter_long_history_batch, batched=True)
    # Apply the sampling function to each split
    sampled_splits = {split: sample_by_domain(dataset[split]) if split == 'train' else dataset[split] for split in dataset}
    # Combine sampled splits back into a DatasetDict
    dataset = DatasetDict(sampled_splits)
    dataset = dataset.remove_columns(['post_id', 'domain', 'upvote_ratio', 'c_root_id_A', 'c_root_id_B', 'created_at_utc_A', 'created_at_utc_B', 'score_A', 'score_B', 'seconds_difference', 'score_ratio'])
    {split: dataset[split].to_json(f'/data1/cjh/Alignment/data/raw_train/SHP/{split}.jsonl') for split in dataset}
    
prepare_shp()
    