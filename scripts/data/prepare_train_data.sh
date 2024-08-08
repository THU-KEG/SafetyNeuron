# check if there is $HF_TOKEN in the environment variables
export HF_TOKEN=
if [ -z "$HF_TOKEN" ]
then
    echo "Warning: HuggingFace dataset LIMA requires permissive access."
    echo "Warning: Please request the access at https://huggingface.co/datasets/GAIR/lima and set the HF_TOKEN environment variable before running this script."
    exit 1
fi

data_dir='/data1/cjh/Alignment'

# echo "Downloading Super-NaturalInstructions dataset..."
# wget -P data/raw_train/super_ni/ https://github.com/allenai/natural-instructions/archive/refs/heads/master.zip
# unzip data/raw_train/super_ni/master.zip -d data/raw_train/super_ni/ && rm data/raw_train/super_ni/master.zip
# mv data/raw_train/super_ni/natural-instructions-master/* data/raw_train/super_ni/ && rm -r data/raw_train/super_ni/natural-instructions-master


# echo "Downloading the flan_v2 chain-of-thought submix..."
# wget -P data/raw_train/cot/ https://beaker.org/api/v3/datasets/01GXZ52K2Q932H6KZY499A7FE8/files/cot_zsopt.jsonl
# wget -P data/raw_train/cot/ https://beaker.org/api/v3/datasets/01GXZ51ZV283RAZW7J3ECM4S58/files/cot_fsopt.jsonl


# echo "Downloading the flan_v2 collection, here we use two subsampled versions: for tulu v1 we subsampled 100K, for tulu v2 we subsampled 50K..."
# mkdir -p data/raw_train/flan_v2/
# wget -O data/raw_train/flan_v2/tulu_v1_resampled_flan_100k.jsonl https://beaker.org/api/v3/datasets/01GZTTS2EJFPA83PXS4FQCS1SA/files/flan_v2_resampled_100k.jsonl
# wget -O data/raw_train/flan_v2/tulu_v2_resampled_flan_50k.jsonl https://beaker.org/api/v3/datasets/01HBS0N5ZSDF5AECA9VMB1RKXQ/files/flan_v2_resampled_50k.jsonl


# echo "Downloading self-instruct data..."
# wget -P data/raw_train/self_instruct/ https://raw.githubusercontent.com/yizhongw/self-instruct/main/data/gpt3_generations/batch_221203/all_instances_82K.jsonl


# echo "Downloading unnatural-instructions data..."
# wget -P data/raw_train/unnatural_instructions/ https://github.com/orhonovich/unnatural-instructions/raw/main/data/core_data.zip
# unzip data/raw_train/unnatural_instructions/core_data.zip -d data/raw_train/unnatural_instructions/


# echo "Downloading Stanford alpaca data..."
# wget -P data/raw_train/stanford_alpaca/ https://github.com/tatsu-lab/stanford_alpaca/raw/main/alpaca_data.json


# echo "Downloading the dolly dataset..."
# wget -P data/raw_train/dolly/ https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl


# echo "Downloading the OpenAssistant data (oasst1)..."
# wget -P data/raw_train/oasst1/ https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/2023-04-12_oasst_ready.trees.jsonl.gz
# gzip -d data/raw_train/oasst1/2023-04-12_oasst_ready.trees.jsonl.gz


# echo "Downloading the code alpaca dataset..."
# wget -P data/raw_train/code_alpaca/ https://github.com/sahil280114/codealpaca/raw/master/data/code_alpaca_20k.json


# echo "Downloading the gpt4-llm dataset..."
# wget -P data/raw_train/gpt4_alpaca/ https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/raw/main/data/alpaca_gpt4_data.json
# wget -P data/raw_train/gpt4_alpaca/ https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/raw/main/data/alpaca_gpt4_data_zh.json


# echo "Downloading the baize dataset..."
# wget -P data/raw_train/baize/ https://github.com/project-baize/baize-chatbot/raw/main/data/alpaca_chat_data.json
# wget -P data/raw_train/baize/ https://github.com/project-baize/baize-chatbot/raw/main/data/medical_chat_data.json
# wget -P data/raw_train/baize/ https://github.com/project-baize/baize-chatbot/raw/main/data/quora_chat_data.json
# wget -P data/raw_train/baize/ https://github.com/project-baize/baize-chatbot/raw/main/data/stackoverflow_chat_data.json


# echo "Downloading ShareGPT dataset..."
# wget -nc -P data/raw_train/sharegpt/ https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/HTML_cleaned_raw_dataset/sg_90k_part1_html_cleaned.json
# wget -nc -P data/raw_train/sharegpt/ https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/HTML_cleaned_raw_dataset/sg_90k_part2_html_cleaned.json
# echo "Splitting the ShareGPT dataset with 2048 max tokens per conversation..."
# python split_sharegpt_conversations.py \
#     --in-files data/raw_train/sharegpt/sg_90k_part1_html_cleaned.json data/raw_train/sharegpt/sg_90k_part2_html_cleaned.json \
#     --out-file data/raw_train/sharegpt/sharegpt_html_cleaned_and_split_2048.json \
#     --model-name-or-path /data3/MODELS/llama-7b-hf \
#     --max-length 2048
# echo "Splitting the ShareGPT dataset with 4096 max tokens per conversation..."
# python split_sharegpt_conversations.py \
#     --in-files data/raw_train/sharegpt/sg_90k_part1_html_cleaned.json data/raw_train/sharegpt/sg_90k_part2_html_cleaned.json \
#     --out-file data/raw_train/sharegpt/sharegpt_html_cleaned_and_split_4096.json \
#     --model-name-or-path /data3/MODELS/llama-7b-hf \
#     --max-length 4096


# echo "Downloading LIMA dataset..."
# wget --header="Authorization: Bearer $HF_TOKEN" -P data/raw_train/lima/ https://huggingface.co/datasets/GAIR/lima/raw/main/train.jsonl


# echo "Downloading WizardLM dataset..."
# wget -P data/raw_train/wizardlm/ https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_V2_196k/resolve/main/WizardLM_evol_instruct_V2_143k.json


# echo "Downloading the OpenOrca dataset..."
# wget -P data/raw_train/open_orca/ https://huggingface.co/datasets/Open-Orca/OpenOrca/resolve/main/1M-GPT4-Augmented.parquet
# wget -P data/raw_train/open_orca/ https://huggingface.co/datasets/Open-Orca/OpenOrca/resolve/main/3_5M-GPT3_5-Augmented.parquet


# echo "Downloading the Science Instructions dataset..."
# wget -P data/raw_train/science https://beaker.org/api/v3/datasets/01HBS3G7TA8AT15C7RWTJAN66X/files/science_train.jsonl


# echo "Downloading the HardCoded dataset..."
# wget -P data/raw_train/hard_coded/ https://beaker.org/api/v3/datasets/01HBS14BBV16K45MMFSYJR86CA/files/hard_coded_examples.xlsx

# echo "Downloading the hh-rlhf dataset..."
# wget -P ${data_dir}/data/raw_train/hh_rlhf_harmless https://huggingface.co/datasets/Anthropic/hh-rlhf/resolve/main/harmless-base/train.jsonl.gz
# wget -P ${data_dir}/data/raw_train/hh_rlhf_harmless https://huggingface.co/datasets/Anthropic/hh-rlhf/resolve/main/harmless-base/test.jsonl.gz
# gzip -d ${data_dir}/data/raw_train/hh_rlhf_harmless/train.jsonl.gz
# gzip -d ${data_dir}/data/raw_train/hh_rlhf_harmless/test.jsonl.gz
# wget -P ${data_dir}/data/raw_train/hh_rlhf_helpful https://huggingface.co/datasets/Anthropic/hh-rlhf/resolve/main/helpful-base/train.jsonl.gz
# wget -P ${data_dir}/data/raw_train/hh_rlhf_helpful https://huggingface.co/datasets/Anthropic/hh-rlhf/resolve/main/helpful-base/test.jsonl.gz
# gzip -d ${data_dir}/data/raw_train/hh_rlhf_helpful/train.jsonl.gz
# gzip -d ${data_dir}/data/raw_train/hh_rlhf_helpful/test.jsonl.gz

# echo "Downloading the BeaverTails dataset..."
# wget -P ${data_dir}/data/raw_train/BeaverTails https://huggingface.co/datasets/PKU-Alignment/BeaverTails/resolve/main/round0/330k/train.jsonl.xz
# wget -P ${data_dir}/data/raw_train/BeaverTails https://huggingface.co/datasets/PKU-Alignment/BeaverTails/resolve/main/round0/330k/test.jsonl.xz
# unxz -d ${data_dir}/data/raw_train/BeaverTails/train.jsonl.xz
# unxz -d ${data_dir}/data/raw_train/BeaverTails/test.jsonl.xz

# echo "Downloading HarmfulQA dataset..."
# wget -P ${data_dir}/data/raw_train/HarmfulQA/ https://huggingface.co/datasets/declare-lab/HarmfulQA/resolve/main/data_for_hub.json
# python scripts/split_harmfulqa_conversations.py \
#     --in-files ${data_dir}/data/raw_train/HarmfulQA/data_for_hub.json \
#     --out-file ${data_dir}/data/raw_train/HarmfulQA/harmful_qa_2048.json \
#     --model-name-or-path /data3/MODELS/llama-7b-hf \
#     --max-length 2048

echo "Downloading the HuggingFace H4 Stack Exchange Preference Dataset..."
wget -P ${data_dir}/data/raw_train/H4_stack_exchange https://huggingface.co/datasets/prhegde/preference-data-math-stack-exchange/resolve/main/math_stack_exchange.json

# echo "Processing datasets..."
# python scripts/reformat_datasets.py --raw_data_dir ${data_dir}/data/raw_train/ --output_dir ${data_dir}/data/processed/ --dataset HarmfulQA
# python scripts/reformat_datasets.py --raw_data_dir ${data_dir}/data/raw_train/ --output_dir ${data_dir}/data/processed/ --dataset sharegpt