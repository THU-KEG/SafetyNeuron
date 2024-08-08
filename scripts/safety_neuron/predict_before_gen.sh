export CUDA_VISIBLE_DEVICES=4,5,6,7
export HF_DATASETS_CACHE=/data1/cjh/.cache
data_root=/data1/cjh/Alignment
model_root=/data2
peft_root=${data_root}/output
cost_model_root=/data3/MODELS
MODELS=(Llama-2-7b-hf gemma-7b Mistral-7B-v0.1)
PEFT_NAME=(sharegpt_ia3_ff_1 sharegpt_ia3_ff_1_hh_harmless_dpo_ia3_ff)
# TOPK=3000
NUM_SAMPLE=200
BATCH_SIZE=100
# dataset_name=(jailbreak_llms hh_harmless beavertails harmbench red_team_attempts)
train_set=(data/eval/hh_rlhf_harmless/hh_harmless_data_train.jsonl)
test_set=(data/eval/jailbreak_llms/jailbreak_llms.jsonl data/raw_train/BeaverTails/test.jsonl data/eval/harmbench/harmbench.jsonl data/eval/red_team_attempts/red_team_attempts.jsonl)

for MODEL in ${MODELS[@]}
do
    PEFT_PATH=()
    for ((i=0; i<${#PEFT_NAME[*]}; ++i)) 
    do
        PEFT_PATH[i]=${peft_root}/${MODEL}_${PEFT_NAME[i]}
    done

    test_set_dir=()
    for ((i=0; i<${#test_set[*]}; ++i)) 
    do
        test_set_dir[i]=${data_root}/${test_set[i]}
    done

    python -m src.predict_before_gen \
        --train_dataset ${data_root}/${train_set[0]} \
        --eval_datasets ${test_set_dir[@]} \
        --output_dir ./results \
        --output_filename safety_guard.json \
        --model_name_or_path ${model_root}/${MODEL} \
        --tokenizer_name_or_path ${model_root}/${MODEL} \
        --cost_model_name_or_path ${cost_model_root}/beaver-7b-v1.0-cost \
        --index_path ${data_root}/hooked_llama/neuron_activation/${MODEL}_${PEFT_NAME[-1]}_sft_vs_dpo_on_hh_harmless_sft_completion.pt \
        --peft_path ${PEFT_PATH[0]} \
        --eval_batch_size ${BATCH_SIZE} \
        --num_samples ${NUM_SAMPLE} \
        --topk 1500

    python -m src.predict_before_gen \
        --train_dataset ${data_root}/${train_set[0]} \
        --eval_datasets ${test_set_dir[@]} \
        --output_dir ./results \
        --output_filename safety_guard.json \
        --model_name_or_path ${model_root}/${MODEL} \
        --tokenizer_name_or_path ${model_root}/${MODEL} \
        --cost_model_name_or_path ${cost_model_root}/beaver-7b-v1.0-cost \
        --index_path ${data_root}/hooked_llama/neuron_activation/${MODEL}_${PEFT_NAME[-1]}_sft_vs_dpo_on_hh_harmless_sft_completion.pt \
        --peft_path ${PEFT_PATH[@]} \
        --eval_batch_size ${BATCH_SIZE} \
        --num_samples ${NUM_SAMPLE} \
        --topk 1500

done