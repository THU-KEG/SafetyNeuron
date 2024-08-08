export CUDA_VISIBLE_DEVICES=4,5
export HF_DATASETS_CACHE=/data1/cjh/.cache
data_root=/data1/cjh/Alignment
model_root1=/data3/MODELS
model_root2=/data2
peft_root=${data_root}/output
BATCH_SIZE=20
NUM_SAMPLE=200
MAX_NEW_TOKENS=128
MODELS=(Llama-2-7b-hf gemma-7b Mistral-7B-v0.1)
PEFT_NAME=(sharegpt_ia3_ff_1 sharegpt_ia3_ff_1_hh_harmless_dpo_ia3_ff)

for MODEL in ${MODELS[@]}
do 
    PEFT_PATH=(${peft_root}/${MODEL}_${PEFT_NAME[0]} ${peft_root}/${MODEL}_${PEFT_NAME[1]})
    # Evaluating llama 7B model using temperature 0.1 to get the pass@1 score
    python -m eval.codex_humaneval.run_eval \
        --data_file ${data_root}/data/eval/codex_humaneval/HumanEval.jsonl.gz \
        --eval_pass_at_ks 1 \
        --unbiased_sampling_size_n 1 \
        --save_dir results/codex_humaneval/llama_7B_temp_0_1 \
        --model ${model_root2}/${MODEL} \
        --tokenizer ${model_root2}/${MODEL} \
        --eval_batch_size ${BATCH_SIZE} \
        --use_chat_format \
        --hooked

    python -m eval.codex_humaneval.run_eval \
        --data_file ${data_root}/data/eval/codex_humaneval/HumanEval.jsonl.gz \
        --eval_pass_at_ks 1 \
        --unbiased_sampling_size_n 1 \
        --save_dir results/codex_humaneval/llama_7B_temp_0_1 \
        --model ${model_root2}/${MODEL} \
        --tokenizer ${model_root2}/${MODEL} \
        --eval_batch_size ${BATCH_SIZE} \
        --red_peft_path ${PEFT_PATH[0]} \
        --use_chat_format \
        --hooked

    python -m eval.codex_humaneval.run_eval \
        --data_file ${data_root}/data/eval/codex_humaneval/HumanEval.jsonl.gz \
        --eval_pass_at_ks 1 \
        --unbiased_sampling_size_n 1 \
        --save_dir results/codex_humaneval/llama_7B_temp_0_1 \
        --model ${model_root2}/${MODEL} \
        --tokenizer ${model_root2}/${MODEL} \
        --eval_batch_size ${BATCH_SIZE} \
        --red_peft_path ${PEFT_PATH[@]} \
        --use_chat_format \
        --hooked

    python -m eval.codex_humaneval.run_eval \
        --data_file ${data_root}/data/eval/codex_humaneval/HumanEval.jsonl.gz \
        --eval_pass_at_ks 1 \
        --unbiased_sampling_size_n 1 \
        --save_dir results/codex_humaneval/llama_7B_temp_0_1 \
        --model ${model_root2}/${MODEL} \
        --tokenizer ${model_root2}/${MODEL} \
        --eval_batch_size ${BATCH_SIZE} \
        --blue_peft_path ${PEFT_PATH[@]} \
        --index_path ${data_root}/hooked_llama/neuron_activation/${MODEL}_${PEFT_NAME[-1]}_sft_vs_dpo_on_hh_harmless_sft_completion.pt \
        --use_chat_format \
        --hooked

    python -m eval.codex_humaneval.run_eval \
        --data_file ${data_root}/data/eval/codex_humaneval/HumanEval.jsonl.gz \
        --eval_pass_at_ks 1 \
        --unbiased_sampling_size_n 1 \
        --save_dir results/codex_humaneval/llama_7B_temp_0_1 \
        --model ${model_root2}/${MODEL} \
        --tokenizer ${model_root2}/${MODEL} \
        --eval_batch_size ${BATCH_SIZE} \
        --red_peft_path ${PEFT_PATH[0]} \
        --blue_peft_path ${PEFT_PATH[@]} \
        --index_path ${data_root}/hooked_llama/neuron_activation/${MODEL}_${PEFT_NAME[-1]}_sft_vs_dpo_on_hh_harmless_sft_completion.pt \
        --use_chat_format \
        --hooked

done
# Evaluating tulu 7B model using temperature 0.1 to get the pass@1 score
# We don't use chat format for codex_humaneval, since it's not a chat dataset
# But you can use it by adding --use_chat_format and --chat_formatting_function create_prompt_with_tulu_chat_format
python -m eval.codex_humaneval.run_eval \
    --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz  \
    --eval_pass_at_ks 1 5 10 20 \
    --unbiased_sampling_size_n 20 \
    --temperature 0.1 \
    --save_dir results/codex_humaneval/tulu_7B_temp_0_1 \
    --model ../checkpoints/tulu_7B/ \
    --tokenizer ../checkpoints/tulu_7B/ \
    --eval_batch_size 32 \
    --load_in_8bit


# Evaluating tulu 7B model using temperature 0.8 to get the pass@10 score
# We don't use chat format for codex_humaneval, since it's not a chat dataset
# But you can use it by adding --use_chat_format and --chat_formatting_function create_prompt_with_tulu_chat_format
python -m eval.codex_humaneval.run_eval \
    --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz  \
    --eval_pass_at_ks 10 \
    --unbiased_sampling_size_n 20 \
    --temperature 0.8 \
    --save_dir results/codex_humaneval/tulu_7B_temp_0_8 \
    --model ../checkpoints/tulu_7B/ \
    --tokenizer ../checkpoints/tulu_7B/ \
    --eval_batch_size 32 \
    --load_in_8bit


# Evaluating chatgpt using temperature 0.1 to get the pass@1 score
python -m eval.codex_humaneval.run_eval \
    --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz \
    --eval_pass_at_ks 1 5 10 20 \
    --unbiased_sampling_size_n 20 \
    --temperature 0.1 \
    --openai_engine "gpt-3.5-turbo-0301" \
    --save_dir results/codex_humaneval/chatgpt_temp_0.1/ \
    --eval_batch_size 10


# Evaluating chatgpt using temperature 0.8 to get the pass@10 score
python -m eval.codex_humaneval.run_eval \
    --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz \
    --eval_pass_at_ks 1 5 10 20 \
    --unbiased_sampling_size_n 20 \
    --temperature 0.8 \
    --openai_engine "gpt-3.5-turbo-0301" \
    --save_dir results/codex_humaneval/chatgpt_temp_0.8/ \
    --eval_batch_size 10


# Evaluating gpt4 using temperature 0.1 to get the pass@1 score
python -m eval.codex_humaneval.run_eval \
    --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz \
    --eval_pass_at_ks 1 5 10 20 \
    --unbiased_sampling_size_n 20 \
    --temperature 0.1 \
    --openai_engine "gpt-4-0314" \
    --save_dir results/codex_humaneval/gpt4_temp_0.1 \
    --eval_batch_size 1


# Evaluating gpt4 using temperature 0.8 to get the pass@10 score
python -m eval.codex_humaneval.run_eval \
    --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz \
    --eval_pass_at_ks 1 5 10 20 \
    --unbiased_sampling_size_n 20 \
    --temperature 0.8 \
    --openai_engine "gpt-4-0314" \
    --save_dir results/codex_humaneval/gpt4_temp_0.8 \
    --eval_batch_size 1