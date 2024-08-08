export CUDA_VISIBLE_DEVICES=4,5
export HF_DATASETS_CACHE=/data1/cjh/.cache
data_root=/data1/cjh/Alignment
model_root1=/data3/MODELS
model_root2=/data2
peft_root=${data_root}/output
BATCH_SIZE=10
NUM_SAMPLE=200
MAX_NEW_TOKENS=128
MODELS=(Llama-2-7b-hf gemma-7b Mistral-7B-v0.1)
PEFT_NAME=(sharegpt_ia3_ff_1 sharegpt_ia3_ff_1_hh_harmless_dpo_ia3_ff)
# python -m eval.mmlu.run_eval \
    # --ntrain 0 \
    # --data_dir ${data_root}/data/eval/mmlu \
    # --save_dir results/mmlu/${MODEL}_0shot_harmless \
    # --model_name_or_path ${model_root}/${MODEL} \
    # --tokenizer_name_or_path ${model_root}/${MODEL} \
    # --eval_batch_size 50 \
    # --hooked
for MODEL in ${MODELS[1]}
do 
    PEFT_PATH=(${peft_root}/${MODEL}_${PEFT_NAME[0]} ${peft_root}/${MODEL}_${PEFT_NAME[1]})

    # python -m eval.mmlu.run_eval \
    #     --ntrain 0 \
    #     --data_dir ${data_root}/data/eval/mmlu \
    #     --save_dir results/mmlu/${MODEL} \
    #     --model_name_or_path ${model_root2}/${MODEL} \
    #     --tokenizer_name_or_path ${model_root2}/${MODEL} \
    #     --eval_batch_size ${BATCH_SIZE} \
    #     --use_chat_format \
    #     --hooked

    # python -m eval.mmlu.run_eval \
    #     --ntrain 0 \
    #     --data_dir ${data_root}/data/eval/mmlu \
    #     --save_dir results/mmlu/${MODEL}-sft \
    #     --model_name_or_path ${model_root2}/${MODEL} \
    #     --tokenizer_name_or_path ${model_root2}/${MODEL} \
    #     --eval_batch_size ${BATCH_SIZE} \
    #     --red_peft_path ${PEFT_PATH[0]} \
    #     --use_chat_format \
    #     --hooked

    # python -m eval.mmlu.run_eval \
    #     --ntrain 0 \
    #     --data_dir ${data_root}/data/eval/mmlu \
    #     --save_dir results/mmlu/${MODEL}-dpo \
    #     --model_name_or_path ${model_root2}/${MODEL} \
    #     --tokenizer_name_or_path ${model_root2}/${MODEL} \
    #     --eval_batch_size ${BATCH_SIZE} \
    #     --red_peft_path ${PEFT_PATH[@]} \
    #     --use_chat_format \
    #     --hooked

    python -m eval.mmlu.run_eval \
        --ntrain 0 \
        --data_dir ${data_root}/data/eval/mmlu \
        --save_dir results/mmlu/${MODEL}-patch \
        --model_name_or_path ${model_root2}/${MODEL} \
        --tokenizer_name_or_path ${model_root2}/${MODEL} \
        --eval_batch_size ${BATCH_SIZE} \
        --blue_peft_path ${PEFT_PATH[@]} \
        --index_path ${data_root}/hooked_llama/neuron_activation/${MODEL}_${PEFT_NAME[-1]}_sft_vs_dpo_on_hh_harmless_sft_completion.pt \
        --use_chat_format \
        --hooked

    python -m eval.mmlu.run_eval \
        --ntrain 0 \
        --data_dir ${data_root}/data/eval/mmlu \
        --save_dir results/mmlu/${MODEL}-sft-patch \
        --model_name_or_path ${model_root2}/${MODEL} \
        --tokenizer_name_or_path ${model_root2}/${MODEL} \
        --eval_batch_size ${BATCH_SIZE} \
        --red_peft_path ${PEFT_PATH[0]} \
        --blue_peft_path ${PEFT_PATH[@]} \
        --index_path ${data_root}/hooked_llama/neuron_activation/${MODEL}_${PEFT_NAME[-1]}_sft_vs_dpo_on_hh_harmless_sft_completion.pt \
        --use_chat_format \
        --hooked
done
# Evaluating llama 7B model using 0 shot directly
# python -m eval.mmlu.run_eval \
#     --ntrain 0 \
#     --data_dir ${data_root}/data/eval/mmlu \
#     --save_dir results/mmlu/llama-7B-lora-oasst1 \
#     --model_name_or_path ${model_root}/llama-7b-hf \
#     --tokenizer_name_or_path ${model_root}/llama-7b-hf \
#     --eval_batch_size 20 \
#     --load_in_8bit \
#     --peft_path output/llama7b_lora_oasst1

# Evaluating llama 7B lora model using 0 shot directly
# python -m eval.mmlu.run_eval \
#     --ntrain 0 \
#     --data_dir ${data_root}/data/eval/mmlu \
#     --save_dir results/mmlu/${MODEL}_0shot \
#     --model_name_or_path ${model_root}/${MODEL} \
#     --tokenizer_name_or_path ${model_root}/${MODEL} \
#     --eval_batch_size 50 \
    # --load_in_8bit \
    # --peft_path output/${MODEL}_lora_flan

# python -m eval.mmlu.run_eval \
#     --ntrain 0 \
#     --data_dir ${data_root}/data/eval/mmlu \
#     --save_dir results/mmlu/${MODEL}_0shot \
#     --model_name_or_path ${model_root}/${MODEL} \
#     --tokenizer_name_or_path ${model_root}/${MODEL} \
#     --eval_batch_size 10 \
#     --load_in_8bit \


# # Evaluating llama 7B model using 5 shot directly
# python -m eval.mmlu.run_eval \
#     --ntrain 5 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/llama-7B-5shot \
#     --model_name_or_path ../hf_llama_models/7B \
#     --tokenizer_name_or_path ../hf_llama_models/7B \
#     --eval_batch_size 4 \
#     --load_in_8bit


# # Evaluating Tulu 7B model using 0 shot and chat format
# python -m eval.mmlu.run_eval \
#     --ntrain 0 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/tulu-7B-0shot \
#     --model_name_or_path ../checkpoints/tulu_7B \
#     --tokenizer_name_or_path ../checkpoints/tulu_7B \
#     --eval_batch_size 4 \
#     --load_in_8bit \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format


# # Evaluating Tulu 7B model using 5 shot and chat format
# python -m eval.mmlu.run_eval \
#     --ntrain 5 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/tulu-7B-5shot \
#     --model_name_or_path ../checkpoints/tulu_7B \
#     --tokenizer_name_or_path ../checkpoints/tulu_7B \
#     --eval_batch_size 4 \
#     --load_in_8bit \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format


# # Evaluating llama2 chat model using 0-shot and chat format
# python -m eval.mmlu.run_eval \
#     --ntrain 0 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/llama2-chat-7B-5shot \
#     --model_name_or_path ../hf_llama2_models/7B-chat \
#     --tokenizer_name_or_path ../hf_llama2_models/7B-chat \
#     --eval_batch_size 4 \
#     --load_in_8bit \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format


# # Evaluating llama2 chat model using 5-shot and chat format
# python -m eval.mmlu.run_eval \
#     --ntrain 5 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/llama2-chat-7B-5shot \
#     --model_name_or_path ../hf_llama2_models/7B-chat \
#     --tokenizer_name_or_path ../hf_llama2_models/7B-chat \
#     --eval_batch_size 4 \
#     --load_in_8bit \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format


# # Evaluating chatgpt using 0 shot
# python -m eval.mmlu.run_eval \
#     --ntrain 0 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/chatgpt-0shot/ \
#     --openai_engine "gpt-3.5-turbo-0301" \
#     --eval_batch_size 20


# # Evaluating chatgpt using 5 shot
# python -m eval.mmlu.run_eval \
#     --ntrain 5 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/chatgpt-5shot/ \
#     --openai_engine "gpt-3.5-turbo-0301" \
#     --eval_batch_size 20


# # Evaluating gpt4 using 0 shot
# python -m eval.mmlu.run_eval \
#     --ntrain 0 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/gpt4-0shot/ \
#     --openai_engine "gpt-4-0314" \
#     --n_instances 100 \
#     --eval_batch_size 20


# # Evaluating gpt4 using 5 shot
# python -m eval.mmlu.run_eval \
#     --ntrain 5 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/gpt4-5shot/ \
#     --openai_engine "gpt-4-0314" \
#     --n_instances 100 \
#     --eval_batch_size 20