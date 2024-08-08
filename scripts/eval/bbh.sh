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

for MODEL in ${MODELS[@]}
do 
    PEFT_PATH=(${peft_root}/${MODEL}_${PEFT_NAME[0]} ${peft_root}/${MODEL}_${PEFT_NAME[1]})
    # evaluating llama 7B model using chain-of-thought
    # evaluating lora llama 7B model using chain-of-thought
    python -m eval.bbh.run_eval \
        --data_dir ${data_root}/data/eval/bbh \
        --save_dir results/bbh/${MODEL}-cot \
        --model ${model_root2}/${MODEL} \
        --tokenizer ${model_root2}/${MODEL} \
        --eval_batch_size ${BATCH_SIZE} \
        --max_num_examples_per_task 40 \
        --use_chat_format \
        --hooked

    python -m eval.bbh.run_eval \
        --data_dir ${data_root}/data/eval/bbh \
        --save_dir results/bbh/${MODEL}-sft-cot \
        --model ${model_root2}/${MODEL} \
        --tokenizer ${model_root2}/${MODEL} \
        --eval_batch_size ${BATCH_SIZE} \
        --max_num_examples_per_task 40 \
        --use_chat_format \
        --red_peft_path ${PEFT_PATH[0]} \
        --hooked

    python -m eval.bbh.run_eval \
        --data_dir ${data_root}/data/eval/bbh \
        --save_dir results/bbh/${MODEL}-dpo-cot \
        --model ${model_root2}/${MODEL} \
        --tokenizer ${model_root2}/${MODEL} \
        --eval_batch_size ${BATCH_SIZE} \
        --max_num_examples_per_task 40 \
        --use_chat_format \
        --red_peft_path ${PEFT_PATH[@]} \
        --hooked

    python -m eval.bbh.run_eval \
        --data_dir ${data_root}/data/eval/bbh \
        --save_dir results/bbh/${MODEL}-sft-patch-cot \
        --model ${model_root2}/${MODEL} \
        --tokenizer ${model_root2}/${MODEL} \
        --eval_batch_size ${BATCH_SIZE} \
        --max_num_examples_per_task 40 \
        --use_chat_format \
        --red_peft_path ${PEFT_PATH[0]} \
        --blue_peft_path ${PEFT_PATH[@]} \
        --index_path ${data_root}/hooked_llama/neuron_activation/${MODEL}_${PEFT_NAME[-1]}_sft_vs_dpo_on_hh_harmless_sft_completion.pt \
        --hooked
        
    python -m eval.bbh.run_eval \
        --data_dir ${data_root}/data/eval/bbh \
        --save_dir results/bbh/${MODEL}-patch-cot \
        --model ${model_root2}/${MODEL} \
        --tokenizer ${model_root2}/${MODEL} \
        --eval_batch_size ${BATCH_SIZE} \
        --max_num_examples_per_task 40 \
        --use_chat_format \
        --blue_peft_path ${PEFT_PATH[@]} \
        --index_path ${data_root}/hooked_llama/neuron_activation/${MODEL}_${PEFT_NAME[-1]}_sft_vs_dpo_on_hh_harmless_sft_completion.pt \
        --hooked
done
    # python -m eval.bbh.run_eval \
    #     --data_dir ${data_root}/data/eval/bbh \
    #     --save_dir results/bbh/${MODEL}_${PEFT_NAME}_hooked \
    #     --model ${model_root}/${MODEL} \
    #     --tokenizer ${model_root}/${MODEL} \
    #     --eval_batch_size 20 \
    #     --max_num_examples_per_task 40 \
    #     --use_chat_format \
    #     --peft_path ${data_root}/output/${MODEL}_${PEFT_NAME} \
    #     --hooked

    # for size in 30 65
    # do
    #     MODEL=llama-${size}b-hf
    #     # evaluating lora llama 7B model using chain-of-thought
    #     python -m eval.bbh.run_eval \
    #         --data_dir ${data_root}/data/eval/bbh \
    #         --save_dir results/bbh/${MODEL}_lora_flan_cot/ \
    #         --model ${model_root}/${MODEL} \
    #         --tokenizer ${model_root}/${MODEL} \
    #         --eval_batch_size 10 \
    #         --max_num_examples_per_task 40 \
    #         --load_in_8bit \
    #         --lora_path output/${MODEL}_lora_flan

    #     # evaluating lora llama 7B model using chain-of-thought
    #     python -m eval.bbh.run_eval \
    #         --data_dir ${data_root}/data/eval/bbh \
    #         --save_dir results/bbh/${MODEL}_cot/ \
    #         --model ${model_root}/${MODEL} \
    #         --tokenizer ${model_root}/${MODEL} \
    #         --eval_batch_size 10 \
    #         --max_num_examples_per_task 40 \
    #         --load_in_8bit \

    # done

    # evaluating llama 7B model using direct answering (no chain-of-thought)
    # python -m eval.bbh.run_eval \
    #     --data_dir data/eval/bbh \
    #     --save_dir results/bbh/llama-7B-no-cot/ \
    #     --model ../hf_llama_models/7B \
    #     --tokenizer ../hf_llama_models/7B \
    #     --eval_batch_size 10 \
    #     --max_num_examples_per_task 40 \
    #     --load_in_8bit \
    #     --no_cot


    # # evaluating tulu 7B model using chain-of-thought and chat format
    # python -m eval.bbh.run_eval \
    #     --data_dir data/eval/bbh \
    #     --save_dir results/bbh/tulu-7B-cot/ \
    #     --model ../checkpoint/tulu_7B \
    #     --tokenizer ../checkpoints/tulu_7B \
    #     --eval_batch_size 10 \
    #     --max_num_examples_per_task 40 \
    #     --load_in_8bit \
    #     --use_chat_format \
    #     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format


    # # evaluating llama2 chat model using chain-of-thought and chat format
    # python -m eval.bbh.run_eval \
    #     --data_dir data/eval/bbh \
    #     --save_dir results/bbh/llama2-chat-7B-cot \
    #     --model ../hf_llama2_models/7B-chat \
    #     --tokenizer ../hf_llama2_models/7B-chat \
    #     --eval_batch_size 10 \
    #     --max_num_examples_per_task 40 \
    #     --load_in_8bit \
    #     --use_chat_format \
    #     --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format


    # # evaluating gpt-3.5-turbo-0301 using chain-of-thought
    # python -m eval.bbh.run_eval \
    #     --data_dir data/eval/bbh \
    #     --save_dir results/bbh/chatgpt-cot/ \
    #     --openai_engine "gpt-3.5-turbo-0301" \
    #     --eval_batch_size 10 \
    #     --max_num_examples_per_task 40


    # # evaluating gpt-3.5-turbo-0301 using direct answering (no chain-of-thought)
    # python -m eval.bbh.run_eval \
    #     --data_dir data/eval/bbh \
    #     --save_dir results/bbh/chatgpt-no-cot/ \
    #     --openai_engine "gpt-3.5-turbo-0301" \
    #     --eval_batch_size 10 \
    #     --max_num_examples_per_task 40 \
    #     --no_cot


    # # evaluating gpt-4 using chain-of-thought
    # python -m eval.bbh.run_eval \
    #     --data_dir data/eval/bbh \
    #     --save_dir results/bbh/gpt4-cot/ \
    #     --openai_engine "gpt-4-0314" \
    #     --eval_batch_size 10 \
    #     --max_num_examples_per_task 40


    # # evaluating gpt-4 using direct answering (no chain-of-thought)
    # python -m eval.bbh.run_eval \
    #     --data_dir data/eval/bbh \
    #     --save_dir results/bbh/gpt4-no-cot/ \
    #     --openai_engine "gpt-4-0314" \
    #     --eval_batch_size 10 \
    #     --max_num_examples_per_task 40 \
    #     --no_cot