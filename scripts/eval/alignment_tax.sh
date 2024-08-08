export CUDA_VISIBLE_DEVICES=4,5,6,7
export HF_DATASETS_CACHE=/data1/cjh/.cache
data_root=/data1/cjh/Alignment
model_root1=/data3/MODELS
model_root2=/data2
peft_root=${data_root}/output
BATCH_SIZE=100
NUM_SAMPLE=200
MAX_NEW_TOKENS=128
DATASET=data/raw_train/BeaverTails/test.jsonl
TOPK=(20000)
MODELS=(Llama-2-7b-hf gemma-7b Mistral-7B-v0.1)
DPO=(hh_harmless hh_helpful)
NEURONS=(shp rewardbench_safety)
# MODELS=(gemma-7b Mistral-7B-v0.1)

for seed in 1
do
    PEFT_NAME=(sharegpt_ia3_ff_${seed} sharegpt_ia3_ff_${seed}_${DPO[0]}_dpo_ia3_ff)
    GUIDED_PEFT_NAME=(sharegpt_ia3_ff_${seed} sharegpt_ia3_ff_${seed}_${DPO[1]}_dpo_ia3_ff)
    for MODEL in ${MODELS[@]}
    do
        PEFT_PATH=()
        GUIDED_PEFT_PATH=()
        for ((i=0; i<${#PEFT_NAME[*]}; ++i)) 
        do
            PEFT_PATH[i]=${peft_root}/${MODEL}_${PEFT_NAME[i]}
            GUIDED_PEFT_PATH[i]=${peft_root}/${MODEL}_${GUIDED_PEFT_NAME[i]}
        done

        for dataset in hh_harmless
        do
            INDEX_PATH=(
                sft_vs_dpo_on_${dataset}_sft_completion
            )

            for index_path in ${INDEX_PATH[@]}
            do
                python -m eval.arena.run_eval \
                    --dataset ${data_root}/${DATASET} \
                    --save_dir results/arena \
                    --model_name_or_path ${model_root2}/${MODEL} \
                    --tokenizer_name_or_path ${model_root2}/${MODEL} \
                    --cost_model_name_or_path ${model_root1}/beaver-7b-v1.0-cost \
                    --reward_model_name_or_path ${model_root1}/beaver-7b-v1.0-reward \
                    --eval_batch_size ${BATCH_SIZE} \
                    --num_samples ${NUM_SAMPLE} \
                    --max_new_tokens ${MAX_NEW_TOKENS} \
                    --topk_ablate ${TOPK[@]} \
                    --use_chat_format \
                    --red_peft_path ${PEFT_PATH[@]} \
                    --blue_peft_path ${GUIDED_PEFT_PATH[@]} \
                    --guided_generation \
                    --index_path ${data_root}/hooked_llama/neuron_activation/${MODEL}_sharegpt_ia3_ff_${seed}_${NEURONS[0]}_dpo_ia3_ff_${index_path}.pt \
                    --intersect_index_path ${data_root}/hooked_llama/neuron_activation/${MODEL}_sharegpt_ia3_ff_${seed}_${NEURONS[1]}_dpo_ia3_ff_${index_path}.pt

                python -m eval.arena.run_eval \
                    --dataset ${data_root}/${DATASET} \
                    --save_dir results/arena \
                    --model_name_or_path ${model_root2}/${MODEL} \
                    --tokenizer_name_or_path ${model_root2}/${MODEL} \
                    --cost_model_name_or_path ${model_root1}/beaver-7b-v1.0-cost \
                    --reward_model_name_or_path ${model_root1}/beaver-7b-v1.0-reward \
                    --eval_batch_size ${BATCH_SIZE} \
                    --num_samples ${NUM_SAMPLE} \
                    --max_new_tokens ${MAX_NEW_TOKENS} \
                    --topk_ablate ${TOPK[@]} \
                    --use_chat_format \
                    --blue_peft_path ${PEFT_PATH[@]} \
                    --red_peft_path ${GUIDED_PEFT_PATH[@]} \
                    --guided_generation \
                    --index_path ${data_root}/hooked_llama/neuron_activation/${MODEL}_sharegpt_ia3_ff_${seed}_${NEURONS[0]}_dpo_ia3_ff_${index_path}.pt \
                    --intersect_index_path ${data_root}/hooked_llama/neuron_activation/${MODEL}_sharegpt_ia3_ff_${seed}_${NEURONS[1]}_dpo_ia3_ff_${index_path}.pt
               
                # python -m eval.arena.run_eval \
                #     --dataset ${data_root}/${DATASET} \
                #     --save_dir results/arena \
                #     --model_name_or_path ${model_root2}/${MODEL} \
                #     --tokenizer_name_or_path ${model_root2}/${MODEL} \
                #     --cost_model_name_or_path ${model_root1}/beaver-7b-v1.0-cost \
                #     --reward_model_name_or_path ${model_root1}/beaver-7b-v1.0-reward \
                #     --eval_batch_size ${BATCH_SIZE} \
                #     --num_samples ${NUM_SAMPLE} \
                #     --max_new_tokens ${MAX_NEW_TOKENS} \
                #     --topk_ablate ${TOPK[@]} \
                #     --use_chat_format \
                #     --blue_peft_path ${PEFT_PATH[@]} \
                #     --guided_generation \
                #     --index_path ${data_root}/hooked_llama/neuron_activation/${MODEL}_sharegpt_ia3_ff_${seed}_hh_harmless_dpo_ia3_ff_${index_path}.pt \
                #     --intersect_index_path ${data_root}/hooked_llama/neuron_activation/${MODEL}_sharegpt_ia3_ff_${seed}_hh_helpful_dpo_ia3_ff_${index_path}.pt
            done
        done
    done
done
