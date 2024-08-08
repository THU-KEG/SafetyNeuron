export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export HF_DATASETS_CACHE=/data1/cjh/.cache
data_root=/data1/cjh/Alignment
model_root1=/data3/MODELS
model_root2=/data2
peft_root=${data_root}/output
BATCH_SIZE=100
NUM_SAMPLE=200
MAX_NEW_TOKENS=128
DATASET=data/raw_train/BeaverTails/test.jsonl
TOPK=(0 200 400 600 800 1000 1200 1500 2000 3000 4000 5000 6000 7000 8000 9000 10000 12000 14000 16000 18000 20000)
MODELS=(Llama-2-7b-hf gemma-7b Mistral-7B-v0.1)
evaluate_only=false

for model in hh_harmless
do
for seed in 1
do
    PEFT_NAME=(sharegpt_ia3_ff_${seed} sharegpt_ia3_ff_${seed}_hh_harmless_dpo_ia3_ff)
    for MODEL in ${MODELS[@]}
    do
        PEFT_PATH=()
        for ((i=0; i<${#PEFT_NAME[*]}; ++i)) 
        do
            PEFT_PATH[i]=${peft_root}/${MODEL}_${PEFT_NAME[i]}
        done
        # evaluation only
        if [ "$evaluate_only" = true ]
        then    
            python -m eval.arena.run_eval \
                --dataset ${data_root}/${DATASET} \
                --save_dir results/arena \
                --model_name_or_path ${model_root2}/${MODEL} \
                --tokenizer_name_or_path ${model_root2}/${MODEL} \
                --cost_model_name_or_path ${model_root1}/beaver-7b-v1.0-cost \
                --eval_batch_size ${BATCH_SIZE} \
                --num_samples ${NUM_SAMPLE} \
                --max_new_tokens ${MAX_NEW_TOKENS} \
                --topk_ablate 0 \
                --use_chat_format 

            python -m eval.arena.run_eval \
                --dataset ${data_root}/${DATASET} \
                --save_dir results/arena \
                --model_name_or_path ${model_root2}/${MODEL} \
                --tokenizer_name_or_path ${model_root2}/${MODEL} \
                --cost_model_name_or_path ${model_root1}/beaver-7b-v1.0-cost \
                --eval_batch_size ${BATCH_SIZE} \
                --num_samples ${NUM_SAMPLE} \
                --max_new_tokens ${MAX_NEW_TOKENS} \
                --topk_ablate 0 \
                --use_chat_format \
                --red_peft_path ${PEFT_PATH[0]} 

            python -m eval.arena.run_eval \
                --dataset ${data_root}/${DATASET} \
                --save_dir results/arena \
                --model_name_or_path ${model_root2}/${MODEL} \
                --tokenizer_name_or_path ${model_root2}/${MODEL} \
                --cost_model_name_or_path ${model_root1}/beaver-7b-v1.0-cost \
                --eval_batch_size ${BATCH_SIZE} \
                --num_samples ${NUM_SAMPLE} \
                --max_new_tokens ${MAX_NEW_TOKENS} \
                --topk_ablate 0 \
                --use_chat_format \
                --red_peft_path ${PEFT_PATH[@]} 

        # activation patching
        else
            for dataset in hh_harmless
            do
                # INDEX_PATH=(
                #     sft_vs_dpo_on_${dataset}_prompt
                #     sft_vs_dpo_on_${dataset}_prompt_last
                #     base_vs_dpo_on_${dataset}_base_completion
                #     base_vs_dpo_on_${dataset}_dpo_completion
                #     sft_vs_dpo_on_${dataset}_dpo_completion
                # )

                # for index_path in ${INDEX_PATH[@]}
                # do
                #     python -m eval.arena.run_eval \
                #         --dataset ${data_root}/${DATASET} \
                #         --save_dir results/arena \
                #         --model_name_or_path ${model_root2}/${MODEL} \
                #         --tokenizer_name_or_path ${model_root2}/${MODEL} \
                #         --cost_model_name_or_path ${model_root1}/beaver-7b-v1.0-cost \
                #         --eval_batch_size ${BATCH_SIZE} \
                #         --num_samples ${NUM_SAMPLE} \
                #         --max_new_tokens ${MAX_NEW_TOKENS} \
                #         --topk_ablate ${TOPK[@]} \
                #         --use_chat_format \
                #         --red_peft_path ${PEFT_PATH[0]} \
                #         --blue_peft_path ${PEFT_PATH[@]} \
                #         --guided_generation \
                #         --index_path ${data_root}/hooked_llama/neuron_activation/${MODEL}_${PEFT_NAME[-1]}_${index_path}.pt \
                #         --value_path ${data_root}/hooked_llama/neuron_activation/${MODEL}_${PEFT_NAME[-1]}_${index_path}.pt 

                #     python -m eval.arena.run_eval \
                #         --dataset ${data_root}/${DATASET} \
                #         --save_dir results/arena \
                #         --model_name_or_path ${model_root2}/${MODEL} \
                #         --tokenizer_name_or_path ${model_root2}/${MODEL} \
                #         --cost_model_name_or_path ${model_root1}/beaver-7b-v1.0-cost \
                #         --eval_batch_size ${BATCH_SIZE} \
                #         --num_samples ${NUM_SAMPLE} \
                #         --max_new_tokens ${MAX_NEW_TOKENS} \
                #         --topk_ablate ${TOPK[@]} \
                #         --use_chat_format \
                #         --blue_peft_path ${PEFT_PATH[@]} \
                #         --guided_generation \
                #         --index_path ${data_root}/hooked_llama/neuron_activation/${MODEL}_${PEFT_NAME[-1]}_${index_path}.pt \
                #         --value_path ${data_root}/hooked_llama/neuron_activation/${MODEL}_${PEFT_NAME[-1]}_${index_path}.pt 
                # done

                INDEX_PATH=(
                    difference_on_hh_prompt_last_token
                    std_on_hh_prompt_last_token
                )

                for index_path in ${INDEX_PATH[@]}
                do
                    python -m eval.arena.run_eval \
                        --dataset ${data_root}/${DATASET} \
                        --save_dir results/arena \
                        --model_name_or_path ${model_root2}/${MODEL} \
                        --tokenizer_name_or_path ${model_root2}/${MODEL} \
                        --cost_model_name_or_path ${model_root1}/beaver-7b-v1.0-cost \
                        --eval_batch_size ${BATCH_SIZE} \
                        --num_samples ${NUM_SAMPLE} \
                        --max_new_tokens ${MAX_NEW_TOKENS} \
                        --topk_ablate ${TOPK[@]} \
                        --use_chat_format \
                        --red_peft_path ${PEFT_PATH[0]} \
                        --blue_peft_path ${PEFT_PATH[@]} \
                        --guided_generation \
                        --index_path ${data_root}/hooked_llama/neuron_activation/${MODEL}_${index_path}.pt \
                        --value_path ${data_root}/hooked_llama/neuron_activation/${MODEL}_${index_path}.pt 

                    python -m eval.arena.run_eval \
                        --dataset ${data_root}/${DATASET} \
                        --save_dir results/arena \
                        --model_name_or_path ${model_root2}/${MODEL} \
                        --tokenizer_name_or_path ${model_root2}/${MODEL} \
                        --cost_model_name_or_path ${model_root1}/beaver-7b-v1.0-cost \
                        --eval_batch_size ${BATCH_SIZE} \
                        --num_samples ${NUM_SAMPLE} \
                        --max_new_tokens ${MAX_NEW_TOKENS} \
                        --topk_ablate ${TOPK[@]} \
                        --use_chat_format \
                        --blue_peft_path ${PEFT_PATH[@]} \
                        --guided_generation \
                        --index_path ${data_root}/hooked_llama/neuron_activation/${MODEL}_${index_path}.pt \
                        --value_path ${data_root}/hooked_llama/neuron_activation/${MODEL}_${index_path}.pt 
                done
            done
        fi
    done
done
done