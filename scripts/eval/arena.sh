export CUDA_VISIBLE_DEVICES=1,2,3,4
export HF_DATASETS_CACHE=/data1/cjh/.cache
data_root=/data1/cjh/Alignment
model_root1=/data3/MODELS
model_root2=/data2
peft_root=${data_root}/output
# model_root2=/data3/MODELS
DATASET=data/raw_train/BeaverTails/test.jsonl
# DATASET=data/eval/red_team_attempts/red_team_attempts.jsonl
BATCH_SIZE=100
NUM_SAMPLE=200
MAX_NEW_TOKENS=128
# TOPK=(0 200 400 600 800 1000 1200 1500 2000 3000 4000 5000 6000 7000 8000 9000 10000 12000 14000 16000 18000 20000)
TOPK=(0 2000 4000 6000 8000 10000 12000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000)
# TOPK=(20000)
# MODELS=(gemma-7b)
MODELS=(Llama-2-7b-hf Mistral-7B-v0.1 gemma-7b)
evaluate_only=false

for model in hh_harmless
do
for seed in 1
do
    PEFT_NAME=(sharegpt_ia3_ff_${seed} sharegpt_ia3_ff_${seed}_${model}_dpo_ia3_ff)
    # PEFT_NAME=(sharegpt_lora_qv sharegpt_lora_qv_merged_hh_harmless_dpo_lora_qv)
    GUIDED_PEFT_NAME=(sharegpt_ia3_ff_${seed} sharegpt_ia3_ff_1_rewardbench_safety_dpo_ia3_ff)
    for MODEL in ${MODELS[@]}
    do
        PEFT_PATH=()
        for ((i=0; i<${#PEFT_NAME[*]}; ++i)) 
        do
            PEFT_PATH[i]=${peft_root}/${MODEL}_${PEFT_NAME[i]}
            GUIDED_PEFT_PATH[i]=${peft_root}/${MODEL}_${GUIDED_PEFT_NAME[i]}
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
                INDEX_PATH=(
                    # base_vs_dpo_on_${dataset}_base_completion
                    # base_vs_dpo_on_${dataset}_dpo_completion
                    sft_vs_dpo_on_${dataset}_sft_completion
                    # sft_vs_dpo_on_${dataset}_dpo_completion
                    # chat_vs_sft_on_${dataset}_sft_completion
                    # chat_vs_sft_on_${dataset}_chat_completion
                    # difference_on_hh_prompt_last_token
                    # std_on_hh_prompt_last_token
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
                        --index_path ${data_root}/hooked_llama/neuron_activation/${MODEL}_${PEFT_NAME[-1]}_${index_path}.pt \
                        --value_path ${data_root}/hooked_llama/neuron_activation/${MODEL}_${PEFT_NAME[-1]}_${index_path}.pt \
                        --sliding_window 20000

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
                        --index_path ${data_root}/hooked_llama/neuron_activation/${MODEL}_${PEFT_NAME[-1]}_${index_path}.pt \
                        --value_path ${data_root}/hooked_llama/neuron_activation/${MODEL}_${PEFT_NAME[-1]}_${index_path}.pt \
                        --sliding_window 20000
                done
            done
        fi
            # python -m eval.arena.run_eval \
            #     --dataset ${data_root}/${DATASET} \
            #     --save_dir results/arena \
            #     --model_name_or_path ${model_root1}/${MODEL} \
            #     --tokenizer_name_or_path ${model_root1}/${MODEL} \
            #     --cost_model_name_or_path ${model_root1}/beaver-7b-v1.0-cost \
            #     --eval_batch_size 200 \
            #     --num_samples 200 \
            #     --max_new_tokens 128 \
            #     --topk_ablate 0 200 400 600 800 1000 1200 1500 2000 3000 4000 5000 6000 7000 8000 9000 10000 12000 14000 16000 18000 20000 \
            #     --use_chat_format \
            #     --red_peft_path ${PEFT_PATH[0]} \
            #     --blue_peft_path ${PEFT_PATH[@]} \
            #     --guided_generation \
            #     --index_path ${data_root}/hooked_llama/neuron_activation/${MODEL}_${PEFT_NAME[-1]}_${index_path}.pt \
            #     --value_path ${data_root}/hooked_llama/neuron_activation/${MODEL}_${PEFT_NAME[-1]}_${index_path}.pt 

            # python -m eval.arena.run_eval \
            #     --dataset ${data_root}/${DATASET} \
            #     --save_dir results/arena \
            #     --model_name_or_path ${model_root1}/${MODEL} \
            #     --tokenizer_name_or_path ${model_root1}/${MODEL} \
            #     --cost_model_name_or_path ${model_root1}/beaver-7b-v1.0-reward \
            #     --eval_batch_size 200 \
            #     --num_samples 200 \
            #     --max_new_tokens 128 \
            #     --topk_ablate 0 200 400 600 800 1000 1200 1500 2000 3000 4000 5000 6000 7000 8000 9000 10000 12000 14000 16000 18000 20000 \
            #     --use_chat_format \
            #     --red_peft_path ${PEFT_PATH[@]} \
            #     --blue_peft_path ${GUIDED_PEFT_PATH[@]} \
            #     --guided_generation \
            #     --index_path ${data_root}/hooked_llama/neuron_activation/${MODEL}_${PEFT_NAME[-1]}_${index_path}.pt \
            #     --value_path ${data_root}/hooked_llama/neuron_activation/${MODEL}_${PEFT_NAME[-1]}_${index_path}.pt 
            
            # python -m eval.arena.run_eval \
            #     --dataset ${data_root}/${DATASET} \
            #     --save_dir results/arena \
            #     --model_name_or_path ${model_root2}/${MODEL} \
            #     --tokenizer_name_or_path ${model_root2}/${MODEL} \
            #     --cost_model_name_or_path ${model_root1}/beaver-7b-v1.0-cost \
            #     --eval_batch_size 200 \
            #     --num_samples 1000 \
            #     --max_new_tokens 128 \
            #     --topk_ablate 0 200 400 600 800 1000 1200 1500 2000 3000 4000 5000 6000 7000 8000 9000 10000 12000 14000 16000 18000 20000 \
            #     --use_chat_format \
            #     --patch_mean \
            #     --index_path ${data_root}/hooked_llama/neuron_activation/${MODEL}_${PEFT_NAME[-1]}_${index_path}.pt \
            #     --value_path ${data_root}/hooked_llama/neuron_activation/${MODEL}_${PEFT_NAME[-1]}_${index_path}.pt 
            
            # python -m eval.arena.run_eval \
            #     --dataset ${data_root}/${DATASET} \
            #     --save_dir results/arena \
            #     --model_name_or_path ${model_root2}/${MODEL} \
            #     --tokenizer_name_or_path ${model_root2}/${MODEL} \
            #     --cost_model_name_or_path ${model_root1}/beaver-7b-v1.0-cost \
            #     --eval_batch_size 200 \
            #     --num_samples 200 \
            #     --max_new_tokens 128 \
            #     --topk_ablate 10000 \
            #     --use_chat_format \
            #     --blue_peft_path ${PEFT_PATH[@]} \
            #     --generation_startswith Sure 

                
            # python -m eval.arena.run_eval \
            #     --dataset ${data_root}/${DATASET} \
            #     --save_dir results/arena \
            #     --model_name_or_path ${model_root1}/${MODEL} \
            #     --tokenizer_name_or_path ${model_root1}/${MODEL} \
            #     --cost_model_name_or_path ${model_root1}/beaver-7b-v1.0-cost \
            #     --eval_batch_size 200 \
            #     --num_samples 200 \
            #     --max_new_tokens 128 \
            #     --topk_ablate 0 200 400 600 800 1000 1200 1500 2000 3000 4000 5000 6000 7000 8000 9000 10000 12000 14000 16000 18000 20000 \
            #     --use_chat_format \
            #     --blue_peft_path ${PEFT_PATH[@]} \
            #     --red_peft_path ${GUIDED_PEFT_PATH[@]} \
            #     --index_path ${data_root}/hooked_llama/neuron_activation/Llama-2-7b-hf_${PEFT_NAME[-1]}_${index_path}.pt \
            #     --value_path ${data_root}/hooked_llama/neuron_activation/Llama-2-7b-hf_${PEFT_NAME[-1]}_${index_path}.pt \
            #     --guided_generation 
                # --cache_difference

            # python -m eval.arena.run_eval \
            #     --dataset ${data_root}/${DATASET} \
            #     --save_dir results/arena \
            #     --model_name_or_path ${model_root1}/${MODEL} \
            #     --tokenizer_name_or_path ${model_root1}/${MODEL} \
            #     --cost_model_name_or_path ${model_root1}/beaver-7b-v1.0-reward \
            #     --eval_batch_size 200 \
            #     --num_samples 200 \
            #     --max_new_tokens 128 \
            #     --topk_ablate 0 200 400 600 800 1000 1200 1500 2000 3000 4000 5000 6000 7000 8000 9000 10000 12000 14000 16000 18000 20000 \
            #     --use_chat_format \
            #     --blue_peft_path ${PEFT_PATH[@]} \
            #     --index_path ${data_root}/hooked_llama/neuron_activation/Llama-2-7b-hf_sharegpt_ia3_ff_${seed}_hh_harmless_dpo_ia3_ff_${index_path}.pt \
            #     --value_path ${data_root}/hooked_llama/neuron_activation/Llama-2-7b-hf_sharegpt_ia3_ff_${seed}_hh_harmless_dpo_ia3_ff_${index_path}.pt \
            #     --guided_generation 
                
            # python -m eval.arena.run_eval \
            #     --dataset ${data_root}/${DATASET} \
            #     --save_dir results/arena \
            #     --model_name_or_path ${model_root1}/${MODEL} \
            #     --tokenizer_name_or_path ${model_root1}/${MODEL} \
            #     --cost_model_name_or_path ${model_root1}/beaver-7b-v1.0-reward \
            #     --eval_batch_size 200 \
            #     --num_samples 200 \
            #     --max_new_tokens 128 \
            #     --topk_ablate 0 200 400 600 800 1000 1200 1500 2000 3000 4000 5000 6000 7000 8000 9000 10000 12000 14000 16000 18000 20000 \
            #     --use_chat_format \
            #     --red_peft_path ${PEFT_PATH[0]} \
            #     --blue_peft_path ${PEFT_PATH[@]} \
            #     --index_path ${data_root}/hooked_llama/neuron_activation/Llama-2-7b-hf_sharegpt_ia3_ff_${seed}_hh_harmless_dpo_ia3_ff_${index_path}.pt \
            #     --value_path ${data_root}/hooked_llama/neuron_activation/Llama-2-7b-hf_sharegpt_ia3_ff_${seed}_hh_harmless_dpo_ia3_ff_${index_path}.pt \
            #     --guided_generation 
        
    done
done
done

# for model in hh_harmless hh_helpful
# do
# for seed in 1
# do
#     PEFT_NAME=(sharegpt_ia3_ff_${seed} sharegpt_ia3_ff_${seed}_${model}_dpo_ia3_ff)
#     # PEFT_NAME=(sharegpt_lora_qv sharegpt_lora_qv_merged_hh_harmless_dpo_lora_qv)
#     GUIDED_PEFT_NAME=(sharegpt_ia3_ff_${seed} sharegpt_ia3_ff_${seed}_hh_harmless_dpo_ia3_ff)
#     MODEL=Llama-2-7b-hf
#     PEFT_PATH=()
#     for ((i=0; i<${#PEFT_NAME[*]}; ++i)) 
#     do
#         PEFT_PATH[i]=${model_root2}/${MODEL}_${PEFT_NAME[i]}
#         GUIDED_PEFT_PATH[i]=${model_root2}/${MODEL}_${GUIDED_PEFT_NAME[i]}
#     done
#     for dataset in beavertails
#     do
#         INDEX_PATH=(
#             # base_vs_dpo_on_${dataset}_base_completion
#             # base_vs_dpo_on_${dataset}_dpo_completion
#             sft_vs_dpo_on_${dataset}_sft_completion
#             # sft_vs_dpo_on_${dataset}_dpo_completion
#         )
#         #  800 1200 2000 3000 4000 5000
#         for index_path in ${INDEX_PATH[@]}
#         do
#             python -m eval.arena.run_eval \
#                 --dataset ${data_root}/${DATASET} \
#                 --save_dir results/arena \
#                 --model_name_or_path ${model_root1}/${MODEL} \
#                 --tokenizer_name_or_path ${model_root1}/${MODEL} \
#                 --cost_model_name_or_path ${model_root1}/beaver-7b-v1.0-cost \
#                 --eval_batch_size 100 \
#                 --num_samples 1000 \
#                 --max_new_tokens 256 \
#                 --topk_ablate 10000 \
#                 --use_chat_format \
#                 --red_peft_path ${PEFT_PATH[0]} \
#                 --blue_peft_path ${PEFT_PATH[@]} \
#                 --index_path ${data_root}/hooked_llama/neuron_activation/Llama-2-7b-hf_${GUIDED_PEFT_NAME[-1]}_${index_path}.pt \
#                 --value_path ${data_root}/hooked_llama/neuron_activation/Llama-2-7b-hf_${GUIDED_PEFT_NAME[-1]}_${index_path}.pt \
#                 --guided_generation \
#                 --cache_difference
                
#             # python -m eval.arena.run_eval \
#             #     --dataset ${data_root}/${DATASET} \
#             #     --save_dir results/arena \
#             #     --model_name_or_path ${model_root1}/${MODEL} \
#             #     --tokenizer_name_or_path ${model_root1}/${MODEL} \
#             #     --cost_model_name_or_path ${model_root1}/beaver-7b-v1.0-cost \
#             #     --eval_batch_size 200 \
#             #     --num_samples 200 \
#             #     --max_new_tokens 128 \
#             #     --topk_ablate 0 200 400 600 800 1000 1200 1500 2000 3000 4000 5000 6000 7000 8000 9000 10000 12000 14000 16000 18000 20000 \
#             #     --use_chat_format \
#             #     --blue_peft_path ${PEFT_PATH[@]} \
#             #     --red_peft_path ${GUIDED_PEFT_PATH[@]} \
#             #     --index_path ${data_root}/hooked_llama/neuron_activation/Llama-2-7b-hf_${PEFT_NAME[-1]}_${index_path}.pt \
#             #     --value_path ${data_root}/hooked_llama/neuron_activation/Llama-2-7b-hf_${PEFT_NAME[-1]}_${index_path}.pt \
#             #     --guided_generation 
#                 # --cache_difference

#             # python -m eval.arena.run_eval \
#             #     --dataset ${data_root}/${DATASET} \
#             #     --save_dir results/arena \
#             #     --model_name_or_path ${model_root1}/${MODEL} \
#             #     --tokenizer_name_or_path ${model_root1}/${MODEL} \
#             #     --cost_model_name_or_path ${model_root1}/beaver-7b-v1.0-reward \
#             #     --eval_batch_size 200 \
#             #     --num_samples 200 \
#             #     --max_new_tokens 128 \
#             #     --topk_ablate 0 200 400 600 800 1000 1200 1500 2000 3000 4000 5000 6000 7000 8000 9000 10000 12000 14000 16000 18000 20000 \
#             #     --use_chat_format \
#             #     --blue_peft_path ${PEFT_PATH[@]} \
#             #     --index_path ${data_root}/hooked_llama/neuron_activation/Llama-2-7b-hf_sharegpt_ia3_ff_${seed}_hh_harmless_dpo_ia3_ff_${index_path}.pt \
#             #     --value_path ${data_root}/hooked_llama/neuron_activation/Llama-2-7b-hf_sharegpt_ia3_ff_${seed}_hh_harmless_dpo_ia3_ff_${index_path}.pt \
#             #     --guided_generation 
                
#             # python -m eval.arena.run_eval \
#             #     --dataset ${data_root}/${DATASET} \
#             #     --save_dir results/arena \
#             #     --model_name_or_path ${model_root1}/${MODEL} \
#             #     --tokenizer_name_or_path ${model_root1}/${MODEL} \
#             #     --cost_model_name_or_path ${model_root1}/beaver-7b-v1.0-reward \
#             #     --eval_batch_size 200 \
#             #     --num_samples 200 \
#             #     --max_new_tokens 128 \
#             #     --topk_ablate 0 200 400 600 800 1000 1200 1500 2000 3000 4000 5000 6000 7000 8000 9000 10000 12000 14000 16000 18000 20000 \
#             #     --use_chat_format \
#             #     --red_peft_path ${PEFT_PATH[0]} \
#             #     --blue_peft_path ${PEFT_PATH[@]} \
#             #     --index_path ${data_root}/hooked_llama/neuron_activation/Llama-2-7b-hf_sharegpt_ia3_ff_${seed}_hh_harmless_dpo_ia3_ff_${index_path}.pt \
#             #     --value_path ${data_root}/hooked_llama/neuron_activation/Llama-2-7b-hf_sharegpt_ia3_ff_${seed}_hh_harmless_dpo_ia3_ff_${index_path}.pt \
#             #     --guided_generation 
#         done
#     done
# done
# done
# INDEX_PATH=(
#     base_vs_dpo_lora_on_beavertails_base_completion
#     base_vs_dpo_lora_on_beavertails_dpo_completion
#     sft_lora_vs_dpo_lora_on_beavertails_sft_completion
#     sft_lora_vs_dpo_lora_on_beavertails_dpo_completion
# )

# for index in ${INDEX_PATH[@]}
# do
#     python -m eval.arena.run_eval \
#         --dataset ${data_root}/data/raw_train/BeaverTails/test.jsonl \
#         --save_dir results/arena \
#         --model_name_or_path ${model_root1}/${MODEL} \
#         --tokenizer_name_or_path ${model_root1}/${MODEL} \
#         --cost_model_name_or_path ${model_root1}/beaver-7b-v1.0-cost \
#         --eval_batch_size 200 \
#         --num_samples 200 \
#         --max_new_tokens 128 \
#         --topk_ablate 20000 40000 60000 80000 120000 140000 160000 180000 200000 220000 240000 260000 280000 300000 320000 341248 \
#         --use_chat_format \
#         --blue_peft_path ${GUIDED_PEFT_PATH[@]} \
#         --index_path ${data_root}/hooked_llama/neuron_activation/Llama-2-7b-hf_${index}.pt \
#         --value_path ${data_root}/hooked_llama/neuron_activation/Llama-2-7b-hf_${index}.pt \
#         --guided_generation 
# done

# for index in ${INDEX_PATH[@]}
# do
#     python -m eval.arena.run_eval \
#         --dataset ${data_root}/data/raw_train/BeaverTails/test.jsonl \
#         --save_dir results/arena \
#         --model_name_or_path ${model_root1}/${MODEL} \
#         --tokenizer_name_or_path ${model_root1}/${MODEL} \
#         --cost_model_name_or_path ${model_root1}/beaver-7b-v1.0-cost \
#         --eval_batch_size 200 \
#         --num_samples 200 \
#         --max_new_tokens 128 \
#         --topk_ablate 0 200 400 600 800 1000 1200 1500 2000 2500 3000 3500 4000 4500 5000 6000 7000 8000 9000 10000 12000 14000 16000 18000 20000 \
#         --use_chat_format \
#         --blue_peft_path ${PEFT_PATH[@]} \
#         --index_path ${data_root}/hooked_llama/neuron_activation/Llama-2-7b-hf_${index}.pt \
#         --value_path ${data_root}/hooked_llama/neuron_activation/Llama-2-7b-hf_${index}.pt \
#         --patch_mean
# done

# for index in ${INDEX_PATH[@]}
# do
#     python -m eval.arena.run_eval \
#         --dataset ${data_root}/data/raw_train/BeaverTails/test.jsonl \
#         --save_dir results/arena \
#         --model_name_or_path ${model_root1}/${MODEL} \
#         --tokenizer_name_or_path ${model_root1}/${MODEL} \
#         --cost_model_name_or_path ${model_root1}/beaver-7b-v1.0-cost \
#         --eval_batch_size 200 \
#         --num_samples 200 \
#         --max_new_tokens 128 \
#         --topk_ablate 0 200 400 600 800 1000 1200 1500 2000 2500 3000 3500 4000 4500 5000 6000 7000 8000 9000 10000 12000 14000 16000 18000 20000 \
#         --use_chat_format \
#         --blue_peft_path ${PEFT_PATH[@]} \
#         --index_path ${data_root}/hooked_llama/neuron_activation/Llama-2-7b-hf_${index}.pt \
#         --value_path ${data_root}/hooked_llama/neuron_activation/Llama-2-7b-hf_${index}.pt \
#         --patch_zero
# done

# for index in ${INDEX_PATH[@]}
# do
#     python -m eval.arena.run_eval \
#         --dataset ${data_root}/data/raw_train/BeaverTails/test.jsonl \
#         --save_dir results/arena \
#         --model_name_or_path ${model_root1}/${MODEL} \
#         --tokenizer_name_or_path ${model_root1}/${MODEL} \
#         --cost_model_name_or_path ${model_root1}/beaver-7b-v1.0-cost \
#         --eval_batch_size 200 \
#         --num_samples 200 \
#         --max_new_tokens 128 \
#         --topk_ablate 0 200 400 600 800 1000 1200 1500 2000 2500 3000 3500 4000 4500 5000 6000 7000 8000 9000 10000 12000 14000 16000 18000 20000 \
#         --use_chat_format \
#         --blue_peft_path ${PEFT_PATH[@]} \
#         --index_path ${data_root}/hooked_llama/neuron_activation/Llama-2-7b-hf_${index}.pt \
#         --value_path ${data_root}/hooked_llama/neuron_activation/Llama-2-7b-hf_${index}.pt \
#         --add_noise
# done

# PEFT_NAME=(sharegpt_ia3_ff_1 sharegpt_ia3_ff_1_hh_harmless_dpo_ia3_ff)
# MODEL=Llama-2-7b-hf
# PEFT_PATH=()
# for ((i=0; i<${#PEFT_NAME[*]}; ++i)) 
# do
# PEFT_PATH[i]=${model_root2}/${MODEL}_${PEFT_NAME[i]}
# done

# for index in ${INDEX_PATH[@]}
# do
#     python -m eval.arena.run_eval \
#         --dataset ${data_root}/data/raw_train/BeaverTails/test.jsonl \
#         --save_dir results/arena \
#         --model_name_or_path ${model_root1}/${MODEL} \
#         --tokenizer_name_or_path ${model_root1}/${MODEL} \
#         --cost_model_name_or_path ${model_root1}/beaver-7b-v1.0-cost \
#         --eval_batch_size 200 \
#         --num_samples 200 \
#         --max_new_tokens 128 \
#         --topk_ablate 0 200 400 600 800 1000 1200 1500 2000 2500 3000 3500 4000 4500 5000 6000 7000 8000 9000 10000 12000 14000 16000 18000 20000 \
#         --use_chat_format \
#         --blue_peft_path ${PEFT_PATH[@]} \
#         --red_peft_path ${PEFT_PATH[0]} \
#         --index_path ${data_root}/hooked_llama/neuron_activation/Llama-2-7b-hf_${index}.pt \
#         --value_path ${data_root}/hooked_llama/neuron_activation/Llama-2-7b-hf_${index}.pt \
#         --guided_generation \
#         --generation_startswith Sure
# done

# for index in ${INDEX_PATH[@]}
# do
#     python -m eval.arena.run_eval \
#         --dataset ${data_root}/data/raw_train/BeaverTails/test.jsonl \
#         --save_dir results/arena \
#         --model_name_or_path ${model_root1}/${MODEL} \
#         --tokenizer_name_or_path ${model_root1}/${MODEL} \
#         --cost_model_name_or_path ${model_root1}/beaver-7b-v1.0-cost \
#         --eval_batch_size 200 \
#         --num_samples 200 \
#         --max_new_tokens 128 \
#         --topk_ablate 0 200 400 600 800 1000 1200 1500 2000 2500 3000 3500 4000 4500 5000 6000 7000 8000 9000 10000 12000 14000 16000 18000 20000 \
#         --use_chat_format \
#         --blue_peft_path ${PEFT_PATH[@]} \
#         --red_peft_path ${PEFT_PATH[0]} \
#         --index_path ${data_root}/hooked_llama/neuron_activation/Llama-2-7b-hf_${index}.pt \
#         --value_path ${data_root}/hooked_llama/neuron_activation/Llama-2-7b-hf_${index}.pt \
#         --guided_generation 
# done

#  800 1200 2000 3000 4000 5000

# python -m eval.arena.run_eval \
#     --dataset ${data_root}/data/raw_train/BeaverTails/test.jsonl \
#     --save_dir results/arena \
#     --model_name_or_path ${model_root1}/${MODEL} \
#     --tokenizer_name_or_path ${model_root1}/${MODEL} \
#     --cost_model_name_or_path ${model_root1}/beaver-7b-v1.0-cost \
#     --eval_batch_size 100 \
#     --num_samples 200 \
#     --max_new_tokens 128 \
#     --topk_ablate 0 \
#     --use_chat_format \
#     --blue_peft_path ${PEFT_PATH[@]} 

# python -m eval.arena.run_eval \
#     --dataset ${data_root}/data/raw_train/BeaverTails/test.jsonl \
#     --save_dir results/arena \
#     --model_name_or_path ${model_root1}/${MODEL} \
#     --tokenizer_name_or_path ${model_root1}/${MODEL} \
#     --cost_model_name_or_path ${model_root1}/beaver-7b-v1.0-cost \
#     --eval_batch_size 100 \
#     --num_samples 200 \
#     --max_new_tokens 128 \
#     --topk_ablate 0 200 400 600 800 1000 1200 1500 2000 2500 3000 3500 4000 4500 5000 6000 7000 8000 9000 10000 12000 14000 16000 18000 20000 \
#     --use_chat_format \
#     --blue_peft_path ${PEFT_PATH[@]} \
#     --index_path ${data_root}/hooked_llama/neuron_activation/Llama-2-7b-hf_sharegpt_ia3_ff_1_hh_harmless_dpo_ia3_ff_on_beavertails.pt \
#     --value_path ${data_root}/hooked_llama/neuron_activation/Llama-2-7b-hf_sharegpt_ia3_ff_1_hh_harmless_dpo_ia3_ff_on_beavertails.pt \
#     --guided_generation 
# for PEFT_NAME in hh_harmless_pt_kq_1 hh_harmless_ia3_ff_1 hh_harmless_lora_kq_1
# do
# python -m eval.arena.run_eval \
#     --dataset ${data_root}/data/raw_train/BeaverTails/test.jsonl \
#     --save_dir results/arena \
#     --model_name_or_path ${model_root}/${MODEL} \
#     --tokenizer_name_or_path ${model_root}/${MODEL} \
#     --cost_model_name_or_path ${model_root}/beaver-7b-v1.0-cost \
#     --eval_batch_size 20 \
#     --num_samples 200 \
#     --topk_ablate 0 240 400 600 800 1000 1200 1500 2000 2500 3000 3500 4000 4500 5000 \
#     --use_chat_format \
#     --blue_peft_path ${data_root}/output/${MODEL}_${PEFT_NAME} \
#     --index_path ${data_root}/hooked_llama/neuron_activation/llama-7b-hf_peft_ensemble_on_beavertails.pt \
#     --value_path ${data_root}/hooked_llama/neuron_activation/llama-7b-hf_peft_ensemble_on_beavertails.pt \
#     --patch_mean
# done
# for ((i=0; i<${#DATA_SET[*]}; ++i))
# do
# python -m eval.arena.run_eval \
#     --dataset ${data_root}/data/raw_train/BeaverTails/test.jsonl \
#     --save_dir results/arena \
#     --model_name_or_path ${model_root}/${MODEL} \
#     --tokenizer_name_or_path ${model_root}/${MODEL} \
#     --cost_model_name_or_path ${model_root}/beaver-7b-v1.0-cost \
#     --eval_batch_size 20 \
#     --num_samples 200 \
#     --topk_ablate 0 240 400 600 800 1000 1200 1500 2000 2500 3000 3500 4000 \
#     --use_chat_format \
#     --blue_peft_path ${data_root}/output/${MODEL}_${DATA_SET[i]}_${PEFT_NAME} \
    # --index_path ${data_root}/hooked_llama/neuron_activation/${MODEL}_${DATA_SET[i]}_${PEFT_NAME}.pt \
    # --value_path ${data_root}/hooked_llama/neuron_activation/${MODEL}_${DATA_SET[i]}_${PEFT_NAME}.pt 
    # --patch_mean
# done

# python -m eval.arena.run_eval \
#     --dataset ${data_root}/data/raw_train/BeaverTails/test.jsonl \
#     --save_dir results/arena \
#     --model_name_or_path ${model_root}/${MODEL} \
#     --tokenizer_name_or_path ${model_root}/${MODEL} \
#     --cost_model_name_or_path ${model_root}/beaver-7b-v1.0-cost \
#     --eval_batch_size 20 \
#     --num_samples 200 \
#     --topk_ablate 0 240 400 600 800 1000 1200 1500 2000 2500 3000 3500 4000  \
#     --use_chat_format \
#     --blue_peft_path output/${MODEL}_${PEFT_NAME} \
#     --index_path hooked_llama/data/lora_harmless_kq_1.pt \
#     --intersect_index_path hooked_llama/data/hh_helpful_kq_1.pt \
#     --value_path hooked_llama/data/lora_harmless_kq_1.pt \
#     --patch_mean

# python -m eval.arena.run_eval \
#     --dataset ${data_root}/data/raw_train/BeaverTails/test.jsonl \
#     --save_dir results/arena \
#     --model_name_or_path ${model_root}/${MODEL} \
#     --tokenizer_name_or_path ${model_root}/${MODEL} \
#     --cost_model_name_or_path ${model_root}/beaver-7b-v1.0-cost \
#     --eval_batch_size 20 \
#     --num_samples 200 \
#     --topk_ablate 0 240 400 600 800 1000 1200 1500 2000 2500 3000 3500 4000  \
#     --use_chat_format \
#     --blue_peft_path output/${MODEL}_${PEFT_NAME} \
#     --index_path hooked_llama/data/lora_harmless_kq_1.pt \
#     --ignore_index_path hooked_llama/data/hh_helpful_kq_1.pt \
#     --value_path hooked_llama/data/lora_harmless_kq_1.pt \
#     --patch_mean
# python -m eval.arena.run_eval \
#     --dataset ${data_root}/data/raw_train/BeaverTails/test.jsonl \
#     --save_dir results/arena \
#     --model_name_or_path ${model_root}/${MODEL} \
#     --tokenizer_name_or_path ${model_root}/${MODEL} \
#     --cost_model_name_or_path ${model_root}/beaver-7b-v1.0-cost \
#     --eval_batch_size 20 \
#     --num_samples 200 \
#     --topk_ablate 0 240 400 600 800 1000 1200 1500 2000 2500 3000 3500 4000  \
#     --use_chat_format \
#     --blue_peft_path output/${MODEL}_${PEFT_NAME} \
#     --index_path hooked_llama/data/${PEFT_NAME}_completion.pt \
#     --value_path hooked_llama/data/${PEFT_NAME}_prompt.pt \
#     --patch_mean

# python -m eval.arena.run_eval \
#     --dataset ${data_root}/data/raw_train/BeaverTails/test.jsonl \
#     --save_dir results/arena \
#     --model_name_or_path ${model_root}/${MODEL} \
#     --tokenizer_name_or_path ${model_root}/${MODEL} \
#     --cost_model_name_or_path ${model_root}/beaver-7b-v1.0-cost \
#     --eval_batch_size 20 \
#     --num_samples 200 \
#     --topk_ablate 0 240 400 600 800 1000 1200 1500 2000 2500 3000 3500 4000  \
#     --use_chat_format \
#     --blue_peft_path output/${MODEL}_${PEFT_NAME} \
#     --index_path hooked_llama/data/${PEFT_NAME}_prompt.pt \
#     --value_path hooked_llama/data/${PEFT_NAME}_completion.pt \
#     --patch_mean
# python -m eval.arena.run_eval \
#     --dataset ${data_root}/data/raw_train/BeaverTails/test.jsonl \
#     --save_dir results/arena \
#     --model_name_or_path ${model_root}/${MODEL} \
#     --tokenizer_name_or_path ${model_root}/${MODEL} \
#     --cost_model_name_or_path ${model_root}/beaver-7b-v1.0-cost \
#     --eval_batch_size 20 \
#     --num_samples 200 \
#     --topk_ablate 0 1 5 10 20 40 80 120 160 200 240 280 320 400 480 600 800 1000 1200 1500 2000 3000 4000 \
#     --index_path hooked_llama/data/flan_beaver_intersect.pt \
#     --use_chat_format \
#     --blue_peft_path output/${MODEL}_beavertails_${LORA_MODULE}_1 \
#     --patch_mean

# python -m eval.arena.run_eval \
#     --dataset ${data_root}/data/raw_train/BeaverTails/test.jsonl \
#     --save_dir results/arena \
#     --model_name_or_path ${model_root}/${MODEL} \
#     --tokenizer_name_or_path ${model_root}/${MODEL} \
#     --cost_model_name_or_path ${model_root}/beaver-7b-v1.0-cost \
#     --eval_batch_size 20 \
#     --num_samples 200 \
#     --topk_ablate 0 1 5 10 20 40 80 120 160 200 240 280 320 400 480 600 800 1000 1200 1500 2000 3000 4000 \
#     --index_path hooked_llama/data/mse_index_1.pt \
#     --use_chat_format \
#     --blue_peft_path output/${MODEL}_lora_harmless_${LORA_MODULE}_1 \
#     --add_noise
# python -m eval.arena.run_eval \
#     --dataset ${data_root}/data/raw_train/BeaverTails/test.jsonl \
#     --save_dir results/arena \
#     --model_name_or_path ${model_root}/${MODEL} \
#     --tokenizer_name_or_path ${model_root}/${MODEL} \
#     --cost_model_name_or_path ${model_root}/beaver-7b-v1.0-cost \
#     --eval_batch_size 20 \
#     --num_samples 200 \
#     --topk_ablate 3000 \
#     --index_path hooked_llama/data/completion_index_1.pt \
#     --use_chat_format \
#     --blue_peft_path output/${MODEL}_lora_harmless_${LORA_MODULE}_1 \
#     --use_random_neurons

# python -m eval.arena.run_eval \
#     --dataset ${data_root}/data/raw_train/BeaverTails/test.jsonl \
#     --save_dir results/arena \
#     --model_name_or_path ${model_root}/${MODEL} \
#     --tokenizer_name_or_path ${model_root}/${MODEL} \
#     --cost_model_name_or_path ${model_root}/beaver-7b-v1.0-cost \
#     --eval_batch_size 20 \
#     --num_samples 200 \
#     --topk_ablate 0 1 5 10 20 40 80 120 160 200 240 280 320 400 480 600 800 1000 1200 1500 2000 3000 4000 \
#     --index_path hooked_llama/data/completion_index_mean_intersect.pt \
#     --use_chat_format \

# python -m eval.arena.run_eval \
#     --dataset ${data_root}/data/raw_train/BeaverTails/test.jsonl \
#     --save_dir results/arena \
#     --model_name_or_path ${model_root}/${MODEL} \
#     --tokenizer_name_or_path ${model_root}/${MODEL} \
#     --cost_model_name_or_path ${model_root}/beaver-7b-v1.0-cost \
#     --blue_peft_path output/${MODEL}_lora_harmless_mlp_all \
#     --eval_batch_size 30 \
#     --num_samples 30 \
#     --use_chat_format        

# python -m eval.mmlu.run_eval \
#     --ntrain 0 \
#     --data_dir ${data_root}/data/eval/mmlu \
#     --save_dir results/mmlu/${MODEL}_lora_flan_0shot_hooked \
#     --model_name_or_path ${model_root}/${MODEL} \
#     --tokenizer_name_or_path ${model_root}/${MODEL} \
#     --eval_batch_size 50 \
#     --blue_peft_path output/${MODEL}_lora_flan \
#     --hooked