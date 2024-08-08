export CUDA_VISIBLE_DEVICES=3,4,5
export HF_DATASETS_CACHE=/data1/cjh/.cache
data_root=/data1/cjh/Alignment
model_root1=/data3/MODELS
model_root2=/data2
peft_root=${data_root}/output
BATCH_SIZE=100
NUM_SAMPLE=200
MAX_NEW_TOKENS=128
TOPK=(20000)
MODELS=(Llama-2-7b-hf)
dataset_dir=(data/eval/jailbreak_llms/jailbreak_llms.jsonl data/eval/red_team_attempts/red_team_attempts.jsonl data/eval/harmbench/harmbench.jsonl data/raw_train/BeaverTails/test.jsonl)
dataset_name=(jailbreak_llms red_team_attempts harmbench beavertails)
evaluate_only=false

for ((i=0; i<${#dataset_dir[*]}; ++i))
do
for MODEL in ${MODELS[@]}
do
    PEFT_NAME=(sharegpt_ia3_ff_1 sharegpt_ia3_ff_1_hh_harmless_dpo_ia3_ff)
    PEFT_PATH=()
    for ((j=0; j<${#PEFT_NAME[*]}; ++j)) 
    do
        PEFT_PATH[j]=${peft_root}/${MODEL}_${PEFT_NAME[j]}
    done
    for dataset in hh_harmless
    do
        INDEX_PATH=(
            # base_vs_dpo_on_${dataset}_base_completion
            # base_vs_dpo_on_${dataset}_dpo_completion
            # sft_vs_dpo_on_${dataset}_sft_completion
            # sft_vs_dpo_on_${dataset}_dpo_completion
            difference_on_hh_prompt_last_token
            std_on_hh_prompt_last_token
            # sft_vs_dpo_on_${dataset}_prompt
            # sft_vs_dpo_on_${dataset}_prompt_last
        )
        for index_path in ${INDEX_PATH[@]}
        do  
            # evaluation only
            if [ "$evaluate_only" = true ]
            then    
                python -m eval.arena.run_eval \
                    --dataset ${data_root}/${dataset_dir[i]} \
                    --save_dir results/safey_eval/${dataset_name[i]} \
                    --model_name_or_path ${model_root2}/${MODEL} \
                    --tokenizer_name_or_path ${model_root2}/${MODEL} \
                    --cost_model_name_or_path ${model_root1}/beaver-7b-v1.0-cost \
                    --eval_batch_size ${BATCH_SIZE} \
                    --num_samples ${NUM_SAMPLE} \
                    --max_new_tokens ${MAX_NEW_TOKENS} \
                    --topk_ablate 0 \
                    --use_chat_format 

                python -m eval.arena.run_eval \
                    --dataset ${data_root}/${dataset_dir[i]} \
                    --save_dir results/safey_eval/${dataset_name[i]} \
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
                    --dataset ${data_root}/${dataset_dir[i]} \
                    --save_dir results/safey_eval/${dataset_name[i]} \
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
                python -m eval.arena.run_eval \
                    --dataset ${data_root}/${dataset_dir[i]} \
                    --save_dir results/safey_eval/${dataset_name[i]} \
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
                    --dataset ${data_root}/${dataset_dir[i]} \
                    --save_dir results/safey_eval/${dataset_name[i]} \
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

                # python -m eval.arena.run_eval \
                #     --dataset ${data_root}/${dataset_dir[i]} \
                #     --save_dir results/safey_eval/${dataset_name[i]} \
                #     --model_name_or_path ${model_root2}/${MODEL} \
                #     --tokenizer_name_or_path ${model_root2}/${MODEL} \
                #     --cost_model_name_or_path ${model_root1}/beaver-7b-v1.0-cost \
                #     --eval_batch_size ${BATCH_SIZE} \
                #     --num_samples ${NUM_SAMPLE} \
                #     --max_new_tokens ${MAX_NEW_TOKENS} \
                #     --topk_ablate ${TOPK[@]} \
                #     --use_chat_format \
                #     --red_peft_path ${PEFT_PATH[@]}
            fi
        done
    done
done
done