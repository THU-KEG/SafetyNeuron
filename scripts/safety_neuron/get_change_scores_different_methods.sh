export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export HF_DATASETS_CACHE=/data1/cjh/.cache
data_root=/data1/cjh/Alignment
model_root=/data2
BASE_MODEL=(gemma-7b Mistral-7B-v0.1)
BATCH_SIZE=20
NUM_SAMPLE=200

DATASET=(data/eval/hh_rlhf_harmless/hh_harmless_data_test.jsonl)
DATASET_NAME=(hh_harmless)
for MODEL in ${BASE_MODEL[@]}
do
for ((j=0; j<${#DATASET[*]}; ++j))
do
    for seed in 1
    do
        for model in hh_harmless
        do
            PEFT_NAME=(sharegpt_ia3_ff_${seed} sharegpt_ia3_ff_${seed}_${model}_dpo_ia3_ff)
            peft_path=()
            for ((i=0; i<${#PEFT_NAME[*]}; ++i)) 
            do
                peft_path[i]=${data_root}/output/${MODEL}_${PEFT_NAME[i]}
            done

            # python -m src.change_scores \
            #     --dataset ${data_root}/${DATASET[j]} \
            #     --output_file ${data_root}/hooked_llama/neuron_activation/${MODEL}_${PEFT_NAME[-1]}_sft_vs_dpo_on_${DATASET_NAME[j]}_sft_completion.pt \
            #     --model_name_or_path ${model_root}/${MODEL} \
            #     --tokenizer_name_or_path ${model_root}/${MODEL} \
            #     --first_peft_path ${peft_path[@]} \
            #     --second_peft_path ${peft_path[0]} \
            #     --eval_batch_size ${BATCH_SIZE} \
            #     --topk -1 \
            #     --num_samples ${NUM_SAMPLE} \
            #     --exclude_last_n 1 

            python -m src.change_scores \
                --dataset ${data_root}/${DATASET[j]} \
                --output_file ${data_root}/hooked_llama/neuron_activation/${MODEL}_${PEFT_NAME[-1]}_sft_vs_dpo_on_${DATASET_NAME[j]}_dpo_completion.pt \
                --model_name_or_path ${model_root}/${MODEL} \
                --tokenizer_name_or_path ${model_root}/${MODEL} \
                --first_peft_path ${peft_path[0]} \
                --second_peft_path ${peft_path[@]} \
                --eval_batch_size ${BATCH_SIZE} \
                --topk -1 \
                --num_samples ${NUM_SAMPLE} \
                --exclude_last_n 1 

            python -m src.change_scores \
                --dataset ${data_root}/${DATASET[j]} \
                --output_file ${data_root}/hooked_llama/neuron_activation/${MODEL}_${PEFT_NAME[-1]}_base_vs_dpo_on_${DATASET_NAME[j]}_base_completion.pt \
                --model_name_or_path ${model_root}/${MODEL} \
                --tokenizer_name_or_path ${model_root}/${MODEL} \
                --first_peft_path ${peft_path[@]} \
                --eval_batch_size ${BATCH_SIZE} \
                --topk -1 \
                --num_samples ${NUM_SAMPLE} \
                --exclude_last_n 1 

            python -m src.change_scores \
                --dataset ${data_root}/${DATASET[j]} \
                --output_file ${data_root}/hooked_llama/neuron_activation/${MODEL}_${PEFT_NAME[-1]}_base_vs_dpo_on_${DATASET_NAME[j]}_dpo_completion.pt \
                --model_name_or_path ${model_root}/${MODEL} \
                --tokenizer_name_or_path ${model_root}/${MODEL} \
                --second_peft_path ${peft_path[@]} \
                --eval_batch_size ${BATCH_SIZE} \
                --topk -1 \
                --num_samples ${NUM_SAMPLE} \
                --exclude_last_n 1 

            python -m src.change_scores \
                --dataset ${data_root}/${DATASET[j]} \
                --output_file ${data_root}/hooked_llama/neuron_activation/${MODEL}_${PEFT_NAME[-1]}_sft_vs_dpo_on_${DATASET_NAME[j]}_prompt.pt \
                --model_name_or_path ${model_root}/${MODEL} \
                --tokenizer_name_or_path ${model_root}/${MODEL} \
                --first_peft_path ${peft_path[@]} \
                --second_peft_path ${peft_path[0]} \
                --eval_batch_size ${BATCH_SIZE} \
                --topk -1 \
                --num_samples ${NUM_SAMPLE} \
                --exclude_last_n 1 \
                --prompt_only


            python -m src.change_scores \
                --dataset ${data_root}/${DATASET[j]} \
                --output_file ${data_root}/hooked_llama/neuron_activation/${MODEL}_${PEFT_NAME[-1]}_sft_vs_dpo_on_${DATASET_NAME[j]}_prompt_last.pt \
                --model_name_or_path ${model_root}/${MODEL} \
                --tokenizer_name_or_path ${model_root}/${MODEL} \
                --first_peft_path ${peft_path[@]} \
                --second_peft_path ${peft_path[0]} \
                --eval_batch_size ${BATCH_SIZE} \
                --topk -1 \
                --num_samples -1 \
                --exclude_last_n 1 \
                --prompt_only \
                --last_token
    
        done
    done
done
done

DATASET=(data/eval/hh_rlhf_harmless/hh_harmless_data_test.jsonl data/eval/hh_rlhf_harmless/hh_harmless_data_test.jsonl)
DATASET_NAME=(hh_harmless hh_helpful)
for MODEL in ${BASE_MODEL[@]}
do
for ((j=0; j<${#DATASET[*]}; ++j))
do
    python -m src.change_scores \
        --dataset ${data_root}/${DATASET[j]} \
        --output_file ${data_root}/output/activations/${MODEL}/${DATASET_NAME[j]}_prompt_last_token.pt \
        --model_name_or_path ${model_root}/${MODEL} \
        --tokenizer_name_or_path ${model_root}/${MODEL} \
        --eval_batch_size ${BATCH_SIZE} \
        --num_samples -1 \
        --prompt_only \
        --last_token \
        --cache_activation
done
done

python -m src.training_free_neurons