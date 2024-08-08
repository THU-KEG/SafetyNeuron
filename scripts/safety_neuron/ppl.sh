export CUDA_VISIBLE_DEVICES=4,5
export HF_DATASETS_CACHE=/data1/cjh/.cache
data_root=/data1/cjh/Alignment
model_root1=/data2
peft_root=${data_root}/output
TOPK=(20000)
MODELS=(gemma-7b)
# Llama-2-7b-hf Mistral-7B-v0.1 

for MODEL in ${MODELS[@]}
do
    PEFT_NAME=(sharegpt_ia3_ff_1 sharegpt_ia3_ff_1_hh_harmless_dpo_ia3_ff)
    PEFT_PATH=()
    for ((i=0; i<${#PEFT_NAME[*]}; ++i)) 
    do
        PEFT_PATH[i]=${peft_root}/${MODEL}_${PEFT_NAME[i]}
    done
    for dataset in hh_harmless
    do
        INDEX_PATH=(
            # base_vs_dpo_on_${dataset}_base_completion
            # base_vs_dpo_on_${dataset}_dpo_completion
            sft_vs_dpo_on_${dataset}_sft_completion
            # sft_vs_dpo_on_${dataset}_dpo_completion
        )
        for index_path in ${INDEX_PATH[@]}
        do  
            python -m hooked_models.ppl \
                --model_name_or_path ${model_root1}/${MODEL} \
                --tokenizer_name_or_path ${model_root1}/${MODEL} \
                --topk_ablate ${TOPK[@]}  \
                --blue_peft_path ${PEFT_PATH[@]} \
                --index_path ${data_root}/hooked_llama/neuron_activation/${MODEL}_${PEFT_NAME[-1]}_${index_path}.pt 
            
            python -m hooked_models.ppl \
                --model_name_or_path ${model_root1}/${MODEL} \
                --tokenizer_name_or_path ${model_root1}/${MODEL} \
                --topk_ablate ${TOPK[@]} \
                --red_peft_path ${PEFT_PATH[0]} \
                --blue_peft_path ${PEFT_PATH[@]} \
                --index_path ${data_root}/hooked_llama/neuron_activation/${MODEL}_${PEFT_NAME[-1]}_${index_path}.pt 

        done
    done
done