export CUDA_VISIBLE_DEVICES=0
export HF_DATASETS_CACHE=/data1/cjh/.cache
data_root=/data1/cjh/Alignment
model_root=/data2
MODEL=Llama-2-7b-hf
# model_root=${data_root}/output
# PEFT_NAME=harmfulqa_kq_1

INDEX_PATH=(
    # base_vs_dpo_on_beavertails_base_completion
    # base_vs_dpo_on_beavertails_dpo_completion
    # base_vs_dpo0_on_beavertails_base_completion
    # base_vs_dpo0_on_beavertails_dpo0_completion
    sft_vs_dpo_on_hh_harmless_sft_completion
    # sft_vs_dpo_on_beavertails_dpo_completion
    # sft0_vs_dpo0_on_beavertails_sft0_completion
    # sft0_vs_dpo0_on_beavertails_dpo0_completion
)

for index in ${INDEX_PATH[@]}
do
    python -m hooked_models.neuron2word \
        --model_name_or_path ${model_root}/${MODEL} \
        --tokenizer_name_or_path ${model_root}/${MODEL} \
        --index_path ${data_root}/hooked_llama/neuron_activation/Llama-2-7b-hf_sharegpt_ia3_ff_1_hh_harmless_dpo_ia3_ff_${index}.pt \
        --topk_neuron 10 \
        --topk_token 20

done

