export CUDA_VISIBLE_DEVICES=4,5,6,7
export HF_DATASETS_CACHE=/data1/cjh/.cache
data_root=/data1/cjh/Alignment
model_root1=/data3/MODELS
model_root2=/data2
peft_root=${data_root}/output
BATCH_SIZE=100
NUM_SAMPLE=200
MAX_NEW_TOKENS=128
MODELS=(Llama-2-7b-hf gemma-7b Mistral-7B-v0.1)
PEFT_NAME=(sharegpt_ia3_ff_1 sharegpt_ia3_ff_1_hh_harmless_dpo_ia3_ff)

for MODEL in ${MODELS[@]}
do 
    PEFT_PATH=(${peft_root}/${MODEL}_${PEFT_NAME[0]} ${peft_root}/${MODEL}_${PEFT_NAME[1]})
    # Evaluating llama 7B model using chain-of-thought
    # python -m eval.gsm.run_eval \
    #     --data_dir ${data_root}/data/eval/gsm/ \
    #     --max_num_examples ${NUM_SAMPLE} \
    #     --save_dir results/gsm/${MODEL}-cot-8shot \
    #     --model ${model_root2}/${MODEL} \
    #     --tokenizer ${model_root2}/${MODEL} \
    #     --eval_batch_size 20 \
    #     --n_shot 8 \
    #     --use_chat_format \
    #     --hooked
    
    #     python -m eval.gsm.run_eval \
    #     --data_dir ${data_root}/data/eval/gsm/ \
    #     --max_num_examples ${NUM_SAMPLE} \
    #     --save_dir results/gsm/${MODEL}-sft-cot-8shot \
    #     --model ${model_root2}/${MODEL} \
    #     --tokenizer ${model_root2}/${MODEL} \
    #     --eval_batch_size 20 \
    #     --n_shot 8 \
    #     --use_chat_format \
    #     --red_peft_path ${PEFT_NAME[0]} \
    #     --hooked

    #     python -m eval.gsm.run_eval \
    #     --data_dir ${data_root}/data/eval/gsm/ \
    #     --max_num_examples ${NUM_SAMPLE} \
    #     --save_dir results/gsm/${MODEL}-dpo-cot-8shot \
    #     --model ${model_root2}/${MODEL} \
    #     --tokenizer ${model_root2}/${MODEL} \
    #     --eval_batch_size 20 \
    #     --n_shot 8 \
    #     --use_chat_format \
    #     --red_peft_path ${PEFT_NAME[@]} \
    #     --hooked

        python -m eval.gsm.run_eval \
        --data_dir ${data_root}/data/eval/gsm/ \
        --max_num_examples ${NUM_SAMPLE} \
        --save_dir results/gsm/${MODEL}-patch-cot-8shot \
        --model ${model_root2}/${MODEL} \
        --tokenizer ${model_root2}/${MODEL} \
        --eval_batch_size 20 \
        --n_shot 8 \
        --use_chat_format \
        --blue_peft_path ${PEFT_PATH[@]} \
        --index_path ${data_root}/hooked_llama/neuron_activation/${MODEL}_${PEFT_NAME[-1]}_sft_vs_dpo_on_hh_harmless_sft_completion.pt \
        --hooked

        python -m eval.gsm.run_eval \
        --data_dir ${data_root}/data/eval/gsm/ \
        --max_num_examples ${NUM_SAMPLE} \
        --save_dir results/gsm/${MODEL}-sft-patch-cot-8shot \
        --model ${model_root2}/${MODEL} \
        --tokenizer ${model_root2}/${MODEL} \
        --eval_batch_size 20 \
        --n_shot 8 \
        --use_chat_format \
        --red_peft_path ${PEFT_PATH[0]} \
        --blue_peft_path ${PEFT_PATH[@]} \
        --index_path ${data_root}/hooked_llama/neuron_activation/${MODEL}_${PEFT_NAME[-1]}_sft_vs_dpo_on_hh_harmless_sft_completion.pt \
        --hooked
done