export CUDA_VISIBLE_DEVICES=4,5,6,7
export HF_DATASETS_CACHE=/data1/cjh/.cache
data_root=/data1/cjh/Alignment
model_root=/data2
peft_root=${data_root}/output
cost_model_root=/data3/MODELS
MODELS=(Llama-2-7b-hf gemma-7b Mistral-7B-v0.1)
PEFT_NAME=(sharegpt_ia3_ff_1 sharegpt_ia3_ff_1_hh_harmless_dpo_ia3_ff)
# TOPK=3000
NUM_SAMPLE=20
BATCH_SIZE=10
# dataset_dir=(data/eval/jailbreak_llms/jailbreak_llms.jsonl data/eval/hh_rlhf_harmless/hh_harmless_data_train.jsonl data/raw_train/BeaverTails/test.jsonl data/eval/harmbench/harmbench.jsonl data/eval/red_team_attempts/red_team_attempts.jsonl)
# dataset_name=(jailbreak_llms hh_harmless beavertails harmbench red_team_attempts)
dataset_dir=(data/eval/hh_rlhf_harmless/hh_harmless_data_train.jsonl)
dataset_name=(hh_harmless)
TOPK=(1500 1200 900 700 500 300 150 50 30 20 10)
SEED=(1 42 66 88 3419)

for MODEL in ${MODELS[@]:0:1}
do
    PEFT_PATH=()
    for ((i=0; i<${#PEFT_NAME[*]}; ++i)) 
    do
        PEFT_PATH[i]=${peft_root}/${MODEL}_${PEFT_NAME[i]}
    done

for ((i=0; i<${#dataset_dir[*]}; ++i)) 
do
    # python -m src.neuron_activation \
    #     --dataset ${data_root}/${dataset_dir[i]} \
    #     --output_dir ${data_root}/output/activations/harmless_prediction \
    #     --output_filename ${dataset_name[i]}_${NUM_SAMPLE}_${MODEL}_${PEFT_NAME[0]} \
    #     --model_name_or_path ${model_root}/${MODEL} \
    #     --tokenizer_name_or_path ${model_root}/${MODEL} \
    #     --cost_model_name_or_path ${cost_model_root}/beaver-7b-v1.0-cost \
    #     --index_path ${data_root}/hooked_llama/neuron_activation/${MODEL}_${PEFT_NAME[-1]}_sft_vs_dpo_on_hh_harmless_sft_completion.pt \
    #     --peft_path ${PEFT_PATH[0]} \
    #     --eval_batch_size ${BATCH_SIZE} \
    #     --num_samples ${NUM_SAMPLE} \
    #     --topk 1500

    # python -m src.neuron_activation \
    #     --dataset ${data_root}/${dataset_dir[i]} \
    #     --output_dir ${data_root}/output/activations/harmless_prediction \
    #     --output_filename ${dataset_name[i]}_${NUM_SAMPLE}_${MODEL}_${PEFT_NAME[-1]} \
    #     --model_name_or_path ${model_root}/${MODEL} \
    #     --tokenizer_name_or_path ${model_root}/${MODEL} \
    #     --cost_model_name_or_path ${cost_model_root}/beaver-7b-v1.0-cost \
    #     --index_path ${data_root}/hooked_llama/neuron_activation/${MODEL}_${PEFT_NAME[-1]}_sft_vs_dpo_on_hh_harmless_sft_completion.pt \
    #     --peft_path ${PEFT_PATH[@]} \
    #     --eval_batch_size ${BATCH_SIZE} \
    #     --num_samples ${NUM_SAMPLE} \
    #     --topk 1500

    python -m src.neuron_activation \
        --dataset ${data_root}/${dataset_dir[i]} \
        --output_dir ${data_root}/output/activations/harmless_prediction \
        --output_filename ${dataset_name[i]}_${NUM_SAMPLE}_${MODEL}_${PEFT} \
        --model_name_or_path ${model_root}/${MODEL} \
        --tokenizer_name_or_path ${model_root}/${MODEL} \
        --cost_model_name_or_path ${cost_model_root}/beaver-7b-v1.0-cost \
        --index_path ${data_root}/hooked_llama/neuron_activation/Llama-2-7b-hf_sharegpt_ia3_ff_1_hh_harmless_dpo_ia3_ff_sft_vs_dpo_on_hh_harmless_sft_completion.pt \
        --peft_path ${PEFT_PATH[0]} \
        --eval_batch_size ${BATCH_SIZE} \
        --num_samples ${NUM_SAMPLE} \
        --topk ${TOPK[@]} \
        --seed ${SEED[@]} \
        --use_random_neurons 

    # python -m src.neuron_activation \
    #     --dataset ${data_root}/${dataset_dir[i]} \
    #     --output_dir ${data_root}/output/activations/harmless_prediction \
    #     --output_filename ${dataset_name[i]}_${NUM_SAMPLE}_${MODEL}_${PEFT}_last \
    #     --model_name_or_path ${model_root}/${MODEL} \
    #     --tokenizer_name_or_path ${model_root}/${MODEL} \
    #     --cost_model_name_or_path ${cost_model_root}/beaver-7b-v1.0-cost \
    #     --index_path ${data_root}/hooked_llama/neuron_activation/Llama-2-7b-hf_sharegpt_ia3_ff_1_hh_harmless_dpo_ia3_ff_sft_vs_dpo_on_hh_harmless_sft_completion.pt \
    #     --peft_path ${data_root}/output/${MODEL}_${PEFT} \
    #     --eval_batch_size ${BATCH_SIZE} \
    #     --num_samples ${NUM_SAMPLE} \
    #     --topk ${TOPK[@]} \
    #     --seed ${SEED[@]} \
    #     --use_random_neurons \
    #     --last_layer_neurons
        
    # python -m src.neuron_activation \
    #     --dataset ${data_root}/${dataset_dir[i]} \
    #     --output_dir ${data_root}/output/activations/harmless_prediction \
    #     --output_filename ${dataset_name[i]}_${NUM_SAMPLE}_${MODEL}_${PEFT}_everywhere \
    #     --model_name_or_path ${model_root}/${MODEL} \
    #     --tokenizer_name_or_path ${model_root}/${MODEL} \
    #     --cost_model_name_or_path ${cost_model_root}/beaver-7b-v1.0-cost \
    #     --index_path ${data_root}/hooked_llama/neuron_activation/Llama-2-7b-hf_sharegpt_ia3_ff_1_hh_harmless_dpo_ia3_ff_sft_vs_dpo_on_hh_harmless_sft_completion.pt \
    #     --peft_path ${data_root}/output/${MODEL}_${PEFT} \
    #     --eval_batch_size ${BATCH_SIZE} \
    #     --num_samples ${NUM_SAMPLE} \
    #     --topk ${TOPK[@]} \
    #     --seed ${SEED[@]} \
    #     --use_random_neurons \
    #     --random_neurons_everywhere

done
done