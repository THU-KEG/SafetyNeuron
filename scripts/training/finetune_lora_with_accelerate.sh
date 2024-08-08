export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export HF_DATASETS_CACHE=/data1/cjh/.cache
export HF_TOKEN=
export WANDB_PROJECT='SFT'

data_root=/data1/cjh/Alignment
model_root=/data2
# model_root=${data_root}/output
MODEL=(gemma-7b)
# MODEL=${MODEL}_sharegpt_full
NUM_GPUS=5
BATCH_SIZE_PER_GPU=4
TOTAL_BATCH_SIZE=120
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))

# datasets=(hh_rlhf_harmless/hh_rlhf_harmful_data_train)
# save_name=(hh_harmful)
datasets=(sharegpt/sharegpt_data)
save_name=(sharegpt)

# prompt tuning
# for (( i=0; i<${#datasets[*]}; ++i))
# do
#     accelerate launch \
#         --use_deepspeed \
#         --deepspeed_config_file configs/ds_configs/stage2_no_offloading_accelerate.conf \
#         --mixed_precision bf16 \
#         --num_machines 1 \
#         --num_processes $NUM_GPUS \
#         training/finetune.py \
#         --train_file ${data_root}/data/processed/${datasets[$i]}.jsonl \
#         --model_name_or_path ${model_root}/${MODEL} \
#         --use_flash_attn \
#         --use_prompt_tuning \
#         --tokenizer_name ${model_root}/${MODEL} \
#         --use_slow_tokenizer \
#         --max_seq_length 2048 \
#         --preprocessing_num_workers 100 \
#         --checkpointing_steps epoch \
#         --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
#         --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
#         --learning_rate 1e-4 \
#         --lr_scheduler_type linear \
#         --warmup_ratio 0.03 \
#         --weight_decay 0. \
#         --num_train_epochs 3 \
#         --output_dir ${data_root}/output/${MODEL}_${save_name[$i]}_${seed}/ \
#         --report_to wandb \
#         --with_tracking \
#         --logging_steps 10 \
#         --num_virtual_tokens 64 \
#         --prompt_tuning_init_text "You are a helpful and harmless AI assistant." \
#         --seed ${seed} 
# done

# IA3 training
for seed in 42
do
for model in ${MODEL[@]}
do
    for (( i=0; i<${#datasets[*]}; ++i))
    do
        echo "Training ${model} model using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"
        accelerate launch \
            --use_deepspeed \
            --deepspeed_config_file configs/ds_configs/stage2_no_offloading_accelerate.conf \
            --mixed_precision bf16 \
            --num_machines 1 \
            --num_processes $NUM_GPUS \
            --main_process_port $((29500+${seed})) \
            src/training/finetune.py \
            --train_file ${data_root}/data/processed/${datasets[i]}.jsonl \
            --model_name_or_path ${model_root}/${model} \
            --use_flash_attn \
            --use_ia3 \
            --tokenizer_name ${model_root}/${model} \
            --use_slow_tokenizer \
            --max_seq_length 4096 \
            --preprocessing_num_workers 100 \
            --checkpointing_steps epoch \
            --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
            --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
            --learning_rate 1e-3 \
            --lr_scheduler_type cosine \
            --weight_decay 0.1 \
            --num_train_epochs 3 \
            --output_dir ${data_root}/output/${model}_${save_name[i]}_ia3_ff_${seed} \
            --report_to wandb \
            --with_tracking \
            --logging_steps 10 \
            --ia3_module down_proj \
            --feedforward_modules down_proj \
            --seed ${seed} \
            --gradient_checkpointing 
    done
done
done
# LoRA training
# for (( i=0; i<${#datasets[*]}; ++i))
# do
#     accelerate launch \
#         --use_deepspeed \
#         --deepspeed_config_file configs/ds_configs/stage2_no_offloading_accelerate.conf \
#         --mixed_precision bf16 \
#         --num_machines 1 \
#         --num_processes $NUM_GPUS \
#         --main_process_port $((29500+${seed})) \
#         training/finetune.py \
#         --train_file ${data_root}/data/processed/${datasets[i]}.jsonl \
#         --model_name_or_path ${model_root}/${MODEL} \
#         --use_flash_attn \
#         --tokenizer_name ${model_root}/${MODEL} \
#         --use_slow_tokenizer \
#         --max_seq_length 4096 \
#         --preprocessing_num_workers 64 \
#         --checkpointing_steps epoch \
#         --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
#         --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
#         --learning_rate 2e-5 \
#         --lr_scheduler_type cosine \
#         --weight_decay 0.1 \
#         --num_train_epochs 3 \
#         --output_dir ${data_root}/output/${MODEL}_${save_name[i]}_lora_qv_${seed} \
#         --report_to wandb \
#         --with_tracking \
#         --logging_steps 10 \
#         --use_lora \
#         --lora_rank 64 \
#         --lora_alpha 64 \
#         --lora_dropout 0 \
#         --lora_module v_proj q_proj \
#         --seed ${seed} \
#         --gradient_checkpointing
# done

# for (( i=0; i<${#datasets[*]}; ++i))
# do
#     accelerate launch \
#         --use_deepspeed \
#         --deepspeed_config_file configs/ds_configs/stage2_no_offloading_accelerate.conf \
#         --mixed_precision bf16 \
#         --num_machines 1 \
#         --num_processes $NUM_GPUS \
#         training/finetune.py \
#         --train_file ${data_root}/data/processed/${datasets[i]}.jsonl \
#         --model_name_or_path ${model_root}/${MODEL} \
#         --use_flash_attn \
#         --tokenizer_name ${model_root}/${MODEL} \
#         --use_slow_tokenizer \
#         --max_seq_length 4096 \
#         --preprocessing_num_workers 100 \
#         --checkpointing_steps epoch \
#         --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
#         --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
#         --learning_rate 2e-5 \
#         --lr_scheduler_type cosine \
#         --weight_decay 0.1 \
#         --num_train_epochs 3 \
#         --output_dir ${data_root}/output/${MODEL}_${save_name[i]}_full \
#         --report_to wandb \
#         --with_tracking \
#         --logging_steps 10 \
#         --seed 1 \
#         --gradient_checkpointing
# done

# python open_instruct/merge_lora.py \
#     --base_model_name_or_path ../hf_llama2_models/${MODEL_SIZE} \
#     --lora_model_name_or_path output/tulu_v2_${MODEL_SIZE}_lora/ \
#     --output_dir output/tulu_v2_${MODEL_SIZE}_lora_merged/ \
#     --save_tokenizer
