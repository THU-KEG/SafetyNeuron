# export CUDA_VISIBLE_DEVICES=4,5
export HF_DATASETS_CACHE=/data1/cjh/.cache
export HF_TOKEN=
export WANDB_PROJECT='DPO'

data_root=/data1/cjh/Alignment
# model_root=/data3/MODELS
model_root=/data2
model_root1=${data_root}/output
NUM_GPUS=4
BATCH_SIZE_PER_GPU=2
TOTAL_BATCH_SIZE=120
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
# BASE_MODEL=(gemma-7b Mistral-7B-v0.1)
# datasets=(data/raw_train/hh_rlhf_helpful/train.jsonl data/raw_train/reward_bench/safety.jsonl data/raw_train/reward_bench/reasoning.jsonl)
# save_name=(hh_helpful rewardbench_safety rewardbench_reasoning)
BASE_MODEL=(Llama-2-7b-hf Mistral-7B-v0.1 gemma-7b)
# datasets=(data/raw_train/H4_stack_exchange/math_stack_exchange.json data/raw_train/SHP/train.jsonl)
# save_name=(H4_stack_exchange shp)
datasets=(data/raw_train/IEFeedback/IEFeedback_OpenSource_Dataset.json)
save_name=(IEFeedback)
for seed in 1
do
for model in ${BASE_MODEL[@]:2}
do
    for (( i=0; i<${#datasets[*]}; ++i))
    do
        PEFT_NAME=sharegpt_ia3_ff_${seed}
        MODEL=${model}_${PEFT_NAME}
        echo "Training ${MODEL} model using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

        deepspeed --include localhost:4\
            --master_port $((29500+${seed})) \
            --module src.training.dpo \
            --deepspeed configs/ds_configs/stage2_no_offloading.conf \
            --beta 0.1 \
            --cache_dir /data1/cjh/.cache \
            --model_name_or_path ${model_root}/${model} \
            --tokenizer_name ${model_root}/${model} \
            --use_flash_attn True \
            --use_fast_tokenizer False \
            --remove_unused_columns False \
            --train_file ${data_root}/${datasets[i]} \
            --test_file ${data_root}/${datasets[i]} \
            --max_seq_length 4096 \
            --max_prompt_length 2048 \
            --preprocessing_num_workers 64 \
            --do_train \
            --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
            --per_device_eval_batch_size 3 \
            --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
            --learning_rate 1e-3 \
            --lr_scheduler_type cosine \
            --warmup_ratio 0.0 \
            --weight_decay 0.1 \
            --evaluation_strategy no \
            --logging_steps 1 \
            --save_strategy epoch \
            --num_train_epochs 3 \
            --output_dir ${model_root1}/${MODEL}_${save_name[i]}_dpo_ia3_ff \
            --report_to wandb \
            --bf16 \
            --gradient_checkpointing \
            --peft_name_or_path ${model_root1}/${MODEL}\
            --use_ia3 \
            --ia3_module down_proj \
            --feedforward_modules down_proj \
            --seed ${seed}
    done
done
done
# IA3 on full
# deepspeed --include localhost:0,1,2,3\
#     --master_port $((29500+${SEED})) \
#     --module training.dpo \
#     --deepspeed configs/ds_configs/stage2_no_offloading.conf \
#     --beta 0.1 \
#     --cache_dir /data1/cjh/.cache \
#     --model_name_or_path ${model_root1}/${MODEL} \
#     --tokenizer_name ${model_root1}/${MODEL} \
#     --use_flash_attn True \
#     --use_fast_tokenizer False \
#     --remove_unused_columns False \
#     --train_file ${data_root}/data/raw_train/hh_rlhf_harmless/train.jsonl \
#     --test_file ${data_root}/data/raw_train/hh_rlhf_harmless/test.jsonl \
#     --max_seq_length 4096 \
#     --max_prompt_length 2048 \
#     --preprocessing_num_workers 64 \
#     --do_train \
#     --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
#     --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
#     --learning_rate 1e-3 \
#     --lr_scheduler_type cosine \
#     --warmup_ratio 0.0 \
#     --weight_decay 0.1 \
#     --evaluation_strategy epoch \
#     --logging_steps 10 \
#     --save_strategy epoch \
#     --num_train_epochs 3 \
#     --output_dir ${model_root1}/${MODEL}_hh_harmless_dpo_ia3_ff_${SEED} \
#     --report_to wandb \
#     --bf16 \
#     --gradient_checkpointing \
#     --use_ia3 \
#     --ia3_module down_proj \
#     --feedforward_modules down_proj \
#     --seed ${SEED}

# LORA on full
# deepspeed --include localhost:4,5,6,7\
#     --master_port $((29500+${SEED})) \
#     --module training.dpo \
#     --deepspeed configs/ds_configs/stage2_no_offloading.conf \
#     --beta 0.1 \
#     --cache_dir /data1/cjh/.cache \
#     --model_name_or_path ${model_root1}/${MODEL} \
#     --tokenizer_name ${model_root1}/${MODEL} \
#     --use_flash_attn True \
#     --use_fast_tokenizer False \
#     --remove_unused_columns False \
#     --train_file ${data_root}/data/raw_train/hh_rlhf_harmless/train.jsonl \
#     --test_file ${data_root}/data/raw_train/hh_rlhf_harmless/test.jsonl \
#     --max_seq_length 4096 \
#     --max_prompt_length 2048 \
#     --preprocessing_num_workers 64 \
#     --do_train \
#     --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
#     --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
#     --learning_rate 2e-5 \
#     --lr_scheduler_type cosine \
#     --warmup_ratio 0.0 \
#     --weight_decay 0.1 \
#     --evaluation_strategy epoch \
#     --logging_steps 10 \
#     --save_strategy epoch \
#     --num_train_epochs 3 \
#     --output_dir ${model_root1}/${MODEL}_hh_harmless_dpo_lora_qv_${SEED} \
#     --report_to wandb \
#     --bf16 \
#     --gradient_checkpointing \
#     --use_lora \
#     --lora_rank 64 \
#     --lora_alpha 64 \
#     --lora_dropout 0.0 \
#     --lora_module v_proj q_proj \
#     --seed ${SEED}