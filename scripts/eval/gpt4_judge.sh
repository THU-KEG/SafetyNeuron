# Please make sure OPENAI_API_KEY is set in your environment variables
export OPENAI_API_KEY=

# use normal huggingface generation function
python -m eval.gpt4_judge.run_eval \
    --openai_engine gpt-4-turbo-2024-04-09 \
    --arena_result_path results/arena/Llama-2-7b-hf_sharegpt_ia3_ff_1_vs_Llama-2-7b-hf_sharegpt_ia3_ff_1_hh_harmless_dpo_ia3_ff/top0_guided_by_Llama-2-7b-hf_sharegpt_ia3_ff_1_hh_harmless_dpo_ia3_ff_idx_Llama-2-7b-hf_base_vs_dpo_on_beavertails_base_completion_table.csv \
    --save_dir results/gpt4_judge \
    --eval_batch_size 20 
    # --count_tokens