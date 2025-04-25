#!/bin/bash

for i in {1..10}
do
    python ladder_match.py \
        --test_type ladder_match \
        --source_data_path source_dataset/shapez_2d_source_data_min_1_max_100_each_100/source_data.jsonl \
        --output_dir_path result/shapez_2d/gpt-4o-mini/1_llm \
        --service_name qianduoduo \
        --model_name gpt-4o-mini \
        --response_name response \
        --prompt_type 1_llm \
        --min_steps 1 \
        --max_steps 100 \
        --each_number 100 \
        --stop_if_accuracy_below_threshold True \
        --accuracy_threshold 0.1 \
        --max_waiting_time 200 \
        --num_workers 5 \
        --save_interval 1 \
        --config_path reasoning_api_params \
        --threshold 3 \
        --initial_rank 1 \
        --try_times 1
done
