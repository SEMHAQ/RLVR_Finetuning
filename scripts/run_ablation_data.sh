#!/bin/bash
# 数据规模消融实验
# Usage: bash scripts/run_ablation_data.sh

set -e

echo "=== 数据规模消融实验 ==="
echo "开始时间: $(date)"

for size in 1000 3000 5000 7500; do
    echo ""
    echo ">>> 训练 data_size=$size"
    python scripts/train_grpo.py \
        --model Qwen/Qwen2.5-Math-1.5B \
        --dataset gsm8k \
        --reward rule \
        --lr 2e-5 \
        --batch_size 6 \
        --num_generations 3 \
        --max_completion_length 256 \
        --use_lora \
        --max_samples $size \
        --tag "ablation_data_${size}"

    echo ">>> 评测 data_size=$size"
    python scripts/eval.py \
        --model outputs/grpo_ablation_data_${size}/final \
        --dataset gsm8k \
        --tag "ablation_data_${size}"
done

echo ""
echo "=== 实验完成 ==="
echo "结束时间: $(date)"
