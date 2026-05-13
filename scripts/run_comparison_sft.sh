#!/bin/bash
# SFT vs GRPO 对比实验
# Usage: bash scripts/run_comparison_sft.sh

set -e

echo "=== SFT vs GRPO 对比实验 ==="
echo "开始时间: $(date)"

# SFT baseline
echo ""
echo ">>> SFT 训练"
python scripts/train_sft.py \
    --model Qwen/Qwen2.5-Math-1.5B \
    --lr 2e-5 \
    --batch_size 6 \
    --epochs 3 \
    --use_lora \
    --tag "sft_baseline"

echo ">>> SFT 评测"
python scripts/eval.py \
    --model outputs/sft_baseline/final \
    --dataset gsm8k \
    --tag "sft_baseline"

echo ""
echo ">>> SFT 评测 MATH-500"
python scripts/eval.py \
    --model outputs/sft_baseline/final \
    --dataset math500 \
    --tag "sft_baseline_math500"

echo ""
echo "=== 实验完成 ==="
echo "结束时间: $(date)"
