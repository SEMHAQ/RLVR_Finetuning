#!/bin/bash
# num_generations 消融实验
# Usage: bash scripts/run_ablation_ngen.sh

set -e

echo "=== num_generations 消融实验 ==="
echo "开始时间: $(date)"

for ngen in 2 4 8; do
    echo ""
    echo ">>> 训练 num_generations=$ngen"
    # batch_size 必须是 num_generations 的倍数
    bs=$((ngen * 2))
    python scripts/train_grpo.py \
        --model Qwen/Qwen2.5-Math-1.5B \
        --dataset gsm8k \
        --reward rule \
        --lr 2e-5 \
        --batch_size $bs \
        --num_generations $ngen \
        --max_completion_length 256 \
        --use_lora \
        --tag "ablation_ngen_${ngen}"

    echo ">>> 评测 num_generations=$ngen"
    python scripts/eval.py \
        --model outputs/grpo_ablation_ngen_${ngen}/final \
        --dataset gsm8k \
        --tag "ablation_ngen_${ngen}"
done

echo ""
echo "=== 实验完成 ==="
echo "结束时间: $(date)"
