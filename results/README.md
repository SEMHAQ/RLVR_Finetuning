# 实验结果

## 当前结果汇总

| 实验 | 方法 | 准确率 | 变化 | 说明 |
|------|------|--------|------|------|
| baseline | Qwen2.5-Math-1.5B (zero-shot) | 44.28% | — | 基线，无训练 |
| grpo_v1 | 全量微调 (lr=5e-7) | 44.20% | -0.08% | 学习率太小，训练无效 |
| grpo_v2_lora | LoRA (1000样本, lr=1e-5) | 44.66% | +0.38% | 数据量不足，提升有限 |
| grpo_v3_lora | LoRA (全量, lr=2e-5) | 79.38% | +35.1% | 全量数据 + 合适 lr，效果显著 |

## 关键发现

1. **学习率至关重要**：lr=5e-7 完全无效，lr=2e-5 效果显著
2. **数据量很重要**：1000 样本只提升 0.38%，全量 7473 样本提升 35.1%
3. **LoRA 效果不逊于全量微调**：显存从 ~20GB 降到 ~8GB，效果更好
4. **GRPO 对数学推理有效**：从 44% → 79%，提升近一倍

## 训练配置

| 参数 | grpo_v3_lora |
|------|-------------|
| 模型 | Qwen/Qwen2.5-Math-1.5B |
| 方法 | LoRA (rank=16) + GRPO |
| 数据 | GSM8K 全量 (7473 样本) |
| 学习率 | 2e-5 |
| batch_size | 24 (effective=72) |
| num_generations | 3 |
| max_completion_length | 256 |
| epoch | 1 |
| 总步数 | 934 |
| 显存占用 | ~8GB |

## 文件说明

| 文件名 | 内容 |
|--------|------|
| `eval_baseline.json` | 基线评估（44.28%） |
| `eval_grpo_v1.json` | GRPO v1 评估（44.20%，未生效） |
| `eval_grpo_v2_lora.json` | GRPO v2 LoRA 评估（44.66%） |
| `eval_grpo_v3_lora.json` | GRPO v3 LoRA 评估（79.38%） |
| `train_summary.json` | 最近一次训练参数 |

## 使用方式

```bash
# 评估基线
python scripts/eval.py --model Qwen/Qwen2.5-Math-1.5B --tag baseline

# 训练 GRPO (LoRA)
python scripts/train_grpo.py --model Qwen/Qwen2.5-Math-1.5B --output_dir outputs/grpo_v3_lora --use_lora --lr 2e-5 --batch_size 24 --num_epochs 1

# 评估训练后模型
python scripts/eval.py --model outputs/grpo_v3_lora/final --tag grpo_v3_lora
```

跑完后 `git add results/ && git commit -m "results: ..." && git push`。
