# 实验结果

## 当前结果汇总

| 实验 | 模型 | 准确率 | 说明 |
|------|------|--------|------|
| baseline | Qwen2.5-Math-1.5B (zero-shot) | 44.28% | 基线，无训练 |
| grpo_v1 | Qwen2.5-Math-1.5B + GRPO | 44.20% | lr=5e-7 太小，训练无效 |

## 文件说明

| 文件名 | 内容 |
|--------|------|
| `eval_baseline.json` | 基线评估结果（44.28%） |
| `eval_grpo_v1.json` | GRPO v1 评估结果（44.20%，未生效） |
| `train_summary.json` | GRPO v1 训练参数 |

## 使用方式

```bash
# 评估基线
python scripts/eval.py --model Qwen/Qwen2.5-Math-1.5B --tag baseline

# 训练 GRPO
python scripts/train_grpo.py --model Qwen/Qwen2.5-Math-1.5B --output_dir outputs/grpo_baseline

# 评估训练后模型
python scripts/eval.py --model outputs/grpo_baseline/final --tag grpo_v1
```

跑完后 `git add results/ && git commit -m "results: ..." && git push`。
