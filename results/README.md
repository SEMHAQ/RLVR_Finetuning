# 实验结果

每次运行 eval 或 train 脚本，结果会自动保存到这个目录。

## 文件说明

| 文件名模式 | 来源 | 内容 |
|-----------|------|------|
| `eval_<tag>.json` | `scripts/eval.py` | 评估准确率 + 10 个样例 |
| `train_summary.json` | `scripts/train_grpo.py` | 训练超参 + loss + 耗时 |

## 使用方式

```bash
# 评估基线模型 → results/eval_1.5B.json
python scripts/eval.py --model Qwen/Qwen2.5-Math-1.5B --tag baseline

# 训练 GRPO
python scripts/train_grpo.py --model Qwen/Qwen2.5-Math-1.5B --output_dir outputs/grpo_baseline

# 评估训练后模型 → results/eval_grpo_v1.json
python scripts/eval.py --model outputs/grpo_baseline/final --tag grpo_v1
```

跑完后 `git add results/ && git commit -m "results: ..." && git push`，我就能看到了。
