# UniFB: 多源反馈融合的 LLM 强化学习微调

基于多形式反馈与强化学习的大语言模型微调方法研究

## 项目结构

```
RLVR_Finetuning/
├── configs/           # 训练配置
│   └── grpo_baseline.yaml
├── src/               # 核心代码
│   ├── data.py        # 数据加载与预处理
│   ├── rewards.py     # 奖励函数（规则/AI/融合）
│   └── eval.py        # 评估工具
├── scripts/           # 训练与评估脚本
│   ├── train_grpo.py  # GRPO 训练入口
│   └── eval.py        # 评估入口
├── outputs/           # 模型输出（gitignore）
├── requirements.txt
└── RESEARCH_PLAN.md   # 详细研究计划
```

## 环境搭建

```bash
# 1. 创建虚拟环境
conda create -n unifb python=3.11 -y
conda activate unifb

# 2. 安装 PyTorch (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. 安装 flash-attn (可选，提升速度，需要编译)
pip install flash-attn --no-build-isolation

# 4. 安装项目依赖
pip install -r requirements.txt
```

## 快速开始

### Phase 0: GRPO Baseline (当前阶段)

**Step 1: 先评估基线模型**（看看 Qwen2.5-Math-1.5B 在 GSM8K 上的原始表现）

```bash
python scripts/eval.py --model Qwen/Qwen2.5-Math-1.5B --batch_size 4
```

**Step 2: 训练 GRPO**

```bash
# 标准训练（需要 ~16GB 显存）
python scripts/train_grpo.py \
    --model Qwen/Qwen2.5-Math-1.5B \
    --output_dir outputs/grpo_baseline \
    --num_epochs 1 \
    --batch_size 2 \
    --grad_accum 8 \
    --num_generations 5 \
    --lr 5e-7 \
    --reward combined_rule

# 如果显存不够，用 LoRA
python scripts/train_grpo.py \
    --model Qwen/Qwen2.5-Math-1.5B \
    --output_dir outputs/grpo_baseline_lora \
    --use_lora \
    --num_epochs 1
```

**Step 3: 评估训练后的模型**

```bash
python scripts/eval.py --model outputs/grpo_baseline/final --batch_size 4
```

## 训练超参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_generations` | 5 | 每个 prompt 采样 5 个 response，计算组内相对优势 |
| `beta` | 0.0 | KL 惩罚系数，设为 0 表示不限制与参考模型的距离 |
| `lr` | 5e-7 | 学习率，GRPO 需要较小的学习率 |
| `batch_size` | 2 | 每设备 batch size，配合 grad_accum=8 得到有效 batch=80 |

## 奖励函数

当前支持：
- `rule` — 纯答案正确性检查
- `format` — 格式奖励（是否包含 ####）
- `combined_rule` — 上述两者之和

计划中：
- `ai_feedback` — LLM-as-judge 评分
- `scalar` — 外部打分器
- `fused` — 多源自适应融合

## 常见问题

**Q: OOM (显存不足) 怎么办？**
- 减小 `--batch_size` 到 1
- 加 `--use_lora`
- 减小 `--num_generations` 到 3
- 减小 `--max_completion_length` 到 256

**Q: 没有 flash-attn 怎么办？**
代码会自动回退到默认 attention，只是训练速度会慢一些。

**Q: 训练多久能看到效果？**
GSM8K 训练集 7.5K 条，GRPO 1 个 epoch 大约 5-8 小时（3090，batch_size=10）。建议先跑 200 步看看 reward 曲线趋势。
