# MSF-GRPO：面向数学推理的多源反馈融合强化学习微调方法

## Research Plan

---

## 一、研究背景与动机

### 1.1 问题场景：数学应用题自动求解

数学应用题求解是智能教育系统的核心能力。与简单答案判断不同，高质量的自动求解需要同时满足三个维度：

- **答案正确性**：最终结果是否正确（规则可验证）
- **推理过程质量**：解题步骤是否合理、完整（需 AI 评判）
- **表达规范性**：解题过程是否结构化、可读（需标量评分）

现有 LLM 微调方法（RLHF、DPO、GRPO）仅使用单一反馈信号，无法同时优化这三个维度。本文提出 MSF-GRPO（Multi-Source Feedback GRPO），通过融合规则反馈、AI 反馈和标量反馈，增强大语言模型在数学应用题求解任务上的推理能力。

### 1.2 LLM 微调演进路线

```
SFT (Supervised Fine-Tuning)
  ↓ 依赖人工标注的 demonstration 数据
RLHF (Reinforcement Learning from Human Feedback) [InstructGPT]
  ↓ 需要训练独立 Reward Model，流程复杂
DPO (Direct Preference Optimization) [Rafailov et al., 2023]
  ↓ 绕过 RM，直接从偏好对优化，但仅支持 pairwise
GRPO (Group Relative Policy Optimization) [DeepSeek-R1]
  ↓ 无需 critic model，组内相对优势估计
```

### 1.2 当前反馈信号类型

| 反馈形式 | 代表工作 | 信号来源 | 优势 | 局限 |
|---------|---------|---------|------|------|
| Pairwise Preference | InstructGPT, DPO | 人类标注 A>B | 直觉、易收集 | 信噪比低、标注成本高 |
| AI Feedback (RLAIF) | Constitutional AI | LLM 判别 | 可扩展、低成本 | 存在偏见传播风险 |
| Rule-based Reward (RLVR) | DeepSeek-R1, Scaf-GRPO | 规则/验证器 | 精确、无标注成本 | 仅适用于可验证任务 |
| Multi-dimensional Preference | SPO | 多维人类偏好 | 捕捉多维度需求 | 顺序优化存在遗忘问题 |
| Multi-objective Reward | MO-GRPO | 多个 RM | 平衡多目标 | 方差不均衡导致 reward hacking |

### 1.3 核心问题

**现有方法均针对单一或有限形式的反馈信号设计，缺乏一个统一框架来融合多种异构反馈信号进行 RL 微调。**

具体痛点：
1. **信号孤立**：RLHF、RLAIF、RLVR 各自独立发展，没有系统性地组合利用
2. **融合困难**：不同反馈信号的尺度、噪声水平、可靠性差异大，简单拼接会导致高方差信号主导训练
3. **动态缺失**：不同训练阶段对反馈信号的需求不同（早期需要粗粒度引导，后期需要精细偏好），但现有方法使用静态权重
4. **冲突处理**：多维反馈之间可能冲突（如 helpfulness vs harmlessness），现有顺序优化方法存在灾难性遗忘

---

## 二、研究定位与贡献

### 论文标题

**MSF-GRPO：面向数学推理的多源反馈融合强化学习微调方法**

### 核心贡献

1. **MSF-GRPO 方法**：提出将规则反馈（答案正确性）、AI 反馈（推理过程质量）、标量反馈（表达规范性）统一融合到 GRPO 框架的微调方法
2. **自适应反馈融合机制**：基于训练阶段和信号质量动态调整各反馈源权重，防止高方差信号主导训练
3. **冲突感知的多目标优化**：在 GRPO 中引入 Pareto 约束，避免多维反馈冲突导致的灾难性遗忘
4. **数学应用题求解实验**：在 GSM8K 等基准上验证多源反馈相比单一反馈的推理增强效果

---

## 三、方法设计

### 3.1 整体框架

```
┌─────────────────────────────────────────────────────┐
│                   UniFB Framework                     │
│                                                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐│
│  │ Pairwise  │ │  Scalar  │ │   AI     │ │  Rule    ││
│  │Preference │ │  Rating  │ │ Feedback │ │ Reward   ││
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘│
│       │            │            │            │       │
│       ▼            ▼            ▼            ▼       │
│  ┌─────────────────────────────────────────────────┐│
│  │         Feedback Encoder (统一编码层)            ││
│  │   将异构信号映射到共享 reward 空间 R ∈ ℝ^d      ││
│  └────────────────────┬────────────────────────────┘│
│                       │                              │
│                       ▼                              │
│  ┌─────────────────────────────────────────────────┐│
│  │      Adaptive Fusion Module (自适应融合)         ││
│  │   r_fused = Σ w_i(t) · r_i,  w_i 由信号质量决定 ││
│  └────────────────────┬────────────────────────────┘│
│                       │                              │
│                       ▼                              │
│  ┌─────────────────────────────────────────────────┐│
│  │    Conflict-Aware GRPO (冲突感知策略优化)        ││
│  │   多目标优势估计 + Pareto 约束                   ││
│  └─────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────┘
```

### 3.2 反馈统一编码器 (Feedback Encoder)

不同形式的反馈信号需要先统一到同一表示空间。

**Pairwise Preference → Reward**
- 经典 Bradley-Terry 模型：$r(x, y) = \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))$
- 可直接复用 DPO 的隐式 reward：$r(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$

**Scalar Rating → Reward**
- 归一化处理：$r_i^{norm} = \frac{r_i - \mu}{\sigma}$
- 可选：通过轻量 MLP 映射到与 pairwise reward 同尺度空间

**AI Feedback (RLAIF) → Reward**
- LLM-as-judge 输出偏好概率：$p_{AI}(y_w \succ y_l | x)$
- 转化为 reward 差：$r_{AI}(x, y) = \log \frac{p_{AI}}{1 - p_{AI}}$

**Rule-based Reward (RLVR) → Reward**
- 直接使用验证器输出：$r_{rule}(x, y) \in \{0, 1\}$ 或连续值
- 可靠性最高，但覆盖范围有限

**关键设计**：引入反馈可靠性先验 $\rho_i$，反映每种反馈源的固有可信度：
$$\rho_{rule} > \rho_{pairwise} > \rho_{AI} > \rho_{scalar}$$
（可根据任务调整）

### 3.3 自适应反馈融合 (Adaptive Fusion)

**核心思路**：不是固定权重，而是根据训练阶段和信号质量动态调整。

**方法一：方差感知加权（受 MO-GRPO 启发）**

MO-GRPO 发现 GRPO 在多目标下会偏向高方差 reward，导致 reward hacking。解决方案：

$$w_i(t) = \frac{\rho_i / \sigma_i^2(t)}{\sum_j \rho_j / \sigma_j^2(t)}$$

其中 $\sigma_i^2(t)$ 是第 $i$ 个反馈源在当前 batch 的 reward 方差。低方差信号获得更高权重。

**方法二：基于梯度冲突检测**

当不同反馈源的梯度方向冲突时（$\langle \nabla r_i, \nabla r_j \rangle < 0$），降低冲突较大信号的权重：

$$w_i(t) \propto \rho_i \cdot \text{align}_i(t)$$

其中 $\text{align}_i(t) = \cos(\nabla r_i, \nabla r_{fused})$ 衡量该信号与融合信号的一致性。

**方法三：课程式调度（Curriculum Schedule）**

不同训练阶段侧重不同反馈：

| 阶段 | 主要反馈 | 原因 |
|------|---------|------|
| 早期 (t < T₁) | Rule-based + Scalar | 粗粒度引导，快速建立基础能力 |
| 中期 (T₁ < t < T₂) | + AI Feedback | 扩展覆盖面，补充不可验证任务 |
| 后期 (t > T₂) | + Pairwise Preference | 精细对齐，捕捉人类细微偏好 |

### 3.4 冲突感知 GRPO (Conflict-Aware GRPO)

在 GRPO 框架下处理多维反馈冲突。

**标准 GRPO**（DeepSeek-R1）：
- 对每个 prompt $x$，采样一组 response $\{y_1, ..., y_G\}$
- 计算组内相对优势：$\hat{A}_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$
- 策略梯度更新

**多目标 GRPO 扩展**（受 MO-GRPO + SPO 启发）：

为每个反馈源分别计算优势：
$$\hat{A}_i^{(k)} = \frac{r_i^{(k)} - \text{mean}(\mathbf{r}^{(k)})}{\text{std}(\mathbf{r}^{(k)})}$$

融合优势：
$$\hat{A}_i^{fused} = \sum_k w_k(t) \cdot \hat{A}_i^{(k)}$$

**Pareto 约束**（防止灾难性遗忘）：
$$\hat{A}_i^{final} = \hat{A}_i^{fused} + \lambda \cdot \min_k \hat{A}_i^{(k)}$$

$\lambda$ 控制"短板保护"强度——确保即使融合后某个维度的 signal 被稀释，也不会完全被忽略。

### 3.5 损失函数

综合 GRPO 的策略梯度损失 + 辅助约束：

$$\mathcal{L} = \mathcal{L}_{GRPO}^{fused} + \alpha \cdot \mathcal{L}_{consistency} + \beta \cdot \mathcal{L}_{coverage}$$

- $\mathcal{L}_{GRPO}^{fused}$：使用融合 reward 的 GRPO 损失
- $\mathcal{L}_{consistency}$：不同反馈源之间的一致性约束，防止某信号被完全忽略
- $\mathcal{L}_{coverage}$：确保所有反馈源在训练中都被使用到（覆盖率正则）

---

## 四、实验设计

### 4.1 基础设置

**基础模型**：
- Qwen2.5-Math-1.5B（主实验，3090 可训练）
- 可选：Qwen2.5-Math-7B（扩展实验，需更大显存）

**训练框架**：TRL（GRPOTrainer）

**基线方法**：
| 基线 | 类型 | 说明 |
|------|------|------|
| Vanilla GRPO | 单一 rule reward | DeepSeek-R1 的核心算法 |
| DPO | 单一 pairwise | 标准偏好优化 |
| RLAIF | 单一 AI feedback | Constitutional AI 风格 |
| MO-GRPO | 多目标（简单归一化） | 方差感知但无融合 |
| UniFB (Ours) | 多源融合 | 本文方法 |

### 4.2 实验场景：数学推理增强

**主数据集**：GSM8K（7.5K 训练 / 1.3K 测试）

**扩展数据集**：MATH-500（验证泛化性，可选）

**多源反馈设计**（对应 ITS 的三个评估维度）：

| 反馈维度 | ITS 含义 | 信号来源 | 实现方式 |
|---------|---------|---------|---------|
| 答案正确性 | 学生最终答案是否正确 | 规则验证 | 提取 #### 后数字，与 ground truth 比对 |
| 推理过程质量 | 解题步骤是否合理完整 | AI 反馈 | 用 LLM 对 CoT 过程评分（1-5 分） |
| 表达清晰度 | 解题过程是否易读 | 标量评分 | 格式规范性 + 步骤完整性打分 |

**评测指标**：
- 主指标：pass@1 accuracy（答案正确率）
- 辅助指标：推理过程质量评分（GPT-4 评判）、格式规范性

### 4.3 消融实验

| 消融项 | 目的 |
|--------|------|
| 仅规则反馈（GRPO baseline） | 验证多源相比单源的增益 |
| 规则 + AI 反馈 | 验证 AI 评判推理过程的贡献 |
| 规则 + 标量反馈 | 验证表达清晰度信号的贡献 |
| 三源融合（完整方法） | 完整效果 |
| 固定权重 vs 自适应权重 | 验证动态融合的必要性 |
| 无冲突处理 | 验证 Pareto 约束的有效性 |

### 4.4 分析实验

1. **Reward Hacking 分析**：监控各维度 reward 的变化曲线，验证多源融合是否缓解单一信号过拟合
2. **训练稳定性**：对比不同方法的 reward 方差和 KL 散度变化
3. **反馈贡献度**：分析每个反馈源在训练各阶段的梯度范数占比
4. **案例分析**：展示多源反馈融合后模型生成的解题过程质量提升（对比单反馈）

---

## 五、时间规划

| 阶段 | 时间 | 任务 | 产出 |
|------|------|------|------|
| **Phase 0: 调研** | 第 1-2 周 | 精读论文、熟悉框架、确定技术方案 | 文献综述初稿 |
| **Phase 1: 基础实验** | 第 3-5 周 | 复现 GRPO baseline，搭建训练管线 | 可运行的 GRPO 训练代码 |
| **Phase 2: 方法实现** | 第 6-9 周 | 实现反馈编码器、融合模块、扩展 GRPO | UniFB 核心代码 |
| **Phase 3: 主实验** | 第 10-14 周 | 三场景全量实验 + 消融实验 | 实验结果表格 |
| **Phase 4: 论文撰写** | 第 15-17 周 | 写论文、画图、整理代码 | 论文初稿 |
| **Phase 5: 投稿准备** | 第 18-20 周 | 补充实验、润色论文、rebuttal 准备 | 终稿 |

---

## 六、技术栈

```
训练框架:  verl / TRL / OpenRLHF
基础模型:  Qwen2.5-7B (HuggingFace)
数据处理:  datasets, pandas
评测框架:  lm-evaluation-harness, AlpacaEval
实验追踪:  wandb
代码管理:  git + GitHub
```

---

## 七、风险与应对

| 风险 | 应对策略 |
|------|---------|
| 多反馈融合训练不稳定 | 先从 2 种反馈开始验证，逐步增加 |
| 计算资源不足 | 使用 7B 模型，借助 LoRA 降低显存 |
| 可验证任务覆盖有限 | 重点在数学/代码场景验证 rule reward 的增益 |
| 缺少人类标注数据 | 使用现有 HH-RLHF 数据集 + AI 反馈补充 |
| 方法创新性不足 | 突出"统一融合框架"+"自适应机制"的新颖性 |

---

## 八、论文结构（暂定）

```
1. Introduction
   - 智能数学辅导系统需要多维度评估学生解题
   - 现有 LLM 微调方法仅使用单一反馈信号的局限
   - 本文贡献：多源反馈融合框架

2. Related Work
   - RLHF / DPO / GRPO 演进
   - RLAIF (Constitutional AI)
   - RLVR (DeepSeek-R1)
   - Multi-objective RL for LLM (SPO, MO-GRPO)
   - 智能辅导系统中的 LLM 应用

3. Method
   - 3.1 问题定义：ITS 中的多维度评估需求
   - 3.2 多源反馈编码（规则 / AI / 标量 → 统一 reward）
   - 3.3 自适应反馈融合
   - 3.4 冲突感知 GRPO 训练

4. Experiments
   - 4.1 实验设置（GSM8K + Qwen2.5-Math-1.5B）
   - 4.2 主实验：多源 vs 单源反馈
   - 4.3 消融实验
   - 4.4 分析（reward 曲线、案例展示）

5. Conclusion
```

---

## 九、关键参考论文映射

| 论文 | 对本研究的启发 |
|------|--------------|
| InstructGPT | RLHF 范式基础 |
| Constitutional AI | RLAIF 的可行性，AI 反馈作为独立信号源 |
| DPO | 隐式 reward 建模，简化 RL 流程 |
| SPO | 多维偏好顺序优化 + 遗忘问题 → 启发并行融合 |
| MO-GRPO | 方差感知归一化 + reward hacking 问题 → 启发自适应权重 |
| Multi-Objective RL Survey | MORL 分类 + meta-policy 方向 |
| DeepSeek-R1 | GRPO 算法 + 多阶段训练 + rule-based reward |
| Scaf-GRPO | 解决 learning cliff，梯度 hint 机制 |
| RLVR-World | RLVR 向非推理任务的扩展 |
| RL Meets LLMs Survey | 全景视角，识别研究空白 |

---

## 十、参考文献（References）

### 核心方法论文

[1] Long Ouyang, Jeff Wu, Xu Jiang, et al. "Training Language Models to Follow Instructions with Human Feedback." *NeurIPS*, 2022. arXiv:2203.02155.

[2] Yuntao Bai, Saurav Kadavath, Sandipan Kundu, et al. "Constitutional AI: Harmlessness from AI Feedback." arXiv:2212.08073, 2022.

[3] Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn. "Direct Preference Optimization: Your Language Model Is Secretly a Reward Model." *NeurIPS*, 2023. arXiv:2305.18290.

[4] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov. "Proximal Policy Optimization Algorithms." arXiv:1707.06347, 2017.

[5] Zhihong Shao, Peiyi Wang, Qingxiu Zhu, et al. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." arXiv:2402.03300, 2024. (GRPO 算法首次提出)

[6] DeepSeek-AI. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." arXiv:2501.12948, 2025.

### 多目标 / 多维对齐论文

[7] Wenxuan Zhou, Ravi B. Sojitra, et al. "SPO: Sequential Preference Optimization for Multi-Dimensional Alignment." arXiv:2411.18084, 2024.

[8] MO-GRPO: Mitigating Reward Hacking via Multi-Objective Optimization in Group Relative Policy Optimization. arXiv, 2025. (待确认精确 arXiv ID)

[9] Scaf-GRPO: Tackling the Learning Cliff in Group Relative Policy Optimization. arXiv:2504.15158, 2025.

[10] Multi-Objective Reinforcement Learning for LLM Optimization: A Survey. arXiv, 2024. (待确认精确 arXiv ID)

### RLVR 扩展论文

[11] RLVR-World: Training World Models with Reinforcement Learning and Verifiable Rewards. arXiv, 2025. (待确认精确 arXiv ID)

[12] Reinforcement Learning for Large Language Models: A Survey. arXiv:2404.12289, 2024.

### 评测基准与基座模型

[13] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, et al. "Training Verifiers to Solve Math Word Problems." arXiv:2110.14168, 2021. (GSM8K 数据集)

[14] OpenAI. "GPT-4 Technical Report." arXiv:2303.08774, 2023.

[15] Qwen Team. "Qwen2.5-Math Technical Report: Toward Mathematical Expert Model." arXiv:2409.12122, 2024.

[16] Qwen Team. "Qwen2.5 Technical Report." arXiv:2412.15115, 2024.

### 其他相关工作

[17] Hugo Touvron, Louis Martin, Kevin Stone, et al. "Llama 2: Open Foundation and Fine-Tuned Chat Models." arXiv:2307.09288, 2023.

[18] Leandro von Werra, Younes Belkada, Lewis Tunstall, et al. "TRL: Transformer Reinforcement Learning." GitHub: huggingface/trl, 2020–2025.

---

### 论文对比表中可引用的公开 GSM8K 数据

| 模型 | GSM8K 准确率 | 来源 |
|------|-------------|------|
| GPT-4 (few-shot) | ~92.0% | [14] OpenAI, 2023 |
| DeepSeek-R1 | ~97.3% | [6] DeepSeek-AI, 2025 |
| Qwen2.5-Math-7B (CoT) | ~84% | [15] Qwen Team, 2024 |
| Qwen2.5-Math-1.5B (few-shot) | ~65% | [15] Qwen Team, 2024 |
| Qwen2.5-Math-1.5B (base, 本实验) | 44.28% | 本研究 baseline 评测 |

> **引用前提**：评测设置与原文一致（GSM8K 官方 test 1319 题 + `####` 答案提取 + pass@1 accuracy），可直接对比。
