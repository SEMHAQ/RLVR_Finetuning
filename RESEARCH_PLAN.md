# 基于多形式反馈与强化学习的大模型微调方法研究

## Research Plan

---

## 一、研究背景与动机

### 1.1 LLM 微调演进路线

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

### 论文标题（暂定）

**"UniFB: Unified Multi-Form Feedback Fusion for Reinforcement Learning Fine-Tuning of Large Language Models"**

或中文：**基于统一多形式反馈融合的大语言模型强化学习微调方法**

### 核心贡献

1. **反馈统一建模框架**：提出将 pairwise preference、scalar rating、AI judgment、rule-based verification 四种反馈形式统一映射到共享 reward 空间的方法
2. **自适应反馈融合机制**：基于训练阶段和信号质量动态调整各反馈源的权重，防止高方差信号主导
3. **冲突感知的多目标优化**：在 GRPO 框架下实现多维反馈的 Pareto 优化，避免顺序优化的遗忘问题
4. **跨任务实验验证**：在数学推理、代码生成、开放对话三个场景下验证方法有效性

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
- Qwen2.5-7B / Qwen2.5-Math-7B（与 Scaf-GRPO 对齐，便于对比）
- 可选：Llama-3.1-8B（验证泛化性）

**训练框架**：verl（与 Scaf-GRPO 一致）或 TRL / OpenRLHF

**基线方法**：
| 基线 | 类型 | 说明 |
|------|------|------|
| Vanilla GRPO | 单一 rule reward | DeepSeek-R1 的核心算法 |
| DPO | 单一 pairwise | 标准偏好优化 |
| RLAIF | 单一 AI feedback | Constitutional AI 风格 |
| MO-GRPO | 多目标（简单归一化） | 方差感知但无融合 |
| SPO | 多维顺序优化 | 顺序对齐，有遗忘 |
| UniFB (Ours) | 多形式融合 | 本文方法 |

### 4.2 实验场景

**场景 1：数学推理（可验证任务）**

- 数据集：MATH-500, AIME, AMC, GSM8K
- 反馈形式组合：
  - Rule-based：答案正确性验证
  - Scalar：过程评分（CoT 质量打分）
  - AI Feedback：LLM 对推理过程的评判
- 评测：pass@1 accuracy
- 预期：三反馈融合 > 单独 rule reward（GRPO baseline）

**场景 2：代码生成（半可验证任务）**

- 数据集：HumanEval, MBPP, LiveCodeBench
- 反馈形式组合：
  - Rule-based：测试用例通过率
  - Pairwise：人类对代码风格/可读性的偏好
  - AI Feedback：LLM 对代码质量的评分
- 评测：pass@1, test case 覆盖率, 代码质量分
- 预期：补充 pairwise/AI feedback 可提升代码可读性，不牺牲正确性

**场景 3：开放对话（不可验证任务）**

- 数据集：AlpacaEval, MT-Bench, HH-RLHF
- 反馈形式组合：
  - Pairwise：人类偏好标注
  - Scalar：Likert 评分
  - AI Feedback：GPT-4 评分
- 评测：win rate, GPT-4 score, 多维度评分
- 预期：融合多信号比单一 DPO 更稳定、更全面

### 4.3 消融实验

| 消融项 | 目的 |
|--------|------|
| 去掉 Rule-based reward | 验证可验证信号的贡献 |
| 去掉 AI Feedback | 验证 AI 反馈的补充作用 |
| 去掉 Pairwise | 验证人类偏好的精细对齐作用 |
| 固定权重 vs 自适应权重 | 验证动态融合的必要性 |
| 无 Pareto 约束 | 验证冲突处理的有效性 |
| 顺序优化 vs 并行融合 | 对比 SPO 的顺序方式与本文并行方式 |

### 4.4 分析实验

1. **Reward Hacking 分析**：监控各维度 reward 的变化曲线，验证 MO-GRPO 发现的问题是否被解决
2. **训练稳定性**：对比不同方法的 reward 方差和 KL 散度变化
3. **反馈覆盖率**：分析每个反馈源在训练各阶段的贡献度（梯度范数占比）
4. **泛化性**：在 OOD 任务上评测（如数学训练 → 代码评测）

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
   - LLM alignment 的重要性
   - 现有方法的局限（单一反馈）
   - 本文贡献

2. Related Work
   - RLHF / DPO / GRPO 演进
   - RLAIF (Constitutional AI)
   - RLVR (DeepSeek-R1)
   - Multi-objective RL for LLM (SPO, MO-GRPO)

3. Method: UniFB
   - 3.1 Feedback Encoding（统一编码）
   - 3.2 Adaptive Fusion（自适应融合）
   - 3.3 Conflict-Aware GRPO（冲突感知优化）
   - 3.4 Training Objective（综合损失）

4. Experiments
   - 4.1 数学推理
   - 4.2 代码生成
   - 4.3 开放对话
   - 4.4 消融实验
   - 4.5 分析

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
