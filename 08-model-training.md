# 08 - 模型训练与微调

> **难度：** ⭐⭐⭐⭐⭐ | **定位：** LLM 核心能力定制层，中高级 AI 工程师必考
>
> **前端类比：** 微调之于 LLM，就像 Theme Customization 之于 UI 框架——基座模型是 Ant Design，微调是基于它定制企业主题。Full Fine-tuning 相当于 Fork 整个框架重写，LoRA 相当于只写一个 Theme Override 文件覆盖少量变量。RLHF 则像是 A/B Testing + 用户反馈驱动的持续优化。

## 本章知识树

```
模型训练与微调
├── 8.0 训练基础流程（前向传播 / 损失函数 / 反向传播 / 梯度下降）
├── 8.1 微调基础（Full Fine-tuning vs PEFT, 什么时候需要微调）
├── 8.2 LoRA / QLoRA（原理、rank 选择、代码示例）
├── 8.3 对齐技术（RLHF 三阶段、DPO 直接偏好优化、GRPO）
├── 8.4 训练数据工程（数据清洗、标注、格式）
├── 8.5 训练优化（DeepSpeed ZeRO、FSDP、混合精度）
├── 8.6 TRL v1.0 统一训练框架
├── 8.7 微调 vs RAG vs Prompt Engineering 选型
└── 8.8 评估与迭代
```

---

## 8.0 训练的基础流程

### Q: 训练过程是怎么运作的？Q/K/V 在训练中发挥什么作用？

训练的目标是**调整权重**（W_Q、W_K、W_V 等矩阵），让模型预测得更准。推理是**用权重**生成文本。

```
训练样本："苹果是红色的"
  输入：["苹果", "是", "红色", "的"]
  标签：["是", "红色", "的", <结束>]  ← 每个位置预测下一个词

1. 前向传播（和推理完全一样）
   每个 token → embedding → 每层算 Q/K/V → Attention + FFN
   → 最后一层输出 × 词表矩阵 → 每个位置得到概率分布

2. 计算损失（Loss）
   模型预测"苹果"后面是"是"的概率是 0.6，正确答案就是"是"
   Cross Entropy 损失函数衡量预测概率和真实答案的差距
   → 预测越不准，Loss 越高

3. 反向传播
   从 Loss 出发，往回计算每个权重"对 Loss 的贡献有多大"
   → 包括 W_Q、W_K、W_V、W_O、FFN 的所有权重
   → 这个"贡献度"叫梯度
   → 需要保留前向传播的所有中间结果，才能算出梯度

4. 更新权重（梯度下降）
   W_Q = W_Q - 学习率 × 梯度
   → 把每个权重往"让 Loss 变小"的方向微调一点点
   → 重复亿万次，模型越来越准
```

**训练 vs 推理的本质区别：**

| | 推理 | 训练 |
|---|---|---|
| 目标 | 用现有权重生成文本 | 调整权重让预测更准 |
| Q/K/V 的作用 | 计算 token 关系，生成新表示向量 | 同上，但还要保留中间结果供反向传播用 |
| 方向 | 只有前向（输入 → 输出） | 前向 + 反向（输出 → 输入，算梯度） |
| 权重状态 | 固定不变 | 每步都在更新 |
| KV Cache | 可以用（历史 K/V 不变） | 不能用（权重每步都在变，K/V 也跟着变） |

> **应用开发需要记住的：** 训练是调权重、推理是用权重。"预训练 → 微调 → 对齐"是大流程，反向传播的细节不是应用开发的考点。

---

## 8.1 微调基础

### Q: 什么是微调（Fine-tuning）？Full Fine-tuning 和 PEFT 有什么区别？

**微调是在预训练模型的基础上，使用特定领域/任务数据继续训练，让模型学会新能力或适应新风格。**

```
预训练（Pre-training）：
  海量通用数据 → 学会"语言能力"
  类比：大学通识教育，什么都懂一点

微调（Fine-tuning）：
  特定任务数据 → 学会"专业技能"
  类比：研究生专业培训，精通某个领域

推理（Inference）：
  部署上线 → 服务用户
  类比：毕业上岗，开始干活
```

**Full Fine-tuning vs PEFT（Parameter-Efficient Fine-Tuning）：**

| 维度 | Full Fine-tuning | PEFT（如 LoRA） |
|------|-----------------|----------------|
| **更新参数量** | 100% 所有参数 | 0.1%~10% 参数 |
| **显存需求（7B）** | ~56GB（Adam） | ~8-16GB |
| **训练速度** | 慢，全量梯度计算 | 快 2-5x |
| **灾难性遗忘** | 高风险 | 低风险，基座冻结 |
| **多任务支持** | 每个任务一个完整模型 | 每个任务一个小 Adapter |
| **适用场景** | 大规模数据、需要深度改造 | 数据有限、快速迭代 |
| **前端类比** | Fork Ant Design 改源码 | 写 Theme Override 文件 |

**PEFT 家族全景：**

```
PEFT 方法
├── Adapter 系列
│   ├── LoRA（Low-Rank Adaptation）       ← 最主流 ⭐
│   ├── QLoRA（Quantized LoRA）           ← 消费级 GPU 可用 ⭐
│   ├── DoRA（Weight-Decomposed LoRA）    ← 2024 新秀
│   └── AdaLoRA（Adaptive LoRA）          ← 动态 rank
├── Prompt 系列
│   ├── Prefix Tuning                     ← 在 attention 前加可学习前缀
│   ├── P-Tuning v2                       ← 深层 prompt
│   └── Prompt Tuning                     ← 只加 soft prompt
└── 其他
    ├── IA3                               ← 极少参数
    └── BitFit                            ← 只调 bias
```

**什么时候需要微调？判断清单：**

1. **Prompt Engineering 已无法满足需求** — 即使精心设计 prompt 也达不到效果
2. **需要稳定的输出格式** — JSON Schema、特定代码风格
3. **有足够领域数据** — 至少几百到几千条高质量标注数据
4. **需要降低推理成本** — 小模型微调后可以替代大模型
5. **需要私有知识内化** — RAG 延迟/检索质量无法满足

> **面试话术：** "微调的核心价值是让模型从'通才'变成'专才'。我通常用 PEFT 方法如 LoRA，它只更新不到 1% 的参数，显存占用降 4 倍以上，同时通过冻结基座权重来避免灾难性遗忘。在实际项目中，我会先尝试 Prompt Engineering，效果不够再考虑 RAG，最后才上微调——因为微调的数据成本和维护成本是最高的。"

---

## 8.2 LoRA / QLoRA

### Q: 请解释 LoRA 的原理？为什么低秩分解可以有效微调？

**LoRA（Low-Rank Adaptation）的核心思想：冻结原始权重，通过低秩矩阵分解注入可训练参数。**

> 常见误解：LoRA 是"只微调关键部分的参数"——实际上 LoRA 不是选择性更新原始权重，而是在原始权重旁边新增两个小矩阵（A 和 B），训练时只更新这两个小矩阵，原始权重完全冻结。推理时可以把 ΔW=B×A 合并回原矩阵，没有额外计算开销。

**数学原理：**

```
原始权重更新：
  W' = W + ΔW

  W  ∈ R^(d×d)     原始权重矩阵（冻结）
  ΔW ∈ R^(d×d)     全量更新需要 d² 个参数

LoRA 的关键洞察：
  ΔW 在微调时是"低秩"的，可以分解为两个小矩阵的乘积

  ΔW = B × A

  A ∈ R^(r×d)       降维矩阵（Down Projection）
  B ∈ R^(d×r)       升维矩阵（Up Projection）
  r << d             r 就是 "rank"，通常 4~64

参数量对比：
  Full:  d × d = d²           例: 4096² = 16,777,216
  LoRA:  d × r + r × d = 2dr  例: 2 × 4096 × 16 = 131,072
  压缩比: d²/(2dr) = d/(2r) = 4096/32 = 128x  🔥
```

**LoRA 前向计算过程：**

```
                    ┌──────────────┐
  Input x ────────►│  W (冻结)     │──── W·x ──────┐
    │               └──────────────┘                │
    │                                               ▼
    │               ┌──────┐  ┌──────┐           ┌─────┐
    └──────────────►│  A   │─►│  B   │──► B·A·x ►│  +  │──► Output
                    │(d→r) │  │(r→d) │    × α/r  └─────┘
                    └──────┘  └──────┘
                    可训练参数（很少）

  Output = W·x + (α/r) · B·A·x

  α: scaling factor，控制 LoRA 的影响强度
  初始化: A ~ N(0, σ²), B = 0 → 训练开始时 ΔW = 0
```

**为什么低秩分解有效？**

研究发现（Aghajanyan et al., 2020）：预训练模型在微调时，权重变化矩阵 ΔW 的内在维度（intrinsic dimension）远小于参数空间维度。直觉上，微调只需要在一个低维子空间里调整模型行为，不需要改变所有参数。

**前端类比：** CSS 变量覆盖。主题框架有几千个 CSS 属性（高维空间），但你自定义品牌主题时通常只改十几个核心变量（primary-color, font-family 等），其他属性都由这些变量派生。LoRA 找到的就是这些"核心变量"。

**Rank 选择指南：**

| Rank (r) | 可训练参数占比 | 适用场景 | 效果 |
|----------|-------------|---------|------|
| 4-8 | ~0.1% | 简单任务适配、风格迁移 | 够用 |
| 16-32 | ~0.5% | 大多数微调任务 ⭐ | 性价比最高 |
| 64-128 | ~2% | 复杂领域、大量数据 | 接近全量微调 |
| 256+ | ~5%+ | 通常不必要 | 过拟合风险 |

**实践经验：** rank 不是越大越好。很多实验表明 r=16 和 r=64 的效果差距很小，但显存占用差 4 倍。建议从 r=16 开始，如果效果不够再增大。

### Q: QLoRA 是什么？如何在消费级 GPU 上微调大模型？

**QLoRA = Quantized LoRA，在 4-bit 量化的基座模型上做 LoRA 微调，让单张 24GB 显卡也能微调 70B 模型。**

**QLoRA 的三大核心技术：**

```
QLoRA = NF4 量化 + 双重量化 + 分页优化器

1. NF4（NormalFloat4）量化：
   ┌─────────────────────────────────────────┐
   │  FP16 权重 → 4-bit NF4 量化权重（冻结）  │
   │  信息论最优：假设权重服从正态分布         │
   │  比普通 INT4 量化损失更小                 │
   └─────────────────────────────────────────┘

2. 双重量化（Double Quantization）：
   ┌─────────────────────────────────────────┐
   │  量化的缩放因子（scale）也被量化          │
   │  FP32 scale → FP8 scale                  │
   │  额外节省 ~0.37 bit/param 显存           │
   └─────────────────────────────────────────┘

3. Paged Optimizers：
   ┌─────────────────────────────────────────┐
   │  优化器状态 GPU→CPU 分页                  │
   │  类似虚拟内存，GPU 显存不够时自动换出     │
   │  避免 OOM（Out of Memory）              │
   └─────────────────────────────────────────┘
```

**显存对比（LLaMA-65B 微调）：**

| 方法 | 显存占用 | 最少 GPU |
|------|---------|---------|
| Full Fine-tuning (FP16) | ~780GB | 10× A100 80GB |
| LoRA (FP16 基座) | ~130GB | 2× A100 80GB |
| QLoRA (NF4 基座) | ~33GB | 1× A100 40GB |
| QLoRA (NF4 + 4bit) | ~18GB | 1× RTX 4090 24GB（适用于33B模型）⭐ |

> **注：** 65B模型需要单张48GB GPU（如A40/A100 48GB）

**QLoRA 代码示例（使用 bitsandbytes + peft）：**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# 1. 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NF4 量化
    bnb_4bit_compute_dtype=torch.bfloat16, # 计算用 bf16
    bnb_4bit_use_double_quant=True,       # 双重量化
)

# 2. 加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    quantization_config=bnb_config,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)

# 3. LoRA 配置
lora_config = LoraConfig(
    r=16,                    # rank
    lora_alpha=32,           # scaling = alpha/r = 2
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 13,631,488 || all params: 8,043,212,800 || 0.17%

# 4. 训练
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir="./output",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
    ),
)
trainer.train()

# 5. 保存（只保存 LoRA 权重，几十 MB）
model.save_pretrained("./lora-adapter")

# 6. 推理时合并
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
merged_model = PeftModel.from_pretrained(base_model, "./lora-adapter")
merged_model = merged_model.merge_and_unload()  # 合并回基座
```

> **面试话术：** "LoRA 的核心原理是假设微调时的权重变化矩阵 ΔW 是低秩的，通过分解为 B×A 两个小矩阵来大幅减少可训练参数。QLoRA 在此基础上引入 NF4 量化，把基座模型压缩到 4-bit，配合双重量化和分页优化器，让单张 24GB 消费级 GPU 就能微调 70B 级别模型。在我的项目中，QLoRA + r=16 是性价比最高的配置。"

---

## 8.3 对齐技术

### Q: 请详细解释 RLHF 的三个阶段？

**RLHF（Reinforcement Learning from Human Feedback）是让模型行为与人类偏好对齐的训练方法，分三个阶段。**

**RLHF 是训练阶段的事，不是推理阶段。** 预训练让模型学会语言，RLHF 让模型学会"什么样的回答是好的"——注入人类的价值判断，解决模型输出有害、不诚实、无帮助的问题。

**对齐技术全家族（RLHF 不是唯一方案）：**

| 方法 | 核心思路 | 优势 | 劣势 |
|------|---------|------|------|
| **RLHF + PPO** | 训练奖励模型 + 强化学习 | 效果最好 | 需要 4 个模型，训练复杂 |
| **DPO** | 直接从偏好数据学习，跳过奖励模型 | 简单稳定，只需 2 个模型 | 效果略逊于 RLHF |
| **GRPO** | 组内相对奖励，省去Value Model（Critic）[^grpo] | 更省显存，DeepSeek-R1 使用 | 较新，生态不如 DPO 成熟 |

[^grpo]: 仍需Reference Policy计算KL约束

```
RLHF 三阶段流水线：

Stage 1: SFT（监督微调）
  ┌────────────────────────────────────────┐
  │  人工标注数据 (instruction, response)    │
  │         │                               │
  │         ▼                               │
  │  Base Model ──SFT──► SFT Model          │
  │  (预训练模型)        (能对话了)           │
  └────────────────────────────────────────┘
                    │
                    ▼
Stage 2: RM（奖励模型训练）
  ┌────────────────────────────────────────┐
  │  SFT Model 生成多个回答                  │
  │  人工标注偏好排序: A > B > C             │
  │         │                               │
  │         ▼                               │
  │  训练 Reward Model（打分器）             │
  │  输入: (prompt, response) → 分数 scalar  │
  └────────────────────────────────────────┘
                    │
                    ▼
Stage 3: PPO（强化学习优化）
  ┌────────────────────────────────────────┐
  │  Policy Model (SFT Model 副本)          │
  │         │                               │
  │    生成 response                        │
  │         │                               │
  │    Reward Model 打分                    │
  │         │                               │
  │    PPO 算法更新 Policy                   │
  │    + KL 散度约束（不要偏离 SFT 太远）     │
  │         │                               │
  │         ▼                               │
  │  对齐后的模型 ✅                         │
  └────────────────────────────────────────┘
```

**各阶段详解：**

**Stage 1 — SFT（Supervised Fine-Tuning）**

```python
# SFT 数据格式示例
{
    "instruction": "用一句话解释量子纠缠",
    "input": "",
    "output": "量子纠缠是两个粒子之间的一种关联..."
}
```

- 目标：让 base model 学会"对话"能力
- 数据量：通常 10K-100K 条高质量对话
- 训练：标准交叉熵损失，next-token prediction

**Stage 2 — Reward Model（奖励模型）**

```
偏好数据格式：

Prompt: "写一首关于春天的诗"

Response A (chosen):  "春风拂柳绿如烟..."   ← 人类更喜欢
Response B (rejected): "春天来了真好看..."   ← 质量较差

RM 训练目标（Bradley-Terry 模型）：
  L = -log(σ(r(x, y_w) - r(x, y_l)))

  r(x, y): Reward Model 对 (prompt, response) 的打分
  y_w: 人类偏好的回答 (winner)
  y_l: 被拒绝的回答 (loser)
  σ: sigmoid 函数
```

**Stage 3 — PPO（Proximal Policy Optimization）**

```
PPO 优化目标：

  max  E[r(x, y)]                    ← 最大化奖励分数
  s.t. KL(π_RL || π_SFT) < δ        ← 约束与 SFT 模型的 KL 散度

实际损失函数：
  L = -E[r(x, y)] + β · KL(π_RL || π_SFT)

  β: KL 惩罚系数，防止 reward hacking
```

**RLHF 的痛点：**

| 问题 | 描述 |
|------|------|
| **Reward Hacking** | 模型学会讨好 RM 而非真正有用 |
| **训练不稳定** | PPO 超参敏感，容易崩溃 |
| **成本高** | 需要同时维护 4 个模型（SFT, RM, Policy, Reference） |
| **人工标注贵** | 偏好数据需要高质量人工标注 |

### Q: DPO 如何简化 RLHF？公式推导是怎样的？

**DPO（Direct Preference Optimization）直接跳过 Reward Model 和 PPO，用一个简洁的损失函数直接从偏好数据学习。**

**DPO 完整流程：**

```
第 1 步：SFT（和 RLHF 完全一样）
  人工写高质量问答 → 微调模型学会"怎么回答"
  产出：SFT Model（同时作为 Policy 和 Reference 的初始权重）

第 2 步：收集偏好数据
  同一问题让模型生成多个回答
  人类标注：哪个好（y_w）、哪个差（y_l）
  产出：(问题, 好回答, 差回答) 三元组
  → 不需要精确打分，只需要两两比较

第 3 步：DPO 训练（跳过奖励模型和 PPO）
  直接用偏好数据优化 Policy Model
  Reference Model 全程冻结，只作对比基准
  目标：好回答概率↑，差回答概率↓，且不偏离 Reference 太远
```

**DPO 为什么不需要奖励模型也能对齐：**

RLHF 用奖励模型把"人类偏好"转成数值分数，再用 RL 优化。DPO 直接把"偏好比较"作为训练信号，绕过了打分这一步。DPO 的洞察是：可以直接用偏好数据（好回答/差回答对）训练，本质是一个分类问题：

```
训练数据：
  (问题, 好回答, 差回答)
  例：
  问题：  "如何学习编程"
  好回答："从基础语法开始，多做练习..."
  差回答："随便找个教程看看就行"

DPO 目标：
  让模型生成"好回答"的概率 ↑
  让模型生成"差回答"的概率 ↓
  + KL 约束：不要偏离原始模型太远
```

省掉了"训练打分器"这一步，效果接近 RLHF，训练稳定很多。

```
RLHF（复杂）：                    DPO（简化）：
  SFT → RM → PPO                    SFT → DPO
  需要 4 个模型同时运行               只需要 2 个模型
  训练不稳定                         稳定如 SFT
```

**DPO 的核心公式：**

```
DPO Loss:

  L_DPO = -E[log σ(β · (log π_θ(y_w|x)/π_ref(y_w|x)
                       - log π_θ(y_l|x)/π_ref(y_l|x)))]

简化理解：
  L = -log σ(β · (Δ_win - Δ_lose))

  Δ_win  = log π_θ(y_w|x) - log π_ref(y_w|x)  ← 模型相对参考的提升
  Δ_lose = log π_θ(y_l|x) - log π_ref(y_l|x)  ← 同上

  目标：让 Δ_win > Δ_lose
  即：让模型更倾向生成好回答，远离差回答

关键洞察：
  DPO 证明了最优 reward function 可以用 policy 本身表示：
  r*(x, y) = β · log(π*(y|x) / π_ref(y|x)) + C

  把这个代入 RLHF 目标函数，RM 和 PPO 就被消除了！
```

**DPO vs RLHF 对比：**

| 维度 | RLHF (PPO) | DPO |
|------|-----------|-----|
| **模型数量** | 4个（SFT + RM + Policy + Ref） | 2个（Policy + Ref） |
| **训练稳定性** | 差，超参敏感 | 好，类似 SFT |
| **实现复杂度** | 高（需要 RL 基础设施） | 低（标准分类损失） |
| **GPU 显存** | 非常高 | 中等 |
| **效果上限** | 理论上更高 | 实践中接近 |
| **迭代效率** | 慢（PPO 采样慢） | 快 |

### Q: 什么是 GRPO？DeepSeek 为什么选择它？

**GRPO（Group Relative Policy Optimization）是 DeepSeek 提出的强化学习算法，核心创新：用组内相对排名替代 Reward Model。**

```
PPO 流程：
  生成 response → Reward Model 打分 → 计算优势函数 → 更新策略

GRPO 流程：
  对同一 prompt 生成 G 个 response
  → 组内排名计算相对优势（无需 RM）
  → 更新策略

关键区别：
  ┌────────────────────────────────────┐
  │  PPO:   A(y) = R(y) - V(x)        │  需要 Value Model
  │  GRPO:  A(y_i) = (R_i - mean(R))  │  组内标准化
  │                  / std(R)          │  无需 Value Model
  └────────────────────────────────────┘

  PPO 需要: Policy + Reference + Reward + Value = 4 个模型
  GRPO 需要: Policy + Reference = 2 个模型 (RM 可选)
```

**DeepSeek-R1 使用 GRPO 的原因：**

1. **节省显存** — 不需要 Value Model，大模型训练时节省 ~25% 显存
2. **更稳定** — 组内相对排名天然归一化，避免 reward scale 问题
3. **支持规则奖励** — 数学/代码任务可以用正确性检查替代 RM
4. **可扩展** — 在 DeepSeek-R1 671B 上验证了有效性

> **面试话术：** "对齐技术的演进路线是 RLHF → DPO → GRPO。RLHF 三阶段虽然有效但成本高、训练不稳定。DPO 通过数学等价变换跳过了 Reward Model 和 PPO，直接用偏好数据优化策略。GRPO 是 DeepSeek 的创新，用组内相对排名替代绝对奖励打分，进一步去掉 Value Model，在 DeepSeek-R1 的训练中验证了其在大规模场景下的有效性。"

---

## 8.4 训练数据工程

### Q: 微调的训练数据如何准备？有哪些最佳实践？

**训练数据质量直接决定微调效果，"Garbage In, Garbage Out" 在 LLM 微调中尤为突出。**

**数据格式标准：**

```json
// SFT 数据格式 — Alpaca 格式
{
    "instruction": "将以下英文翻译成中文",
    "input": "The quick brown fox jumps over the lazy dog",
    "output": "敏捷的棕色狐狸跳过了懒惰的狗"
}

// SFT 数据格式 — ChatML/多轮对话格式（更推荐）
{
    "messages": [
        {"role": "system", "content": "你是一个专业的翻译助手"},
        {"role": "user", "content": "翻译：The quick brown fox..."},
        {"role": "assistant", "content": "敏捷的棕色狐狸..."}
    ]
}

// DPO 偏好数据格式
{
    "prompt": "解释什么是机器学习",
    "chosen": "机器学习是人工智能的一个分支，通过算法让计算机从数据中学习规律...",
    "rejected": "机器学习就是让机器学习的技术"
}
```

**数据工程 Pipeline：**

```
原始数据收集
     │
     ▼
┌──────────────┐
│ 数据清洗      │  去重、去噪、格式统一
│ (Cleaning)   │  去除低质量/有害内容
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 数据标注      │  人工标注 / LLM 辅助标注（Self-Instruct）
│ (Labeling)   │  质量审核：交叉验证、一致性检查
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 数据增强      │  Paraphrase、Back Translation
│ (Augment)    │  Self-Instruct：用 GPT-4 生成训练数据
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 数据平衡      │  类别平衡、难度分布
│ (Balance)    │  避免某类数据过多导致偏倚
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 数据验证      │  人工抽检、自动化 Lint
│ (Validate)   │  格式校验、token 长度统计
└──────────────┘
```

**数据质量 > 数据数量 — 关键实践：**

| 实践 | 说明 | 效果 |
|------|------|------|
| **去重** | MinHash / SimHash 去除重复和近似重复 | 防止过拟合 |
| **长度控制** | 过滤过短（<50 token）和过长（>2048 token）的样本 | 提高训练效率 |
| **质量评分** | 用更强的模型（GPT-4）给数据打分 | 筛选高质量数据 |
| **多样性采样** | 确保 instruction 多样性，不要大量重复句式 | 提高泛化能力 |
| **毒性过滤** | 用 Perspective API 等工具过滤有害内容 | 安全合规 |
| **人工抽检** | 随机抽取 5-10% 人工检查 | 发现系统性问题 |

**数据量参考：**

```
任务类型              推荐数据量       说明
──────────          ──────────      ──────
风格迁移/格式化       100-500 条      简单任务，少量高质量数据即可
领域知识注入          1K-10K 条       中等复杂度
复杂推理增强          10K-100K 条     需要覆盖多种推理模式
通用对话助手          100K+ 条        需要广泛覆盖
```

> **面试话术：** "数据工程是微调成功的关键。我的经验是，1000 条高质量数据的效果往往好于 10000 条低质量数据。我通常用 Self-Instruct 让 GPT-4 生成初始数据集，然后通过去重、质量评分、人工抽检迭代数据质量。数据格式上推荐 ChatML 多轮对话格式，更贴合实际使用场景。"

---

## 8.5 训练优化

### Q: DeepSpeed ZeRO 的三个阶段分别优化了什么？

**DeepSpeed ZeRO（Zero Redundancy Optimizer）通过消除数据并行中的冗余存储，让更大的模型用更少的 GPU 训练。**

**问题背景：数据并行的内存浪费**

```
传统数据并行（DDP）：
  每张 GPU 都存储完整的：模型参数 + 梯度 + 优化器状态

  以 7B 模型 (FP16) + Adam 优化器为例：
  ┌─────────────────────────────────────────┐
  │ 模型参数 (FP16):    7B × 2B = 14 GB     │
  │ 梯度 (FP16):       7B × 2B = 14 GB     │
  │ 优化器状态 (FP32):  7B × 12B = 84 GB    │
  │  ├── FP32 参数副本:  7B × 4B = 28 GB    │
  │  ├── FP32 动量:     7B × 4B = 28 GB    │
  │  └── FP32 方差:     7B × 4B = 28 GB    │
  │                                         │
  │ 总计每张 GPU: ~112 GB  ❌ 单卡放不下     │
  └─────────────────────────────────────────┘

  4 张 GPU 数据并行 = 4 × 112 GB = 448 GB
  但其实模型只有 112 GB，3/4 是冗余！
```

**ZeRO 三阶段逐步消除冗余：**

```
                        每张 GPU 显存（7B 模型 × 4 GPU）
                        ┌──────────────────────────────┐
  DDP (无 ZeRO):         │ Params  │ Grads  │ Optimizer │ = 112 GB
                        │  14GB   │  14GB  │   84GB    │
                        └──────────────────────────────┘

  ZeRO Stage 1:         │ Params  │ Grads  │ Opt/4     │ = 49 GB
  (分片优化器状态)        │  14GB   │  14GB  │   21GB    │
                        └──────────────────────────────┘

  ZeRO Stage 2:         │ Params  │ Grad/4 │ Opt/4     │ = 38.5 GB
  (+ 分片梯度)           │  14GB   │  3.5GB │   21GB    │
                        └──────────────────────────────┘

  ZeRO Stage 3:         │ Par/4   │ Grad/4 │ Opt/4     │ = 28 GB
  (+ 分片参数)           │  3.5GB  │  3.5GB │   21GB    │
                        └──────────────────────────────┘
```

| Stage | 分片内容 | 显存节省 | 通信量 | 适用场景 |
|-------|---------|---------|-------|---------|
| **Stage 1** | 优化器状态 | ~4x | 同 DDP | 最常用，无额外通信 |
| **Stage 2** | + 梯度 | ~8x | 同 DDP | 中等模型 |
| **Stage 3** | + 参数 | ~N×（N=GPU 数） | 增加 ~50% | 超大模型，无法单卡放下 |

**DeepSpeed vs FSDP 对比：**

| 维度 | DeepSpeed ZeRO | PyTorch FSDP |
|------|---------------|-------------|
| **开发者** | 微软 | Meta / PyTorch 团队 |
| **集成度** | 需要额外安装 | PyTorch 原生 |
| **功能丰富度** | 更多（offload、推理等） | 核心功能 |
| **配置复杂度** | JSON 配置文件 | Python API |
| **社区** | HuggingFace 深度集成 | PyTorch 生态 |
| **推荐** | 研究和训练 ⭐ | 生产部署 |

**混合精度训练：**

```
混合精度 = FP16/BF16 计算 + FP32 累积

前向计算:  FP16/BF16  ← 速度快、显存省
反向传播:  FP16/BF16  ← 梯度计算
参数更新:  FP32       ← 精度关键步骤用高精度

BF16 vs FP16:
  ┌────────────────────────────────────┐
  │  FP16: 1 sign + 5 exp + 10 mantissa │
  │  BF16: 1 sign + 8 exp + 7 mantissa  │
  │                                      │
  │  BF16 范围更大（不容易 overflow）      │
  │  FP16 精度更高（小数更准）            │
  │  推荐：Ampere+ GPU 用 BF16 ⭐        │
  └────────────────────────────────────┘
```

> **面试话术：** "DeepSpeed ZeRO 通过分片消除数据并行中的冗余存储。Stage 1 分片优化器状态，节省约 4 倍显存且不增加通信量，是最常用的配置。Stage 2 额外分片梯度，Stage 3 连模型参数也分片，适合超大模型。实际训练中，我通常用 ZeRO Stage 2 + BF16 混合精度，配合 gradient checkpointing，能在有限 GPU 上高效训练。"

---

## 8.6 TRL v1.0 统一训练框架

### Q: TRL 是什么？它如何统一了 LLM 训练流程？

**TRL（Transformer Reinforcement Learning）是 HuggingFace 推出的 LLM 训练框架，v1.0 版本统一了 SFT、DPO、RLHF 等所有训练范式。**

```
TRL v1.0 统一 API 架构：

  ┌──────────────────────────────────────────────┐
  │                  TRL v1.0                      │
  │                                                │
  │  ┌──────────┐ ┌──────────┐ ┌──────────────┐   │
  │  │SFTTrainer│ │DPOTrainer│ │ GRPOTrainer  │   │
  │  └────┬─────┘ └────┬─────┘ └──────┬───────┘   │
  │       │            │              │            │
  │       └────────────┼──────────────┘            │
  │                    │                           │
  │            ┌───────┴────────┐                  │
  │            │  Unified API   │                  │
  │            │  统一数据格式   │                  │
  │            │  统一训练接口   │                  │
  │            └───────┬────────┘                  │
  │                    │                           │
  │  ┌─────────────────┴──────────────────────┐   │
  │  │  HuggingFace Transformers + PEFT       │   │
  │  │  + Accelerate + DeepSpeed/FSDP         │   │
  │  └────────────────────────────────────────┘   │
  └──────────────────────────────────────────────┘
```

**TRL 支持的训练方法：**

| Trainer | 用途 | 数据格式 |
|---------|------|---------|
| `SFTTrainer` | 监督微调 | `{"messages": [...]}` |
| `DPOTrainer` | 直接偏好优化 | `{"prompt", "chosen", "rejected"}` |
| `GRPOTrainer` | 组相对策略优化 | `{"prompt": "..."}` + reward function |
| `PPOTrainer` | 近端策略优化 | `{"query": "..."}` + reward model |
| `KTOTrainer` | Kahneman-Tversky 优化 | `{"prompt", "completion", "label"}` |
| `ORPOTrainer` | 无需 reference model | `{"prompt", "chosen", "rejected"}` |

**完整训练 Pipeline 代码示例：**

```python
from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============ Step 1: SFT ============
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

sft_dataset = load_dataset("json", data_files="sft_data.jsonl")

sft_trainer = SFTTrainer(
    model=model,
    train_dataset=sft_dataset["train"],
    args=SFTConfig(
        output_dir="./sft-output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        bf16=True,
        gradient_checkpointing=True,
        # DeepSpeed 集成
        deepspeed="ds_config_zero2.json",
    ),
)
sft_trainer.train()

# ============ Step 2: DPO ============
dpo_dataset = load_dataset("json", data_files="dpo_data.jsonl")

dpo_trainer = DPOTrainer(
    model=model,                    # SFT 后的模型
    ref_model=None,                 # None = 自动用初始权重
    train_dataset=dpo_dataset["train"],
    args=DPOConfig(
        output_dir="./dpo-output",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        learning_rate=5e-7,         # DPO 学习率要小
        beta=0.1,                   # KL 惩罚系数
        bf16=True,
    ),
)
dpo_trainer.train()
```

**前端类比：** TRL 之于 LLM 训练，就像 Next.js 之于前端开发——Next.js 统一了 SSR/SSG/ISR/CSR 多种渲染方式，开发者不需要自己搭建 webpack。TRL 统一了 SFT/DPO/PPO/GRPO 多种训练方式，开发者不需要自己写训练循环。

> **面试话术：** "TRL 是 HuggingFace 的统一 LLM 训练框架。v1.0 的核心价值是统一了数据格式和训练接口——不管是 SFT、DPO 还是 GRPO，都用一致的 API。它和 PEFT、DeepSpeed 深度集成，配合 Accelerate 可以无缝扩展到多 GPU 训练。在实际项目中，我用 TRL 的 SFTTrainer + DPOTrainer 完成了完整的模型对齐流程。"

---

## 8.7 微调 vs RAG vs Prompt Engineering 选型

### Q: 如何选择微调、RAG 和 Prompt Engineering？

**三种技术解决不同层面的问题，选型取决于你的核心需求。**

```
决策树：

  你的需求是什么？
       │
       ├── 需要最新/私有知识？
       │        │
       │        ├── 是 → RAG（检索增强生成）
       │        │      知识可频繁更新
       │        │
       │        └── 否 → 继续 ↓
       │
       ├── 需要特定输出格式/风格/行为？
       │        │
       │        ├── 简单格式 → Prompt Engineering
       │        │      Few-shot + 格式说明
       │        │
       │        └── 复杂/稳定格式 → Fine-tuning
       │               需要高一致性
       │
       ├── 需要降低推理成本？
       │        │
       │        └── 是 → Fine-tuning 小模型
       │               替代大模型 API 调用
       │
       └── 以上都不是 → Prompt Engineering 先试
                        成本最低，迭代最快
```

**三种方案全方位对比：**

| 维度 | Prompt Engineering | RAG | Fine-tuning |
|------|-------------------|-----|-------------|
| **适用场景** | 通用任务、快速原型 | 知识密集型、需要时效性 | 特定风格/行为、性能优化 |
| **开发成本** | 低（几小时） | 中（几天-几周） | 高（几周-几月） |
| **数据需求** | 几个示例 | 文档知识库 | 数百-数万条标注数据 |
| **知识更新** | 即时（改 prompt） | 快（更新索引） | 慢（重新训练） |
| **推理成本** | 高（长 prompt） | 中（检索+生成） | 低（小模型即可） |
| **可解释性** | 高（prompt 可读） | 高（可引用来源） | 低（黑盒模型） |
| **幻觉控制** | 中等 | 好（有来源） | 中等 |
| **前端类比** | CSS inline style | SSR 数据获取 | 编译时主题定制 |

**组合使用策略（推荐）：**

```
最佳实践：三者不是互斥的，而是可以组合

  ┌──────────────────────────────────────────┐
  │  1. Prompt Engineering（基础层）          │
  │     ├── System Prompt 定义角色和规则       │
  │     └── Few-shot 示例引导格式             │
  │                                          │
  │  2. RAG（知识层）                         │
  │     ├── 检索相关文档补充上下文             │
  │     └── 提供引用来源减少幻觉              │
  │                                          │
  │  3. Fine-tuning（能力层）                 │
  │     ├── 训练领域理解能力                   │
  │     └── 固化输出格式和风格                 │
  └──────────────────────────────────────────┘

  实际架构：
  Fine-tuned Model + RAG 检索 + Crafted Prompt = 最佳效果
```

> **面试话术：** "我的选型原则是从轻到重：先试 Prompt Engineering（几小时见效），不够再加 RAG（几天搭建），最后考虑微调（几周开发）。三者也可以组合使用——用微调过的模型做基座，RAG 补充实时知识，Prompt Engineering 控制输出格式。选型的关键维度是数据需求、迭代速度和推理成本。"

---

## 8.8 评估与迭代

### Q: 微调后如何评估模型效果？如何持续迭代？

**微调评估不是一个一次性步骤，而是一个持续循环。**

**评估指标体系：**

```
微调评估指标
├── 自动指标
│   ├── Loss / Perplexity        ← 训练过程监控
│   ├── BLEU / ROUGE             ← 文本相似度（有局限性）
│   ├── Pass@K                   ← 代码生成准确率
│   └── 领域特定指标              ← F1, Accuracy 等
├── LLM-as-Judge
│   ├── GPT-4 / Claude 评分      ← 多维度打分（1-5）
│   ├── Pairwise Comparison      ← A/B 对比哪个更好
│   └── Arena / Chatbot Arena    ← ELO 排名
└── 人工评估
    ├── 盲评                     ← 不知道哪个是哪个模型
    ├── 多评委一致性              ← Kappa / ICC 系数
    └── 领域专家评审              ← 专业准确性
```

**训练过程监控 — 关键信号：**

| 信号 | 正常 | 异常（需要干预） |
|------|------|---------------|
| Train Loss | 平稳下降 | 震荡、突然上升 |
| Eval Loss | 与 train loss 同步下降 | 上升 → 过拟合 ⚠️ |
| Learning Rate | 按 schedule 变化 | N/A |
| Gradient Norm | 稳定范围 | 爆炸（>100）或消失（<1e-7） |

**评估 Checklist：**

```
微调前后对比 Checklist：
  □ Base model 在目标任务上的基线表现
  □ 微调后在目标任务上的表现提升
  □ 在通用 benchmark 上没有明显退化
  □ 在安全评估上没有退化
  □ 推理速度没有显著影响
  □ 边界 case 测试（对抗性输入、长文本等）
```

**迭代改进策略：**

```
效果不好？诊断流程：

  模型表现差
       │
       ├── 训练 loss 不下降？
       │     → 学习率太小 / 数据格式错误 / 冻结层太多
       │
       ├── 训练 loss 低但 eval 高？
       │     → 过拟合 → 加数据 / 降 rank / 加 dropout / 减 epoch
       │
       ├── 特定类型回答差？
       │     → 该类型训练数据不足 → 补充数据
       │
       ├── 格式不稳定？
       │     → 增加格式化数据比例 / 加格式约束 prompt
       │
       └── 通用能力退化？
             → 灾难性遗忘 → 混入通用数据 / 降低学习率
```

> **面试话术：** "微调评估我采用三层体系：自动指标监控训练过程（loss, perplexity），LLM-as-Judge 做快速质量评估（用 GPT-4 打分），最后关键场景做人工盲评。迭代时最重要的是诊断问题根源——过拟合就加数据降 rank，能力退化就混入通用数据，格式不稳定就补充格式化样本。"

---

## 导航

| 上一章 | 当前章 | 下一章 |
|--------|--------|--------|
| [07 - MCP 协议与工具系统](./07-mcp.md) | **08 - 模型训练与微调** | [09 - 推理优化](./09-inference-optimization.md) |
