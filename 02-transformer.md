# 02 - Transformer 架构

> **难度：** ⭐⭐⭐⭐ | **定位：** 理解 LLM 的底层原理，面试高频考点
>
> **前端类比：** Transformer 之于 LLM，就像 V8 引擎之于 Node.js——你不一定要会写引擎，但理解原理能帮你写出更好的 AI 应用。

## 本章知识树

```
Transformer 架构
├── 2.1 整体架构（Encoder-Decoder）
├── 2.2 Self-Attention 机制
├── 2.3 Multi-Head Attention
├── 2.4 MQA / GQA 优化
├── 2.5 位置编码（Sinusoidal / RoPE / ALiBi）
├── 2.6 Feed-Forward Network 与残差连接
├── 2.7 BERT vs GPT：Encoder vs Decoder
├── 2.8 现代 LLM 架构演进
└── 2.9 SSM 与 Mamba：Transformer 的挑战者
```

---

## 2.1 整体架构

### Q: Transformer 是什么？为什么它取代了 RNN/LSTM？

**Transformer 是 2017 年 Google 提出的序列建模架构（"Attention Is All You Need"），是所有现代 LLM 的基础。**

```
Transformer 架构全景：

┌─────────────────────────────────┐
│          Decoder × N            │
│  ┌───────────────────────────┐  │
│  │ Masked Self-Attention     │  │   ← GPT/Claude/LLaMA 只用这一半
│  │ + Cross-Attention         │  │
│  │ + Feed-Forward            │  │
│  │ + Residual + LayerNorm    │  │
│  └───────────────────────────┘  │
├─────────────────────────────────┤
│          Encoder × N            │
│  ┌───────────────────────────┐  │
│  │ Self-Attention            │  │   ← BERT 只用这一半
│  │ + Feed-Forward            │  │
│  │ + Residual + LayerNorm    │  │
│  └───────────────────────────┘  │
├─────────────────────────────────┤
│  Input Embedding + 位置编码     │
└─────────────────────────────────┘
```

**为什么取代 RNN/LSTM？**

| 问题 | RNN/LSTM | Transformer |
|------|----------|-------------|
| 并行计算 | 必须串行处理（token by token） | 全并行（所有 token 同时计算） |
| 长程依赖 | 梯度消失，难以记住 100+ 步前的信息 | Attention 直接连接任意距离的 token |
| 训练速度 | 慢（无法利用 GPU 并行） | 快（矩阵运算，GPU 友好） |
| 可扩展性 | 参数量难以扩大 | Scaling Law 验证，越大越强 |

**前端类比：** RNN 像是 `for` 循环处理数组——必须一个个来。Transformer 像是 `Array.map()` + 并行计算——所有元素同时处理。

---

## 2.2 Self-Attention 机制

### Q: Self-Attention 的原理是什么？Q、K、V 分别代表什么？

**Self-Attention 的核心思想：让每个 token "看到"序列中所有其他 token，并决定关注哪些。**

**Q/K/V 的直觉理解：**

```
想象你在图书馆找书：

Q（Query，查询）= 你要找什么书
K（Key，键）    = 每本书的标签/索引
V（Value，值）  = 每本书的实际内容

过程：
1. 你拿着 Q（想找的书）去跟每本书的 K（标签）做比较
2. 比较结果 = Attention Score（注意力分数）
3. 分数高的书（更相关）→ 你更多地参考它的 V（内容）
4. 最终结果 = 所有书内容的加权组合
```

**数学公式：**

```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V

其中：
- Q = X · W_Q    （输入乘以查询权重矩阵）
- K = X · W_K    （输入乘以键权重矩阵）
- V = X · W_V    （输入乘以值权重矩阵）
- d_k = 维度      （缩放因子，防止点积过大）
- softmax        （归一化为概率分布）
```

**具体例子：**

```
句子："猫 坐在 垫子 上"

计算 "坐在" 对其他词的注意力：
  "坐在" → "猫"：  0.4  （谁坐？关注主语）
  "坐在" → "坐在"：0.1  （自己）
  "坐在" → "垫子"：0.4  （坐在哪？关注宾语）
  "坐在" → "上"：  0.1  （位置补充）

→ "坐在" 的最终表示 = 0.4×V(猫) + 0.1×V(坐在) + 0.4×V(垫子) + 0.1×V(上)
```

**为什么除以 √d_k？** 维度 d_k 越大，QK^T 的点积数值越大，softmax 后会趋近 one-hot（梯度消失）。除以 √d_k 让数值保持在合理范围。

---

## 2.3 Multi-Head Attention

### Q: 什么是 Multi-Head Attention？为什么需要多个 "头"？

**Multi-Head Attention = 把 Attention 拆成多个并行的 "头"，每个头关注不同的语义模式。**

```
单头 Attention：
  一个 Q/K/V → 一种关注模式

Multi-Head Attention（假设 8 个头）：
  Head 1: 关注语法关系（主谓宾）
  Head 2: 关注指代关系（"它"指代谁）
  Head 3: 关注相邻词的搭配
  Head 4: 关注长距离依赖
  Head 5-8: 其他语义模式

  最终：拼接所有头的输出 → 线性投影
```

**计算过程：**

```python
# 伪代码
def multi_head_attention(X, num_heads=8):
    d_model = X.shape[-1]        # 如 768
    d_head = d_model // num_heads # 768/8 = 96

    heads = []
    for i in range(num_heads):
        Q_i = X @ W_Q[i]  # (seq_len, 96)
        K_i = X @ W_K[i]  # (seq_len, 96)
        V_i = X @ W_V[i]  # (seq_len, 96)
        head_i = attention(Q_i, K_i, V_i)
        heads.append(head_i)

    # 拼接 + 线性投影
    concat = torch.cat(heads, dim=-1)  # (seq_len, 768)
    output = concat @ W_O               # (seq_len, 768)
    return output
```

**关键洞察：** 多头的总计算量 ≈ 单头（因为每个头的维度缩小了），但能力更强，因为每个头可以学习不同的注意力模式。

---

## 2.4 MQA 与 GQA

### Q: 什么是 MQA 和 GQA？它们解决了什么问题？

**问题：** 标准 MHA（Multi-Head Attention）在推理时，每个头都有独立的 K/V，KV Cache 占用大量显存。

**三种方案对比：**

| 方案 | K/V 的数量 | KV Cache 大小 | 质量 | 代表模型 |
|------|-----------|--------------|------|----------|
| **MHA** | 每个头独立 K/V | 大（32 组） | 最好 | GPT-3 |
| **MQA** | 所有头共享 1 组 K/V | 最小（1 组） | 有损失 | PaLM |
| **GQA** | 分组共享（如 8 组） | 中等（8 组） | 接近 MHA | LLaMA 2/3, Gemini |

```
MHA（Multi-Head Attention）：每个头都有独立的 Q/K/V
  Head 1: Q1, K1, V1
  Head 2: Q2, K2, V2
  ...
  Head 32: Q32, K32, V32
  → KV Cache: 32 组

MQA（Multi-Query Attention）：所有头共享一组 K/V
  Head 1: Q1, K_shared, V_shared
  Head 2: Q2, K_shared, V_shared
  ...
  → KV Cache: 1 组（节省 32 倍内存！）

GQA（Grouped-Query Attention）：分组共享（折中方案）
  Group 1: Q1-Q4 共享 K1, V1
  Group 2: Q5-Q8 共享 K2, V2
  ...
  → KV Cache: 8 组（节省 4 倍内存）
```

**面试话术：**
> "GQA 是 MHA 和 MQA 的折中方案——LLaMA 2 以后的主流模型基本都用 GQA。它把 32 个 attention head 分成 8 组，每组共享 K/V，推理时 KV Cache 缩小 4 倍，质量接近 MHA。"

---

## 2.5 位置编码

### Q: 为什么 Transformer 需要位置编码？主流方案有哪些？

**问题：** Self-Attention 是 "置换不变" 的——"猫吃鱼" 和 "鱼吃猫" 的 Attention 矩阵一样（只是关注了哪些词，不知道顺序）。位置编码告诉模型每个 token 在序列中的位置。

**三种主流方案：**

| 方案 | 原理 | 能否外推到训练长度之外 | 代表模型 |
|------|------|----------------------|----------|
| **正弦/余弦（Sinusoidal）** | 用不同频率的三角函数 | 理论上可以 | 原始 Transformer |
| **可学习位置编码（Learned）** | 每个位置学一个向量 | 不能 | GPT-2, BERT |
| **RoPE（旋转位置编码）** | 旋转矩阵编码相对位置 | 支持（配合 NTK 插值） | LLaMA, Qwen, DeepSeek |
| **ALiBi** | Attention 分数加位置偏置 | 天然支持 | BLOOM |

**RoPE 为什么是主流？**

1. **相对位置：** 编码的是 token 之间的相对距离，而不是绝对位置
2. **外推性好：** 配合 NTK-aware Scaling，训练 4K 可以推理 128K
3. **计算高效：** 只是旋转操作，几乎不增加计算量

---

## 2.6 Feed-Forward 与残差连接

### Q: Transformer 中的 FFN 和残差连接有什么作用？

**Feed-Forward Network（FFN）：** 每个 Attention 层后面跟着一个两层全连接网络。

```
FFN(x) = W₂ · activation(W₁ · x + b₁) + b₂

Attention 负责：token 之间的交互（"看别人"）
FFN 负责：     每个 token 自身的非线性变换（"思考自己"）
```

**现代 LLM 中 FFN 的进化：**
- 原始 Transformer：ReLU 激活
- LLaMA/现代模型：SwiGLU 激活（效果更好）
- MoE（Mixture of Experts）：FFN 被替换为多个 "专家"，每次只激活部分

**残差连接（Residual Connection）：**

```
output = LayerNorm(x + Sublayer(x))

作用：
1. 缓解梯度消失（梯度可以直接跳过层传播）
2. 让模型更容易训练（类似 ResNet 的思路）
3. 保留原始信息（sublayer 只需要学"增量"）
```

**前端类比：** 残差连接就像 React 的 `props` 直接透传——子组件只处理自己需要修改的部分，其他 props 原封不动传下去。

---

## 2.7 BERT vs GPT

### Q: BERT 和 GPT 有什么区别？为什么现在 GPT 架构（Decoder-Only）成为主流？

| 维度 | BERT（Encoder-Only） | GPT（Decoder-Only） |
|------|---------------------|---------------------|
| 注意力方向 | 双向（看前看后） | 单向（只看前面） |
| 训练目标 | 完形填空（MLM） | 预测下一个词（CLM） |
| 擅长任务 | 理解类（分类/NER/相似度） | 生成类（对话/写作/代码） |
| 代表模型 | BERT, RoBERTa | GPT-4, Claude, LLaMA |
| 当前状态 | 仍用于 Embedding 模型 | LLM 主流架构 |

```
BERT（双向）：
  "我 爱 [MASK] 京"
  → 能同时看到"我爱"和"京"
  → 预测 [MASK] = "北"

GPT（单向）：
  "我 爱 北" → 预测下一个 = "京"
  → 只能看前面的内容
```

**为什么 Decoder-Only 成为主流？**

1. **Scaling Law 更好：** 同样的参数量，Decoder-Only 的涌现能力更强
2. **生成能力天然：** 自回归架构天然适合文本生成
3. **简洁统一：** 一种架构解决所有任务（理解+生成都行）
4. **训练效率：** 每个 token 都是训练信号（不像 BERT 只有 15% mask）

**BERT 还有用吗？** 有——Embedding 模型（如 BGE、GTE）仍然基于 BERT 架构，因为双向理解对语义表示更好。

---

## 2.8 现代 LLM 架构演进

### Q: 从 GPT-1 到现在，LLM 架构有哪些关键演进？

```
2017  Transformer（Google）       ← 基础架构
  ↓
2018  GPT-1（OpenAI）             ← 证明 Decoder-Only + 预训练可行
2018  BERT（Google）              ← 双向理解，理解类任务王者
  ↓
2019  GPT-2（1.5B）               ← 证明 Scaling 有效
  ↓
2020  GPT-3（175B）               ← In-context Learning 涌现
  ↓
2022  InstructGPT + RLHF          ← 对齐技术，从"能说"到"说得好"
2022  ChatGPT（GPT-3.5-turbo）    ← 对话式 AI 爆发
  ↓
2023  GPT-4                       ← 多模态（图+文）
2023  LLaMA 1/2（Meta）           ← 开源 LLM 时代开启
  ↓
2024  Mixture of Experts（MoE）    ← Mixtral, DeepSeek-V2
      → 激活参数 << 总参数，推理更高效
  ↓
2025  DeepSeek-V3/R1              ← MoE + 长思考链（CoT）
2025  LLaMA 3.1/4                 ← 开源追赶闭源
  ↓
2026  Claude 4, GPT-5             ← 更强推理 + 更长上下文 + Agent 原生能力
```

**关键技术节点：**

| 时间 | 技术 | 意义 |
|------|------|------|
| 2017 | Self-Attention | 取代 RNN，开启并行时代 |
| 2020 | Scaling Law | 证明模型越大越好 |
| 2022 | RLHF | 让模型对齐人类偏好 |
| 2024 | MoE | 低成本做大模型 |
| 2024 | RoPE + 长上下文 | 从 4K 扩展到 1M |
| 2025 | Reasoning Model | DeepSeek-R1 等推理专用模型 |

**面试话术：**
> "Transformer 架构从 2017 年至今的演进，核心围绕三条线：1）Scaling——模型越来越大（GPT-3 → GPT-4）；2）效率——MoE、GQA 让大模型推理更便宜；3）对齐——RLHF/DPO 让模型行为更安全可控。"

---

## 2.9 SSM 与 Mamba

### Q: 什么是 SSM 和 Mamba？它们能取代 Transformer 吗？

**SSM（State Space Model，状态空间模型）是 2024-2026 年出现的 Transformer 替代架构，Mamba 是其中最成功的实现。**

**Transformer 的痛点：**

```
Self-Attention 的复杂度 = O(n²)
  → 序列长度翻倍，计算量翻 4 倍
  → 128K context 已经很勉强，1M+ 非常昂贵

KV Cache 的内存占用：
  → 随序列长度线性增长
  → 长上下文推理的主要瓶颈
```

**Mamba 的核心思想：**

```
Transformer：每个 token 关注所有其他 token（全局注意力）
  → 质量好但 O(n²)

Mamba / SSM：通过隐状态（hidden state）压缩历史信息
  → 类似 RNN，但用结构化矩阵保证可并行训练
  → 推理时 O(1) 每个 token（不需要 KV Cache！）
  → 训练时 O(n) 通过并行扫描算法
```

| 维度 | Transformer | Mamba (SSM) |
|------|-------------|-------------|
| 注意力 | 全局（看所有 token） | 局部（通过隐状态传递） |
| 训练复杂度 | O(n²) | O(n) |
| 推理复杂度 | O(n)（含 KV Cache） | O(1) 每个 token |
| KV Cache | 需要，占大量显存 | 不需要 |
| 长序列能力 | 需要特殊优化 | 天然高效 |
| 短序列质量 | 最好 | 略差 |

**2026 年的趋势：混合架构**

纯 Mamba 的质量不如纯 Transformer，但混合架构（Transformer 层 + Mamba 层交替使用）开始展现优势：

```
Jamba（AI21）：Transformer + Mamba 混合
  → 256K context，推理效率比纯 Transformer 高 5 倍

Zamba（Zyphra）：70% Mamba 层 + 30% Attention 层
  → 在同等参数量下，推理速度更快

技术本质：
  Attention 层负责"精确关注"（短距离复杂关系）
  Mamba 层负责"高效传递"（长距离信息压缩）
```

**面试话术：**
> "Mamba/SSM 是 Transformer 的潜在挑战者，核心优势是 O(n) 训练复杂度和无 KV Cache 推理。但目前纯 SSM 的质量还追不上纯 Transformer，2026 年的趋势是混合架构——用 Transformer 处理需要精确注意力的部分，用 Mamba 处理长距离信息传递，兼顾质量和效率。"

---

[← 上一章：01 - LLM 基础](./01-llm-fundamentals.md) | [下一章：03 - Prompt Engineering →](./03-prompt-engineering.md)
