# 00 - 数学基础：AI 背后的数学直觉

> **难度：** ⭐ | **定位：** 了解级别，建立直觉而非推导公式
>
> **前端类比：** 你不需要懂 V8 引擎的 JIT 编译原理也能写好 JavaScript。同理，你不需要手推矩阵求导也能做好 AI 应用——但你需要知道"这些数学在干什么"。

## 你需要知道到什么程度？

**一句话：能用直觉解释概念，不需要手写公式。**

作为 AI 应用开发者（而非算法研究员），你的目标是：

| 需要 | 不需要 |
|------|--------|
| 知道 gradient descent 是在"下山" | 手推 backpropagation 的链式法则 |
| 知道 cross-entropy 衡量"预测有多差" | 推导 cross-entropy 的数学形式 |
| 知道 embedding 是一个 vector | 证明 vector space 的完备性 |
| 知道 softmax 把数字变成概率 | 证明 softmax 的数值稳定性 |

**面试中被问到数学时的策略：** 用直觉 + 类比回答，展示你理解"为什么要用这个数学工具"，而不是背公式。

## 本章知识树

```
数学基础（了解级别）
├── 0.1 线性代数核心概念
│   └── 向量、矩阵乘法、点积、转置 — 为什么 Attention 是矩阵运算
├── 0.2 概率与统计
│   └── 概率分布、条件概率、Bayes 定理、softmax 为什么是概率
├── 0.3 微积分与梯度
│   └── 导数、梯度、梯度下降 — 模型怎么"学习"
├── 0.4 最优化
│   └── 损失函数、SGD、Adam — 训练的核心循环
└── 0.5 信息论基础
    └── 熵、交叉熵、KL 散度 — Loss 函数的来源
```

---

## 0.1 线性代数核心概念

### Q: 向量和矩阵在 AI 中扮演什么角色？为什么说"AI 本质上就是矩阵运算"？

**一句话：AI 模型的所有数据都是向量，所有计算都是矩阵乘法。**

**向量（Vector）= 一组有序数字，表示空间中的一个点。**

```
// 前端类比：向量就是一个数字数组
const embedding = [0.2, -0.5, 0.8, 0.1, ...] // 1536 维向量

// 文本变成向量后，语义相似的文本在空间中距离近
"猫" → [0.2, 0.8, ...]
"狗" → [0.3, 0.7, ...]  // 离"猫"近
"汽车" → [-0.5, 0.1, ...] // 离"猫"远
```

**矩阵（Matrix）= 二维数组，表示一种"变换"。**

```
// 前端类比：矩阵就是 2D array
const matrix = [
  [1, 2, 3],
  [4, 5, 6]
]
// 矩阵 × 向量 = 把向量从一个空间变换到另一个空间
// 神经网络的每一层就是一次矩阵变换
```

**点积（Dot Product）= 两个向量的相似度度量。**

```
// 点积 = 对应元素相乘再求和
dotProduct([1, 2, 3], [4, 5, 6]) = 1×4 + 2×5 + 3×6 = 32

// 在 AI 中的关键应用：
// 1. Embedding 相似度搜索 → 就是算两个向量的点积（或 cosine similarity）
// 2. Attention score → Q 和 K 的点积，越大说明越"相关"
```

**连接到 Transformer：** Attention 机制的核心公式 `Attention(Q, K, V) = softmax(QK^T / √d) V` 全是矩阵运算——Q、K、V 是矩阵，`QK^T` 是矩阵乘法，softmax 逐行应用，最终再乘 V。理解了"矩阵乘法 = 批量点积"，就理解了 Attention 在计算什么。

**面试话术：**
> "在 AI 里，向量是数据的表示方式——文字、图片最终都变成高维向量。矩阵乘法是核心计算——神经网络每一层都是矩阵变换。Attention 机制本质上就是用点积来计算 query 和 key 的相似度，然后加权聚合 value。这就是为什么 GPU 对 AI 很重要——GPU 就是并行矩阵计算的硬件。"

---

## 0.2 概率与统计

### Q: 为什么 LLM 的输出是概率分布？softmax 和 temperature 是怎么回事？

**一句话：LLM 每一步输出所有词的概率分布，softmax 把原始分数变成概率，temperature 控制分布的"形状"。**

**概率分布（Probability Distribution）：**

```
// LLM 预测下一个 token 时，输出每个候选词的"分数"（logits）
logits = { "猫": 5.2, "狗": 3.1, "桌子": 0.3, "的": 1.8, ... }

// 问题：这些分数不是概率（有负数、不归一）
// 解决：用 softmax 转成概率分布
```

**softmax = 把任意数字变成"概率"（0-1 之间，总和为 1）：**

```
// softmax 做两件事：
// 1. 用 e^x 把所有数字变成正数
// 2. 除以总和，让它们加起来 = 1

softmax({ "猫": 5.2, "狗": 3.1, "桌子": 0.3 })
→ { "猫": 0.82, "狗": 0.10, "桌子": 0.06 }
// 现在可以按概率采样了！
```

**Temperature 改变分布形状：**

```
// 原理：softmax 之前，logits 除以 temperature
// temperature = 0.1 → 分布变尖锐（几乎确定选最高分的词）→ 更确定、更保守
// temperature = 1.0 → 正常分布
// temperature = 2.0 → 分布变平坦（各词概率更均匀）→ 更随机、更有创意
```

**条件概率与 Bayes 定理：**

LLM 本质上在算条件概率：P(下一个词 | 前面所有词)。每生成一个 token，就是在已知上文的条件下，选择最可能的下一个词。这就是 **autoregressive generation**。

Bayes 定理 `P(A|B) = P(B|A)P(A) / P(B)` 的直觉：新的证据（B）会更新我们对 A 的信念。在 AI 中，这个思想出现在后验推断、RLHF 的 reward modeling 等场景。

**面试话术：**
> "LLM 的输出层经过 softmax，把 logits 转成词汇表上的概率分布，然后通过采样策略（temperature、top-p）选择下一个 token。temperature 本质上是在控制这个概率分布的熵——低 temperature 减少随机性，高 temperature 增加多样性。"

---

## 0.3 微积分与梯度

### Q: 模型是怎么"学习"的？梯度下降到底在做什么？

**一句话：梯度告诉模型"往哪个方向调参数能减少错误"，梯度下降就是沿着这个方向不断调整。**

**导数 / 梯度的直觉：**

```
// 导数 = "变化率" = 函数在某点的斜率
// 梯度 = 多维空间中的导数，指向函数增长最快的方向

// 想象你站在山上，闭着眼睛，想下山：
// 1. 用脚感受哪个方向最陡峭（= 计算梯度）
// 2. 朝最陡峭的反方向迈一步（= 梯度下降）
// 3. 重复，直到走到谷底（= loss 最小化）
```

**梯度下降（Gradient Descent）：**

```
// 伪代码：
while (loss > 可接受的值) {
  gradient = 计算loss对每个参数的梯度  // 方向
  parameters = parameters - learning_rate * gradient  // 更新
}

// learning_rate（学习率）= 每一步走多大
// 太大 → 跳过最优点（震荡）
// 太小 → 收敛太慢
```

**前端类比：** 梯度下降有点像 binary search 找最优值——每一步都在缩小搜索范围。区别是 binary search 在一维上找精确值，梯度下降在百亿维空间中找近似最优点。

**Backpropagation（反向传播）= 高效计算梯度的算法：**

- 神经网络有很多层，每层有很多参数
- 直接算每个参数的梯度太慢
- Backprop 利用**链式法则（chain rule）**，从输出层向输入层逐层传递梯度
- 直觉：error 从输出层"反向流动"回去，告诉每一层"你该怎么调"

**面试话术：**
> "模型学习的核心机制是梯度下降：先用 forward pass 算出预测和真实值之间的 loss，再用 backpropagation 反向算出每个参数的梯度，最后用 optimizer 按梯度方向更新参数。这个 forward → loss → backward → update 的循环重复数亿次，模型就'学会'了。"

---

## 0.4 最优化

### Q: 损失函数和优化器是什么？SGD 和 Adam 有什么区别？

**一句话：损失函数衡量"模型有多差"，优化器决定"怎么调参数让它变好"。**

**损失函数（Loss Function）= 模型预测 vs 真实答案 的差距分数：**

```
// 类比：考试批改
prediction = "猫"（模型预测）
ground_truth = "狗"（正确答案）
loss = 计算差距(prediction, ground_truth)  // 越小越好

// 常见损失函数：
// - Cross-Entropy Loss → LLM 训练标配（下节详解）
// - MSE（均方误差） → 回归任务
// - Contrastive Loss → Embedding 训练
```

**优化器（Optimizer）= 根据梯度更新参数的策略：**

| 优化器 | 原理 | 类比 |
|--------|------|------|
| **SGD** | 每次用一小批数据算梯度，沿反方向更新 | 闭眼下山，每步只看脚下 |
| **SGD + Momentum** | 加入"惯性"，避免在山谷来回震荡 | 下山时像滚球，有惯性 |
| **Adam** | 自适应学习率 + momentum，每个参数有独立的学习率 | 智能导航下山，陡的地方小步走，平的地方大步走 |

**为什么 Adam 是主流？** 因为它结合了两个优点：
1. **Momentum**：记住之前的梯度方向，避免震荡
2. **自适应学习率**：对每个参数自动调整步长——更新频繁的参数用小学习率，罕见的参数用大学习率

**训练的核心循环（The Training Loop）：**

```
for each batch in training_data:
  // 1. Forward Pass：数据流过模型，得到预测
  prediction = model(batch.input)

  // 2. 计算 Loss：预测 vs 真实答案
  loss = crossEntropyLoss(prediction, batch.label)

  // 3. Backward Pass：反向传播，计算梯度
  loss.backward()

  // 4. Update：优化器更新参数
  optimizer.step()

  // 5. 清零梯度，准备下一轮
  optimizer.zeroGrad()
```

**面试话术：**
> "训练的核心是一个 forward-backward-update 循环。Loss function 定义优化目标，optimizer 决定怎么走。现在主流用 Adam 或其变体 AdamW（带 weight decay 防止过拟合）。对于 LLM 预训练，通常还会配合 learning rate warmup 和 cosine decay 的调度策略。"

---

## 0.5 信息论基础

### Q: Cross-Entropy Loss 到底在衡量什么？为什么它是 LLM 训练的标准损失函数？

**一句话：Cross-Entropy 衡量"模型的预测分布离真实分布有多远"，越小说明模型预测越准。**

**先理解熵（Entropy）= 不确定性的度量：**

```
// 抛硬币：正反各 50%，熵最大（最不确定）
// 作弊硬币：99% 正面，熵很小（几乎确定是正面）

// 在 LLM 中：
// 高熵输出 → 模型不确定下一个词是什么 → 每个词概率差不多
// 低熵输出 → 模型很确定 → 某个词概率远高于其他词
```

**交叉熵（Cross-Entropy）= 用"模型的分布"来编码"真实分布"的代价：**

```
// 直觉：如果模型预测和真实分布完全一致，交叉熵 = 熵（最小值）
// 如果模型预测偏差很大，交叉熵 >> 熵（额外代价）

// LLM 训练时：
// 真实分布 = one-hot（正确的下一个词概率为 1，其他为 0）
// 模型分布 = softmax 输出的概率分布
// Cross-Entropy Loss = -log(模型给正确词的概率)

// 例：正确答案是"猫"
// 模型预测 P("猫") = 0.8 → loss = -log(0.8) = 0.22 ✓ 小 loss
// 模型预测 P("猫") = 0.01 → loss = -log(0.01) = 4.6  ✗ 大 loss
```

**KL 散度（KL Divergence）= 两个分布之间的"距离"：**

```
// KL(P || Q) = 真实分布 P 和模型分布 Q 之间的差异
// 数学关系：Cross-Entropy = Entropy + KL Divergence
// 因为 Entropy 是常数，所以最小化 Cross-Entropy ≈ 最小化 KL Divergence

// 在 AI 中的应用：
// - RLHF：KL 约束防止模型偏离原始行为太远
// - DPO：直接用 KL 散度来优化偏好
// - 知识蒸馏：让学生模型的分布接近教师模型
```

**Perplexity（困惑度）：** 经常在论文中看到的指标，本质上就是 cross-entropy loss 的指数形式：`PPL = e^(cross-entropy)`。PPL 越低，模型越好。直觉：PPL = 5 意味着模型在每个位置平均在 5 个词之间"犹豫"。

**面试话术：**
> "LLM 用 cross-entropy loss 训练，本质上是在最小化模型预测分布和真实分布之间的 KL 散度。这个 loss 有个直观解释：它等于 -log(模型给正确答案的概率)，所以模型越确信正确答案，loss 越低。在 RLHF 中，KL 散度还被用作正则项，防止 reward hacking——确保微调后的模型不会偏离 base model 太远。"

---

## 本章小结

| 数学概念 | 在 AI 中的角色 | 一句话直觉 |
|----------|---------------|-----------|
| 向量 / 矩阵 | 数据表示 + 计算 | Embedding 是向量，神经网络是矩阵变换 |
| 点积 | 相似度计算 | Attention score = Q 和 K 的点积 |
| softmax | 原始分数 → 概率 | LLM 输出层必用，temperature 控制分布形状 |
| 梯度下降 | 模型学习机制 | 沿着"让 loss 减小最快"的方向调参数 |
| Backpropagation | 高效算梯度 | 链式法则逐层传递误差 |
| Loss function | 优化目标 | 衡量模型有多差，训练就是最小化它 |
| Adam optimizer | 参数更新策略 | 自适应学习率 + 动量，现代训练标配 |
| Cross-Entropy | LLM 训练 loss | = -log(正确答案的概率) |
| KL 散度 | 分布差异度量 | RLHF 中防止模型偏离太远 |

> **给前端转 AI 的建议：** 这些数学不需要你从零推导。你需要的是**直觉**——当别人说"这个 loss 下降了"，你知道意味着模型在变好；当别人说"这两个 embedding 的 cosine similarity 很高"，你知道意味着它们语义相似。有了这些直觉，后续章节的 Transformer、训练、推理都会更容易理解。

---

[下一章：01 - LLM 基础概念 →](./01-llm-fundamentals.md)
