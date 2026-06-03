# A2: 深度学习基础 — 从神经网络到 Transformer 之前

> **定位**：了解级别 | 理解 Transformer 之前的技术演进
> **目标**：面试中能说清神经网络原理、CNN/RNN 的局限、以及为什么 Transformer 出现

---

## 知识树

```
深度学习基础（了解级别）
├── A2.1 神经网络基础
│   └── 神经元、激活函数(ReLU/Sigmoid)、前向传播、反向传播
├── A2.2 CNN 卷积神经网络
│   └── 卷积核、特征图、池化 — 为什么 Vision 模型用 CNN
├── A2.3 RNN / LSTM / GRU
│   └── 序列建模、记忆机制 — Transformer 之前怎么处理文本
├── A2.4 训练技巧
│   └── BatchNorm / Dropout / 学习率调度 / 数据增强
├── A2.5 从 Word2Vec 到 Transformer
│   └── Word2Vec → Attention(2014) → Transformer(2017) → ELMo(2018) 的进化链
└── A2.6 深度学习框架
    └── PyTorch vs TensorFlow — 为什么 PyTorch 成为主流
```

---

## A2.1 神经网络基础

### Q: 神经网络的核心组件是什么？

| 组件 | 作用 | 前端类比 |
|------|------|---------|
| **神经元 Neuron** | 加权求和 + 激活函数 | 一个函数：`output = activate(w1*x1 + w2*x2 + b)` |
| **层 Layer** | 多个神经元并排 | 一层中间件处理 |
| **深度网络** | 多层堆叠 | 深度嵌套的函数组合 `f(g(h(x)))` |
| **权重 Weights** | 连接强度，可学习参数 | 可配置的系数 |
| **偏置 Bias** | 基础偏移量 | 函数的默认值 |

### Q: 激活函数为什么重要？

没有激活函数，多层线性变换 = 一层线性变换（线性组合的组合还是线性的）。激活函数引入**非线性**，让网络能拟合复杂模式。

| 激活函数 | 公式直觉 | 特点 |
|---------|---------|------|
| **Sigmoid** | 压缩到 (0,1) | 早期常用，深层容易梯度消失 |
| **ReLU** | 负数变 0，正数不变 | 现在最常用，计算快，缓解梯度消失 |
| **GELU** | ReLU 的平滑版本 | Transformer 中常用 |

### Q: 前向传播和反向传播怎么理解？

```
前向传播 Forward：输入 → 逐层计算 → 输出预测值 → 算损失
反向传播 Backward：损失 → 逐层算梯度 → 更新权重
```

- **前向传播** ≈ 函数调用链执行一遍
- **反向传播** ≈ 从最终结果往回追溯，用链式法则算每个参数的"贡献度"（梯度）

> **面试话术**：
> "神经网络本质上是一个参数化的函数逼近器。前向传播算预测，反向传播算梯度来更新参数。深度的意义在于 — 每层提取不同抽象层次的特征。"

---

## A2.2 CNN 卷积神经网络

### Q: CNN 的核心思想是什么？

CNN 的三个关键操作：

| 操作 | 作用 | 直觉 |
|------|------|------|
| **卷积 Convolution** | 用小窗口（卷积核）在输入上滑动，提取局部特征 | 像一个放大镜逐区域扫描 |
| **特征图 Feature Map** | 卷积的输出，表示某种特征在各位置的响应 | 边缘检测器的热力图 |
| **池化 Pooling** | 下采样，减小尺寸，保留关键信息 | 缩略图 — 降低分辨率但保留主要内容 |

### Q: 为什么 CNN 适合处理图像？

1. **局部性** — 图像的特征是局部的（边缘、纹理），卷积核天然捕捉局部模式
2. **参数共享** — 同一个卷积核在整张图上滑动，比全连接层参数少几个数量级
3. **平移不变性** — 不管猫在图片左边还是右边，同一个卷积核都能检测到

### Q: CNN 在 LLM 时代还有用吗？

有。多模态模型中的视觉编码器仍然基于 CNN 或其变体：
- **ViT**（Vision Transformer）借鉴了 CNN 思路但用 Transformer 架构
- **CLIP** 的图像编码器可选 CNN（ResNet）或 ViT
- 很多 Vision-Language 模型的图像预处理管线仍包含卷积组件

> **面试话术**：
> "CNN 的核心优势是用局部感受野和参数共享高效处理网格结构数据。虽然 ViT 证明 Transformer 也能做视觉任务，但 CNN 的归纳偏置（局部性）在数据量有限时仍有优势。"

---

## A2.3 RNN / LSTM / GRU

### Q: RNN 怎么处理序列数据？

```
RNN 处理流程：

x1 → [h1] → x2 → [h2] → x3 → [h3] → ... → 输出
       ↑            ↑            ↑
     hidden      hidden      hidden
     state       state       state
```

RNN 逐个处理序列元素，用 **隐藏状态 (hidden state)** 传递之前的信息。像 `Array.reduce()` — 每一步的累积器带着历史信息。

### Q: LSTM 解决了 RNN 的什么问题？

| 问题 | RNN | LSTM |
|------|-----|------|
| 长序列记忆 | 早期信息随序列变长而"遗忘" | 引入 cell state 长期记忆 |
| 梯度消失 | 深层梯度趋近于 0 | 门控机制保持梯度流通 |
| 选择性记忆 | 无法控制记住什么 | 遗忘门 + 输入门 + 输出门 |

LSTM 的三个"门"：
- **遗忘门 Forget Gate**：决定丢弃哪些旧信息
- **输入门 Input Gate**：决定存入哪些新信息
- **输出门 Output Gate**：决定输出哪些信息

GRU 是 LSTM 的简化版 — 把三个门合并为两个（重置门 + 更新门），参数更少，效果接近。

### Q: 为什么 Transformer 取代了 RNN/LSTM？

这是理解 Transformer 诞生动机的**关键问题**：

| 维度 | RNN/LSTM | Transformer |
|------|----------|-------------|
| **并行化** | 必须串行处理（t 依赖 t-1） | 可以并行处理所有位置 |
| **长距离依赖** | 距离越远，信息衰减越严重 | Attention 直接连接任意两个位置 |
| **训练效率** | GPU 利用率低（串行瓶颈） | 高度并行，充分利用 GPU |
| **规模化** | 难以扩展到超大模型 | Scaling law — 越大越强 |

> **面试话术**：
> "RNN 的根本限制是串行处理 — 无法并行化意味着无法利用现代 GPU 的算力，也无法扩展到大规模。Transformer 用 self-attention 让每个位置直接看到所有其他位置，既解决了长距离依赖，又实现了完全并行。这就是为什么 'Attention Is All You Need'。"

---

## A2.4 训练技巧

### Q: 深度学习常用的训练技巧有哪些？

| 技巧 | 解决什么问题 | 一句话原理 |
|------|------------|-----------|
| **BatchNorm** | 训练不稳定，收敛慢 | 每层的输入做标准化，稳定分布 |
| **Dropout** | 过拟合 | 训练时随机"关掉"部分神经元，强制冗余学习 |
| **学习率调度** | 学习率固定效果差 | Warmup 慢启动 + Cosine Decay 逐渐降低 |
| **数据增强** | 训练数据不够 | 对图像做翻转/裁剪/颜色变换，增加多样性 |

### Q: 学习率调度的 Warmup + Decay 策略？

```
学习率
  ↑
  |    /‾‾‾‾\
  |   /      \
  |  /        \___________
  | /
  +---------------------------→ 训练步数
    warmup   peak    decay
```

- **Warmup**：开始时用小学习率，逐步增大 — 避免初期梯度不稳定
- **Decay**：达到峰值后逐步降低 — 避免后期震荡，精细调整

这个策略在 LLM 训练和 fine-tuning 中**仍然是标配**。

### Q: Dropout 和前端的关系？

Dropout ≈ 混沌工程。随机关掉部分节点，强迫网络学会冗余表示 — 就像随机关掉微服务节点测试系统容错能力。LLM 推理时通常不用 Dropout，但训练和 fine-tuning 时仍然使用。

---

## A2.5 从 Word2Vec 到 Transformer

### Q: NLP 领域的进化链是怎样的？

这条进化链是理解 Transformer 起源的关键：

| 阶段 | 模型 | 核心突破 | 局限 |
|------|------|---------|------|
| **1. 词袋/TF-IDF** | — | 统计词频 | 完全丢失语序和语义 |
| **2. Word2Vec (2013)** | CBOW/Skip-gram | 词变成向量，语义关系可计算 | 一个词只有一个固定向量（"bank"不分银行/河岸） |
| **3. Attention (2014)** | Bahdanau Attention | 让解码器动态聚焦编码器不同位置 | 仍依赖 RNN，无法完全并行 |
| **4. Transformer (2017)** | Self-Attention | 完全并行 + 长距离依赖 | 计算量随序列长度平方增长 |
| **5. ELMo (2018)** | 双向 LSTM | 上下文相关的词向量 | 仍是 RNN 架构，串行瓶颈 |
| **5. BERT (2018)** | Transformer Encoder | 双向理解上下文 | 适合理解任务，不擅长生成 |
| **6. GPT 系列 (2018→)** | Transformer Decoder | 强大的文本生成能力 | 自回归生成，推理慢 |

### Q: Word2Vec 的核心思想？

"Tell me your friends, and I'll tell you who you are." — 一个词的含义由它周围的词决定。

Word2Vec 训练目标：给定一个词，预测周围的词（Skip-gram），或给定周围的词预测中心词（CBOW）。训练完成后，语义相近的词在向量空间中距离近。

经典例子：`king - man + woman ≈ queen`

### Q: 从 Word2Vec 到 Transformer 的关键跳跃是什么？

```
Word2Vec: 静态 embedding（一个词一个向量，与上下文无关）
    ↓ 问题：多义词怎么办？
Attention (2014, Bahdanau): 让解码器动态聚焦源序列不同位置，解决翻译中对齐问题
    ↓ 仍依赖 RNN，串行
Transformer (2017): Self-Attention 完全替代 RNN，并行处理整个序列
    ↓ 基于 Transformer 进一步探索上下文表示
ELMo (2018): 用双向 LSTM 生成上下文相关向量，但仍是 RNN
```

> **面试话术**：
> "NLP 的进化围绕两个问题：怎么更好地表示语义，怎么更高效地处理序列。Word2Vec 解决了词向量表示，ELMo 加入了上下文，Transformer 用 Attention 机制彻底解决了并行化和长距离依赖，开启了大模型时代。"

---

## A2.6 深度学习框架

### Q: PyTorch 和 TensorFlow 的核心区别？

| 维度 | PyTorch | TensorFlow |
|------|---------|------------|
| **执行方式** | Eager（即时执行） | 早期 Graph 模式（先定义后执行） |
| **调试体验** | 像写 Python 一样直接调试 | 早期需要 Session，调试困难 |
| **前端类比** | 像 JavaScript — 解释执行，灵活 | 像编译型语言 — 先编译再运行 |
| **学术研究** | 绝对主流（>80% 论文） | 份额持续下降 |
| **生产部署** | TorchServe, ONNX | TF Serving, TF Lite（仍有优势） |
| **LLM 生态** | Hugging Face 默认 PyTorch | 少数模型支持 TF |

### Q: 为什么 PyTorch 赢了？

1. **Eager execution** — 即时执行让调试和实验循环更快，研究者更喜欢
2. **Pythonic API** — 更符合 Python 开发者的直觉
3. **Hugging Face 生态** — transformers 库默认 PyTorch，形成正循环
4. **Meta 的投入** — 持续维护和优化

> TensorFlow 2.x 也引入了 eager execution，但为时已晚 — PyTorch 已经占领了学术界和 LLM 社区。

### Q: 前端工程师需要会 PyTorch 吗？

看角色定位：

| 角色 | PyTorch 需求 |
|------|-------------|
| AI 应用开发（API 调用） | 不需要 |
| RAG / Agent 开发 | 了解基本概念即可 |
| Fine-tuning 工程师 | 需要基础 PyTorch + Hugging Face |
| AI Infra / 模型部署 | 需要熟悉 PyTorch 生态 |

> **面试话术**：
> "PyTorch 胜出的原因和 React 类似 — 更好的开发者体验和更活跃的社区形成了飞轮效应。对于 AI 应用开发，Hugging Face 的 transformers 库是最重要的工具，它抽象了底层框架细节。"

---

## 本章小结

| 知识点 | 面试一句话 |
|--------|----------|
| 神经网络 | 参数化的函数逼近器，反向传播更新权重 |
| CNN | 局部感受野 + 参数共享，适合网格数据（图像） |
| RNN/LSTM | 串行处理序列，长距离依赖差，无法并行 — Transformer 的动机 |
| 训练技巧 | BatchNorm 稳定训练，Dropout 防过拟合，Warmup+Decay 调学习率 |
| NLP 进化链 | Word2Vec → ELMo → Attention → Transformer |
| 框架选择 | PyTorch 主流（eager + Hugging Face 生态），前端重点掌握 API 层 |

---

> **导航**
> ← [A1: 机器学习基础](./A1-machine-learning.md) | 下一章：A3 →
