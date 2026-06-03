# AI 应用开发面试指南

> 面向前端工程师转 AI 方向的系统化知识体系。
> 按 AI 学科的历史演进分 7 层，从数学基础到工程落地，18 章 160+ 题。

---

## AI 知识体系七层模型

```
年代        层级                          你需要的程度
─────────────────────────────────────────────────────────────

永恒        Layer 0  数学基础              了解直觉
            │  线性代数 / 概率统计 / 微积分 / 最优化 / 信息论
            │
1950-2012   Layer 1  机器学习              了解概念
            │  监督学习 / 无监督 / 强化学习 / 经典算法 / 评估指标
            │
2012-2017   Layer 2  深度学习              了解演进
            │  神经网络 / CNN / RNN-LSTM / 训练技巧 / Word2Vec→Attention
            │
2017-2022   Layer 3  大模型基座            理解原理  ★
            │  Transformer / Self-Attention / GPT vs BERT / Scaling Law
            │
2022-2024   Layer 4  大模型能力            掌握使用  ★★
            │  微调 LoRA / 对齐 RLHF-DPO / 推理优化 / 多模态
            │
2023-2026   Layer 5  AI 应用架构           深度掌握  ★★★
            │  RAG / Agent / MCP / Prompt Engineering / Embedding
            │
2024-2026   Layer 6  AI 工程化             熟练实践  ★★★
            │  工程化 / 生产部署 / 安全合规 / AI 编程工具
            │
─────────────────────────────────────────────────────────────
                                           ↑ 前端转 AI 的重心在这里
```

**一句话理解每一层：**

| 层 | 一句话 | 类比 |
|----|--------|------|
| L0 数学基础 | AI 的"物理定律" | CSS 底层的盒模型规范 |
| L1 机器学习 | AI 的"jQuery 时代" | 手动操作 DOM |
| L2 深度学习 | AI 的"React 诞生" | 自动化特征提取 |
| L3 大模型基座 | AI 的"V8 引擎" | Transformer 驱动一切 |
| L4 大模型能力 | AI 的"框架生态" | LoRA/RLHF 像 Next.js/Nuxt |
| L5 应用架构 | AI 的"全栈开发" | RAG+Agent = 前端+后端 |
| L6 工程化 | AI 的"DevOps" | 部署/监控/安全 |

---

## 完整知识树

### Layer 0：数学基础 — AI 背后的数学直觉

> 了解级别：不用推导公式，知道"为什么需要"和"直觉是什么"

```
00 数学基础 ⭐                                      → 00-math-foundations.md
│
├── 0.1 线性代数
│   └── 向量 / 矩阵乘法 / 点积 → Embedding 是向量，Attention 是矩阵运算
├── 0.2 概率与统计
│   └── 概率分布 / softmax / Bayes → LLM 输出是概率分布
├── 0.3 微积分与梯度
│   └── 梯度下降 / 反向传播 → 模型怎么"学习"
├── 0.4 最优化
│   └── 损失函数 / SGD / Adam → 训练的核心循环
└── 0.5 信息论
    └── 熵 / 交叉熵 / KL 散度 → Loss 函数的数学来源
```

### Layer 1：机器学习 — AI 的前世

> 了解级别：知道经典方法的直觉，理解为什么演进到深度学习

```
A1 机器学习基础 ⭐                                   → A1-machine-learning.md
│
├── A1.1 三大学习范式（监督 / 无监督 / 强化学习）
├── A1.2 经典算法速览（回归 / SVM / 决策树 / K-means）
├── A1.3 核心概念（过拟合 / 训练集-验证集-测试集 / 正则化）
├── A1.4 特征工程 → 为什么 LLM 时代不再需要手动特征
├── A1.5 评估指标（Precision / Recall / F1 → RAG 评估也用这些）
└── A1.6 ML vs DL vs LLM：三代 AI 的范式区别
```

### Layer 2：深度学习 — 从神经网络到 Transformer 之前

> 了解级别：知道 CNN/RNN 的局限性，理解为什么 Transformer 是革命

```
A2 深度学习基础 ⭐⭐                                  → A2-deep-learning.md
│
├── A2.1 神经网络基础（神经元 / 激活函数 / 前向传播 / 反向传播）
├── A2.2 CNN 卷积神经网络 → 为什么 Vision 模型用 CNN
├── A2.3 RNN / LSTM → 为什么处理不了长序列（Transformer 的动机）
├── A2.4 训练技巧（BatchNorm / Dropout / 学习率调度）
├── A2.5 Word2Vec → ELMo → Attention → Transformer 进化链
└── A2.6 PyTorch vs TensorFlow → 为什么 PyTorch 成为主流
```

### Layer 3：大模型基座 — Transformer 与 LLM 原理

> 理解级别：能解释 Self-Attention、知道主流模型架构演进

```
02 Transformer 架构 ⭐⭐⭐⭐                          → 02-transformer.md
│
├── 2.1 整体架构（Encoder-Decoder、为什么取代 RNN）
├── 2.2 Self-Attention 机制（Q/K/V、缩放点积）
├── 2.3 Multi-Head Attention（多头并行、不同语义模式）
├── 2.4 MQA / GQA（KV Cache 优化、LLaMA/Gemini 在用）
├── 2.5 位置编码（Sinusoidal / RoPE / ALiBi）
├── 2.6 FFN 与残差连接（SwiGLU、MoE 替代）
├── 2.7 BERT vs GPT（Encoder-Only vs Decoder-Only）
├── 2.8 架构演进（GPT-1 → ChatGPT → MoE → Reasoning Model）
└── 2.9 SSM 与 Mamba（O(n) 复杂度、混合架构趋势）

01 LLM 基础概念 ⭐⭐                                 → 01-llm-fundamentals.md
│
├── 1.1 什么是 LLM（vs 传统 NLP、涌现能力）
├── 1.2 Token 与 Tokenizer（BPE、计费影响）
├── 1.3 生成参数（Temperature / Top-P / Top-K）
├── 1.4 上下文窗口 Context Window
├── 1.5 Next Token Prediction 工作原理
├── 1.6 主流模型对比（GPT / Claude / Gemini / DeepSeek / Qwen）
├── 1.7 LLM 六大局限性（幻觉 / 知识截止 / 安全风险...）
├── 1.8 LLM API 调用基础
└── 1.9 灾难性遗忘 Catastrophic Forgetting
```

### Layer 4：大模型能力 — 训练、对齐、加速、多模态

> 掌握级别：能讲清 LoRA 原理、知道推理优化方法、了解多模态架构

```
08 模型训练与微调 ⭐⭐⭐⭐                            → 08-model-training.md
│
├── 8.1 微调基础（Full Fine-tuning vs PEFT）
├── 8.2 LoRA / QLoRA（低秩分解、4-bit 量化训练）
├── 8.3 对齐技术（RLHF 三阶段 / DPO / GRPO）
├── 8.4 训练数据工程
├── 8.5 训练优化（DeepSpeed ZeRO / FSDP）
├── 8.6 TRL v1.0 统一训练框架
├── 8.7 微调 vs RAG vs Prompt Engineering 选型
└── 8.8 评估与迭代

09 推理优化 ⭐⭐⭐⭐⭐                                → 09-inference-optimization.md
│
├── 9.1  推理 vs 训练（Memory-bound vs Compute-bound）
├── 9.2  KV Cache 原理与内存计算
├── 9.3  模型量化（INT8 / INT4 / FP8 / AWQ / GPTQ）
├── 9.4  推理框架（vLLM / SGLang / TensorRT-LLM / Ollama）
├── 9.5  PagedAttention（虚拟内存思想解决碎片化）
├── 9.6  Continuous Batching
├── 9.7  Speculative Decoding（投机解码）
├── 9.8  PD 分离（Prefill-Decode Separation）
├── 9.9  框架选型决策树
└── 9.10 DeepSeek-V3 优化实践（MoE + MLA + FP8）

10 多模态 AI ⭐⭐⭐⭐                                 → 10-multimodal.md
│
├── 10.1 多模态基础
├── 10.2 CLIP（对比学习、图文对齐）
├── 10.3 Vision-Language 模型（GPT-4o / Qwen-VL / Gemini）
├── 10.4 多模态 RAG（跨模态检索、ColPali）
├── 10.5 视频理解 Agent
├── 10.6 GUI Agent / Computer Use
├── 10.7 多模态应用实战（OCR / 图表分析）
└── 10.8 端侧多模态部署
```

### Layer 5：AI 应用架构 — RAG + Agent + 工具

> 深度掌握：这是 AI 应用开发面试的核心考点

```
03 Prompt Engineering ⭐⭐                            → 03-prompt-engineering.md
│
├── 3.1 Zero-shot / Few-shot
├── 3.2 Chain of Thought (CoT) 及变体
├── 3.3 System Prompt 设计（CRISPE 框架）
├── 3.4 结构化输出（JSON Mode / Function Calling）
├── 3.5 Prompt 模板管理
├── 3.6 Prompt 安全（注入攻击与防御）
├── 3.7 高级技巧
├── 3.8 Prompt 调试与评估
└── 3.9 Context Engineering（上下文工程）

04 Embedding 与向量检索 ⭐⭐⭐                        → 04-embedding-and-vector.md
│
├── 4.1 什么是 Embedding
├── 4.2 模型选型（BGE / text-embedding-3 / GTE / jina）
├── 4.3 向量相似度（余弦 / 欧几里得 / 点积）
├── 4.4 向量数据库（Milvus / Qdrant / Pinecone / pgvector）
├── 4.5 索引算法（HNSW / IVF / PQ / LSH）
├── 4.6 Embedding 微调
├── 4.7 多模态 Embedding
└── 4.8 选型决策树

05 RAG 系统 ⭐⭐⭐⭐（25 题，最大章节）                → 05-rag.md
│
├── 5.1  RAG 基础（为什么需要、完整流水线）
├── 5.2  分块策略（固定 / 语义 / Late Chunking / Context Cliff）
├── 5.3  检索优化（混合检索 + RRF / HyDE / Multi-Query / Query 改写）
├── 5.4  Rerank（BiEncoder 粗召回 → CrossEncoder 精排）
├── 5.5  上下文处理（Lost in the Middle / LLMLingua 压缩）
├── 5.6  评估体系（RAGAS / DeepEval / LLM-as-Judge）
├── 5.7  进阶范式（GraphRAG / Agentic RAG / Self-RAG / CRAG）
├── 5.8  生产问题（幻觉防护 / Chunk 冲突 / 权限隔离 / 自愈 RAG / 语义缓存）
├── 5.9  RAG vs SFT 选型
└── 5.10 2026 趋势（Memory-Augmented AI / Retrieval-free）

06 AI Agent ⭐⭐⭐⭐（15 题）                          → 06-ai-agent.md
│
├── 6.1  Agent 核心概念（LLM / Tools / Memory / Planning）
├── 6.2  ReAct 模式
├── 6.3  Function Calling / Tool Use
├── 6.4  规划策略（Plan-and-Solve / REWOO / Dynamic Replanning）
├── 6.5  反思机制（Reflexion / LATS / Generator-Evaluator）
├── 6.6  记忆架构（Flat Vector / Episodic / Graph / Hybrid）
├── 6.7  Multi-Agent（AutoGen / CrewAI / A2A 协议）
├── 6.8  Copilot vs Agent
├── 6.9  Chain vs Loop 架构
├── 6.10 LangGraph 工作流
├── 6.11 可观测性
├── 6.12 Voyager 终身学习
├── 6.13 防死循环与容错（五层防护）
├── 6.14 工具调用失败处理（降级链）
└── 6.15 Human-in-the-Loop（四级风险框架）

07 MCP 协议 ⭐⭐⭐                                    → 07-mcp.md
│
├── 7.1 MCP 基础（"AI 的 USB 接口"）
├── 7.2 架构（Client / Server / Transport）
├── 7.3 MCP vs Function Calling
├── 7.4 MCP Server 开发
├── 7.5 安全与权限
└── 7.6 2026 生态
```

### Layer 6：AI 工程化 — 上线、运维、安全、工具

> 熟练级别：能设计系统架构、做成本优化、保证安全合规

```
11 AI 应用工程化 ⭐⭐⭐                               → 11-ai-engineering.md
│
├── 11.1  流式输出（SSE vs WebSocket）
├── 11.2  API 重试（指数退避 + Jitter + 降级）
├── 11.3  LangChain 与 LCEL
├── 11.4  LangGraph 状态机
├── 11.5  低代码平台（Dify / Coze / n8n）
├── 11.6  Python asyncio
├── 11.7  Pydantic v2
├── 11.8  FastAPI + SSE
├── 11.9  AI 应用测试
└── 11.10 NL2SQL

12 生产部署与运维 ⭐⭐⭐⭐                             → 12-production.md
│
├── 12.1 部署策略（云 API / 自部署 / 混合）
├── 12.2 成本优化（模型路由 / 语义缓存 / 压缩 / 批处理）
├── 12.3 可观测性（LangSmith / Phoenix / OTEL）
├── 12.4 Agent SLA 设计
├── 12.5 系统设计：百万 DAU AI 客服
├── 12.6 系统设计：企业 RAG 平台
├── 12.7 系统设计：LLM API Gateway
└── 12.8 系统设计：AI 内容审核

13 AI 安全与合规 ⭐⭐⭐⭐                              → 13-ai-safety.md
│
├── 13.1 内容安全（四层防护）
├── 13.2 Prompt 注入与防御
├── 13.3 数据隐私（EU AI Act / GDPR）
├── 13.4 可防御 RAG
├── 13.5 输出安全（幻觉 / 偏见 / 毒性）
└── 13.6 红队测试

14 AI 编程工具 ⭐⭐⭐                                 → 14-ai-coding-tools.md
│
├── 14.1 2026 工具生态
├── 14.2 Claude Code vs Cursor vs Copilot
├── 14.3 自主 Coding Agent
├── 14.4 代码智能核心技术（FIM / CWM）
└── 14.5 最佳实践

15 前端转 AI 实战 ⭐⭐⭐                               → 15-frontend-to-ai.md
│
├── 15.1 前端技能迁移地图
├── 15.2 概念对照表（30+ 前端→AI 映射）
├── 15.3 学习路线（3 个月 / 6 个月）
├── 15.4 实战项目（Chat UI / RAG / Agent Dashboard）
├── 15.5 简历改造
├── 15.6 面试高频 Q&A
├── 15.7 项目经验：如何介绍 RAG/Agent 项目（STAR 法则）
├── 15.8 冷启动：没有 AI 项目经验怎么办
└── 15.9 AI 适用场景判断

16 大厂 AI 面试真题集 ⭐⭐⭐⭐                         → 16-big-tech-questions.md
│
├── 16.1 字节跳动（Self-Attention / RAG 设计 / ReAct Agent）
├── 16.2 阿里巴巴（客服 Agent / A2A vs MCP）
├── 16.3 腾讯（RAG 项目经验 / 向量库选型 / 成本控制）
├── 16.4 美团（LLM 六大问题）
├── 16.5 百度（重复生成解决）
└── 16.6 高频真题 Top 10 汇总
```

---

## 章节导航

| 层级          | 编号  | 章节                 | 题数  | 难度    | 文件                                                             |
| ----------- | --- | ------------------ | --- | ----- | -------------------------------------------------------------- |
| **L0 数学**   | 00  | 数学基础               | 5   | ⭐     | [00-math-foundations.md](./00-math-foundations.md)             |
| **L1 机器学习** | A1  | 机器学习基础             | 6   | ⭐     | [A1-machine-learning.md](./A1-machine-learning.md)             |
| **L2 深度学习** | A2  | 深度学习基础             | 6   | ⭐⭐    | [A2-deep-learning.md](./A2-deep-learning.md)                   |
| **L3 大模型**  | 01  | LLM 基础概念           | 9   | ⭐⭐    | [01-llm-fundamentals.md](./01-llm-fundamentals.md)             |
| **L3 大模型**  | 02  | Transformer 架构     | 9   | ⭐⭐⭐⭐  | [02-transformer.md](./02-transformer.md)                       |
| **L4 模型能力** | 08  | 模型训练与微调            | 11  | ⭐⭐⭐⭐  | [08-model-training.md](./08-model-training.md)                 |
| **L4 模型能力** | 09  | 推理优化               | 10  | ⭐⭐⭐⭐⭐ | [09-inference-optimization.md](./09-inference-optimization.md) |
| **L4 模型能力** | 10  | 多模态 AI             | 8   | ⭐⭐⭐⭐  | [10-multimodal.md](./10-multimodal.md)                         |
| **L5 应用架构** | 03  | Prompt Engineering | 9   | ⭐⭐    | [03-prompt-engineering.md](./03-prompt-engineering.md)         |
| **L5 应用架构** | 04  | Embedding 与向量检索    | 8   | ⭐⭐⭐   | [04-embedding-and-vector.md](./04-embedding-and-vector.md)     |
| **L5 应用架构** | 05  | RAG 系统             | 25  | ⭐⭐⭐⭐  | [05-rag.md](./05-rag.md)                                       |
| **L5 应用架构** | 06  | AI Agent           | 15  | ⭐⭐⭐⭐  | [06-ai-agent.md](./06-ai-agent.md)                             |
| **L5 应用架构** | 07  | MCP 协议             | 6   | ⭐⭐⭐   | [07-mcp.md](./07-mcp.md)                                       |
| **L6 工程化**  | 11  | AI 应用工程化           | 10  | ⭐⭐⭐   | [11-ai-engineering.md](./11-ai-engineering.md)                 |
| **L6 工程化**  | 12  | 生产部署与运维            | 8   | ⭐⭐⭐⭐  | [12-production.md](./12-production.md)                         |
| **L6 工程化**  | 13  | AI 安全与合规           | 6   | ⭐⭐⭐⭐  | [13-ai-safety.md](./13-ai-safety.md)                           |
| **L6 工程化**  | 14  | AI 编程工具            | 5   | ⭐⭐⭐   | [14-ai-coding-tools.md](./14-ai-coding-tools.md)               |
| **L6 工程化**  | 15  | 前端转 AI 实战          | 9   | ⭐⭐⭐   | [15-frontend-to-ai.md](./15-frontend-to-ai.md)                 |
| **实战**      | 16  | 大厂面试真题集           | 10  | ⭐⭐⭐⭐  | [16-big-tech-questions.md](./16-big-tech-questions.md)          |

---

## 学习路线

### 路线 A：零基础入门（3-4 周）

```
00 数学直觉 → A1 机器学习概念 → A2 深度学习概念
→ 01 LLM 基础 → 03 Prompt → 05 RAG(5.1-5.5) → 06 Agent(6.1-6.3)
→ 15 前端转 AI
```

### 路线 B：面试突击（1-2 周，跳过 L0-L2）

```
01 LLM 基础 → 05 RAG 全部 → 06 Agent 全部 → 07 MCP
→ 03 Prompt → 12 生产部署 → 15 面试 Q&A
```

### 路线 C：深度理解（6-8 周，全栈）

```
00 → A1 → A2 → 02 Transformer → 08 训练 → 09 推理
→ 04 Embedding → 05 RAG → 06 Agent → 10 多模态
→ 11 工程化 → 12 部署 → 13 安全
```

### 路线 D：前端转型专线（4 周）

```
15 转型(先看技能地图) → 01 基础 → 03 Prompt → 11 工程化(SSE/FastAPI)
→ 05 RAG → 06 Agent → 07 MCP → 14 编程工具
```

---

## 面试评分参考

```
能讲清 L3 + L5 部分内容    → 初级 AI 应用开发
能讲清 L3 + L5 + L6        → 中级 AI 应用开发（多数岗位要求）
能讲清 L3 + L4 + L5 + L6   → 高级 AI 应用开发
L0-L6 全通                  → AI 架构师
```

---

*19 章 | 180+ 题 | 16,000+ 行 | 按 AI 知识体系七层模型组织*

---

## 当前学习者状态（给新会话的 Claude 看）

**学习者背景：** 前端开发者，目标是 AI 应用开发岗位面试

**三阶段学习路径：**

```
阶段 1（当前）：问答扫盲点
  方式：Claude 出题 → 用户作答 → 答错/不会的展开讲并补充到对应 md 文件
  节奏：每次 10 道题，答对直接过，不展开
  避免重复：出题前必须读 quiz-history.md
  目标：用户自觉答对率满意为止

阶段 2（下一步）：动手做项目
  用 Claude API + 向量数据库做一个 RAG 问答应用
  目的：踩真实的坑，项目本身也是面试素材

阶段 3：用户纯输出，Claude 纠错
  用户讲"整条链路是怎么运作的"，Claude 纠错补充
  有阶段 2 实际经验支撑，输出质量更高
```

**已考过的题目：** 见 `quiz-history.md`

**出题规则：**
- 阶段 1、2、3 只出基础概念题，不出 16 章实战题
- 实战题（16 章）在阶段 3 完成后再切换，即基础牢固 + 项目实战 + 系统框架输出完成之后
- 实战题需要串联多个知识点 + 项目经验支撑，过早做会变成背答案而非真正理解

**注意：** 跳过阶段 2 直接到阶段 3，框架会"正确但空洞"，面试追问细节容易露馅。
