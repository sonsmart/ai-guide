# 出题记录

出题前先查这个文件，避免重复。

## 已考过的题目

### Transformer / Attention
- Transformer 是框架还是架构？架构/框架/模型三者区别
- Self-Attention 原理，Q/K/V 分别代表什么
- Q/K/V 是同一输入算出来的吗，怎么计算
- 每个 token 都有自己的 Q/K/V 吗，同时是提问者和被查询者
- Attention 计算结果：新向量 = Σ(Score × V)
- Transformer 内部 Q/K/V vs RAG 检索 Q/K/V 的区别
- Self-Attention 和自回归生成的关系，Self 的含义，vs Cross-Attention
- Multi-Head Attention：为什么需要多个头，计算量会增加吗
- FFN 的作用，为什么光有 Attention 不够
- 位置编码：为什么需要，不加会怎样
- BERT vs GPT：是框架还是模型，和 Transformer 是什么关系
- Decoder-Only 为什么成为主流（至少三个原因）
- Mamba/SSM 是什么

### 推理 / KV Cache
- KV Cache 是什么，如何加速推理，代价是什么
- KV Cache 只用于推理还是训练也用，为什么
- Prefill 和 Decode 阶段各做什么
- 每层 Transformer 的 K/V 一样吗，KV Cache 存哪几层
- 新 token 是怎么生成的（词表矩阵 × 向量 → 概率分布 → 采样）

### 训练
- 训练和推理最本质的区别
- 训练完整流程：前向传播 → Loss → 反向传播 → 梯度下降

### LLM 基础
- 深度学习 / 神经网络 / Transformer / LLM 的关系
- AI 知识体系七层分层
- Token 是什么，三种切法的对比（按字/词/子词）
- Temperature 是什么，调高调低的效果，写代码应该调高还是调低
- Context Window 是什么，对 AI 应用开发的影响
- LLM 的四大先天缺陷（知识过时、无私有数据、幻觉、Context 有限）

### RAG
- RAG 解决了 LLM 的哪四个缺陷
- RAG 两个阶段：Indexing（离线）和 Query（在线）各做什么
- RAG vs 微调选型（内部文档场景）

### Prompt Engineering
- Zero-shot / One-shot / Few-shot 区别，各适用什么场景
- Few-shot 和直接描述规则的本质区别
- Chain of Thought 是什么，为什么有效

### Agent
- AI Agent 四大核心组件（LLM / Planning / Memory / Tools）
- 普通 LLM 调用 vs AI Agent 最本质的区别
- ReAct 是什么，和 Thought→Action→Observation 的关系

### MCP
- MCP 是什么，解决了什么问题，没有 MCP 之前是什么状况

### Embedding
- Embedding 是什么，多模态 Embedding 解决了什么问题

### NLP 进化链
- Word2Vec vs ELMo 核心区别
- Attention(2015) 在进化链中的位置

### RAG 进阶
- Lost in the Middle 是什么，解决方法有哪些（5种）

### 推理参数
- 灾难性遗忘是什么，对应用开发的影响
- 余弦相似度 vs 欧几里得距离的区别

### Prompt 安全
- Prompt 注入攻击是什么，举例

### 工具调用
- Function Calling 是什么，和 MCP 的区别

### System Prompt / Prompt 设计
- System Prompt vs 用户 prompt 的区别

### RAG 进阶（续）
- 混合检索是什么，为什么不只用向量检索
- HyDE 是什么，解决了什么问题

### 微调
- LoRA 是什么，为什么比全量微调更常用

### MCP 架构
- MCP 三层架构：Client / Server / Transport 各负责什么

### 新一轮
- Prompt 注入攻击是什么，举例
- 灾难性遗忘是什么，微调时怎么触发
- Top-P 是什么，和 Temperature 的区别
- GraphRAG 解决了传统 RAG 的什么问题
- Multi-Agent 是什么，什么场景需要用

### 推理优化
- Speculative Decoding 是什么，解决什么瓶颈

### 微调
- 灾难性遗忘是什么，怎么触发怎么避免

### RAG 进阶（续2）
- Agentic RAG vs 普通 RAG

### Prompt Engineering 进阶
- Context Engineering vs Prompt Engineering

### Agent 深度
- Reflexion 反思机制是什么，解决 ReAct 的什么问题

### 本轮新题
- Speculative Decoding 是什么，解决什么瓶颈
- Agentic RAG vs 普通 RAG 本质区别
- RLHF 三阶段各解决什么问题
- LLM 六大局限性
- 向量数据库选型维度

### 本轮新题2
- Context Engineering vs Prompt Engineering
- Reflexion 反思机制，解决 ReAct 什么问题
- RAGAS 四大核心指标
- Copilot vs Agent 范式区别
- 灾难性遗忘怎么触发怎么避免

### 本轮新题3
- RAGAS 四大核心指标
- MCP Tools vs Resources vs Prompts 区别
- 模型量化 INT8 vs INT4 取舍
- Multi-Query Retrieval 是什么
- Agent 四种记忆类型

### 本轮新题4
- RAG 分块策略：固定分块、语义分块、Late Chunking 的区别
- Rerank：BiEncoder 粗召回和 CrossEncoder 精排的区别
- PagedAttention 如何解决 KV Cache 内存碎片化
- Agent 生产环境如何防止死循环
- AI 流式输出场景下 SSE 和 WebSocket 如何选择

## 出题规则

1. 每次出题前检查上面列表，不出重复题
2. 出完题后把新题追加到对应分类里
