# 05 - RAG 系统

> **难度：** ⭐⭐⭐⭐⭐ | **定位：** AI 应用开发的核心架构，面试重中之重
>
> **前端类比：** RAG 之于 LLM，就像 SSR（Server-Side Rendering）之于前端——不依赖客户端（模型）自身的缓存（训练知识），而是在每次请求时从服务器（外部知识库）实时获取数据，拼装成最终页面（回答）。Chunk 就像 Code Splitting，Rerank 就像搜索结果排序。

## 本章知识树

```
RAG 系统
├── 5.1 RAG 基础（什么是 RAG、为什么需要、完整流程）
├── 5.2 文档处理与分块策略
│   ├── 固定分块 vs 语义分块 vs Late Chunking
│   ├── Chunk 大小选择
│   └── Context Cliff 现象
├── 5.3 检索优化
│   ├── 混合检索（向量 + BM25 + RRF 融合）
│   ├── Multi-Query Retrieval
│   ├── HyDE（假设性文档嵌入）
│   └── Query 改写与扩展
├── 5.4 Rerank 重排序（BiEncoder vs CrossEncoder）
├── 5.5 上下文处理
│   ├── Lost in the Middle 问题
│   └── LLMLingua 上下文压缩
├── 5.6 RAG 评估体系
│   ├── RAGAS 四大指标
│   ├── DeepEval（CI/CD 集成）
│   └── LLM-as-a-Judge
├── 5.7 RAG 进阶范式
│   ├── GraphRAG（知识图谱增强）
│   ├── Agentic RAG（Agent 驱动的多轮检索）
│   ├── Self-RAG / CRAG（自检索/纠错型）
│   └── RAG-Fusion
├── 5.8 RAG 生产问题
│   ├── 幻觉防护（三层防护体系）
│   ├── Chunk 冲突检测与解决
│   ├── 权限隔离（四层架构）
│   ├── 自愈 RAG
│   └── 语义缓存
├── 5.9 RAG vs SFT 选型
└── 5.10 2026 RAG 新趋势
```

---

## 5.1 RAG 基础

### Q: 什么是 RAG？为什么 LLM 需要 RAG？

**RAG（Retrieval-Augmented Generation）= 检索增强生成，让 LLM 在回答前先"查资料"，而不是全靠记忆。**

```
传统 LLM（纯记忆）：
  用户问 → LLM 从训练知识中"回忆" → 生成回答
  问题：知识过时、幻觉、无法访问私有数据

RAG（检索增强）：
  用户问 → 检索相关文档 → 把文档塞进 Prompt → LLM 基于文档生成回答
  优点：知识实时、可溯源、可控、成本低
```

**为什么需要 RAG？LLM 有四大先天缺陷：**

| 缺陷 | 描述 | RAG 如何解决 |
|------|------|-------------|
| **知识过时** | 训练数据有截止日期 | 实时检索最新文档 |
| **幻觉** | 自信地编造不存在的信息 | 基于真实文档生成，可溯源 |
| **无私有数据** | 无法访问企业内部数据 | 检索企业知识库 |
| **Context 有限** | 无法一次吃下所有文档 | 只检索最相关的片段 |

**前端类比：** LLM 纯记忆回答 = CSR（Client-Side Rendering），数据全在客户端缓存里，可能过期。RAG = SSR + 数据获取，每次请求都从数据库（知识库）拉最新数据渲染。

### Q: 描述 RAG 的完整 Pipeline？

**RAG Pipeline 分为两大阶段：Indexing（离线索引）和 Query（在线查询）。**

```
=== Indexing 阶段（离线，一次性） ===

  原始文档（PDF/HTML/Markdown/DB）
       │
       ▼
  ┌─────────────┐
  │ 文档加载     │  ← Document Loaders
  │ (Loading)    │    (LangChain/LlamaIndex)
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │ 文档分块     │  ← Chunking Strategy
  │ (Chunking)  │    (固定/语义/递归)
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │ 向量化       │  ← Embedding Model
  │ (Embedding) │    (OpenAI/Cohere/BGE)
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │ 存入向量数据库│  ← Vector Store
  │ (Indexing)   │    (Pinecone/Weaviate/Milvus)
  └─────────────┘

=== Query 阶段（在线，每次请求） ===

  用户问题
       │
       ▼
  ┌─────────────┐
  │ Query 处理   │  ← Query Rewriting / Expansion
  │ (改写/扩展)  │
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │ 检索         │  ← Hybrid Search (向量 + BM25)
  │ (Retrieval)  │    返回 Top-K 文档片段
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │ 重排序       │  ← Reranker (CrossEncoder)
  │ (Reranking)  │    精排 Top-K → Top-N
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │ 上下文组装   │  ← Prompt Template
  │ (Assembly)   │    System + Context + Question
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │ LLM 生成     │  ← GPT-4 / Claude / Gemini
  │ (Generation) │
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │ 后处理       │  ← Citation / Hallucination Check
  │ (Post)       │
  └─────────────┘
```

**核心代码示例（LangChain 风格）：**

```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# === Indexing ===
# 1. 加载文档
loader = PyPDFLoader("company_handbook.pdf")
docs = loader.load()

# 2. 分块
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    separators=["\n\n", "\n", "。", ".", " "]
)
chunks = splitter.split_documents(docs)

# 3. 向量化 + 存储
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(chunks, embeddings)

# === Query ===
# 4. 构建检索链
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
llm = ChatOpenAI(model="gpt-4o", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# 5. 查询
result = qa_chain.invoke({"query": "公司年假政策是什么？"})
print(result["result"])
print(result["source_documents"])  # 溯源
```

**面试话术：**
> "RAG 的核心价值是让 LLM 从'闭卷考试'变成'开卷考试'。完整 Pipeline 分 Indexing 和 Query 两个阶段：Indexing 阶段做文档加载、分块、向量化、存储；Query 阶段做查询改写、检索、重排、组装 Prompt、生成。每个环节都有优化空间，比如分块策略影响召回质量，Rerank 提升精度，Query 改写解决语义鸿沟。"

---

## 5.2 文档处理与分块策略

### Q: 常见的分块策略有哪些？各有什么优缺点？

**分块（Chunking）是 RAG 中最关键的预处理步骤——切得好不好直接决定检索质量。**

**三大主流策略：**

| 策略 | 原理 | 优点 | 缺点 | 适用场景 |
|------|------|------|------|----------|
| **固定分块** | 按字符/token 数切割 | 简单、可控 | 可能切断语义 | 结构化文档 |
| **语义分块** | 按语义相似度断句 | 保持语义完整 | 计算量大、不稳定 | 非结构化长文本 |
| **Late Chunking** | 先整体 Embedding 再切块 | 保留全局上下文 | 需要特殊模型支持 | 需要全局理解的文档 |

**1. 固定分块（Fixed-size Chunking）：**

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,       # 每个 chunk 最大 512 字符
    chunk_overlap=64,     # 相邻 chunk 重叠 64 字符（防止切断语义）
    separators=["\n\n", "\n", "。", ".", " ", ""]  # 优先在段落/句子边界切
)

# RecursiveCharacterTextSplitter 的递归逻辑：
# 1. 先尝试按 "\n\n"（段落）切
# 2. 如果单段太长，按 "\n"（换行）切
# 3. 还长就按 "。"（句号）切
# 4. 最后按字符切（兜底）
```

**前端类比：** 固定分块就像 `webpack` 的 `splitChunks.maxSize`——按大小切分代码，可能把一个完整的函数拆到两个 bundle 里。`chunk_overlap` 就像 `webpack` 的 `minSize` 和 overlap 策略，保证边界不丢失关键信息。

**2. 语义分块（Semantic Chunking）：**

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import OpenAIEmbeddings

# 用 Embedding 模型判断句子之间的语义相似度
# 当相似度下降超过阈值时，视为"语义断裂"，在此处切分
semantic_splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",  # 百分位阈值
    breakpoint_threshold_amount=95           # 相似度低于 95% 分位时切
)

chunks = semantic_splitter.split_text(long_document)
```

**语义分块的工作原理：**

```
句子序列：[S1, S2, S3, S4, S5, S6, S7, S8]

计算相邻句子的 Embedding 余弦相似度：
  sim(S1,S2) = 0.92  ─┐
  sim(S2,S3) = 0.88   │ 高相似度 → 同一 chunk
  sim(S3,S4) = 0.91  ─┘
  sim(S4,S5) = 0.43  ← 语义断裂！在此切分
  sim(S5,S6) = 0.87  ─┐
  sim(S6,S7) = 0.85   │ 同一 chunk
  sim(S7,S8) = 0.89  ─┘

结果：Chunk1 = [S1,S2,S3,S4]  Chunk2 = [S5,S6,S7,S8]
```

**3. Late Chunking（2024 新方法）：**

```
传统方法：先切块 → 再对每个块做 Embedding
  问题：每个 chunk 丢失了全局上下文
  例如："他在2023年获得诺贝尔奖" — 如果"他"的指代在上一个 chunk，这个 chunk 就丢了关键信息

Late Chunking：先对整个文档做 Embedding（获取全局上下文注意力）→ 再切块
  优点：每个 chunk 的 Embedding 都包含了全局语境信息
  前提：需要支持长上下文的 Embedding 模型（如 jina-embeddings-v2, 8192 tokens）
```

```python
# Late Chunking 概念伪代码
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("jinaai/jina-embeddings-v2-base-en")
tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v2-base-en")

# 1. 先对整篇文档做 Embedding，获取每个 token 的上下文向量
inputs = tokenizer(full_document, return_tensors="pt", max_length=8192)
outputs = model(**inputs)
token_embeddings = outputs.last_hidden_state  # [1, seq_len, hidden_dim]

# 2. 按预定义的 chunk 边界，对 token embeddings 做 mean pooling
chunk_boundaries = [(0, 128), (128, 256), (256, 384)]  # token 级边界
chunk_embeddings = []
for start, end in chunk_boundaries:
    chunk_emb = token_embeddings[0, start:end, :].mean(dim=0)
    chunk_embeddings.append(chunk_emb)
# 每个 chunk embedding 都 "看过" 了全文的上下文
```

### Q: Chunk 大小怎么选？什么是 Context Cliff？

**Chunk 大小是 RAG 系统最重要的超参数之一，直接影响检索精度和生成质量。**

**Chunk 大小的权衡：**

```
Chunk 太小（< 128 tokens）：
  ✅ 检索精度高（每个 chunk 语义聚焦）
  ❌ 缺乏上下文（"他获得了诺贝尔奖" — 谁？）
  ❌ 需要更多 chunks 才能覆盖完整信息

Chunk 太大（> 2048 tokens）：
  ✅ 上下文完整
  ❌ 检索精度低（噪声太多，向量被"稀释"）
  ❌ 超出 Embedding 模型的有效窗口 → Context Cliff
```

**Context Cliff（上下文悬崖）现象：**

```
                检索质量
                  │
            1.0   │     ●●●●●●●
                  │   ●           ●
            0.8   │ ●               ●
                  │●                  ●
            0.6   │                     ●
                  │                       ●
            0.4   │                         ●●●●●  ← 质量断崖式下降！
                  │
            0.2   │
                  │
            0.0   └───────────────────────────────→ Chunk 大小 (tokens)
                  128  256  512  1024  2048  4096
                                    ↑
                              ~2500 tokens
                          Context Cliff 临界点
```

**Context Cliff 的本质原因：**

大多数 Embedding 模型（如 `text-embedding-3-small`）虽然声称支持 8192 tokens，但其 **有效注意力窗口** 通常在 512-2500 tokens。超过这个范围后：
- Embedding 向量被过多内容"平均化"，区分度下降
- 位置编码外推能力有限，远距离 token 影响减弱
- 检索相关性急剧下降

**推荐 Chunk 大小：**

| 场景 | 推荐大小 | Overlap |
|------|----------|---------|
| QA 问答 | 256-512 tokens | 10-15% |
| 文档摘要 | 1024-2048 tokens | 5-10% |
| 代码检索 | 按函数/类切分 | 0（自然边界） |
| 法律/合同 | 512-1024 tokens | 15-20% |
| 通用场景 | 512 tokens | 64 tokens |

**实验验证方法：**

```python
import numpy as np
from ragas import evaluate
from ragas.metrics import context_precision, context_recall

# 在不同 chunk size 下评估 RAG 质量
chunk_sizes = [128, 256, 512, 1024, 2048, 4096]
results = {}

for size in chunk_sizes:
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=size//8)
    chunks = splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(chunks, embeddings)

    # 用 RAGAS 评估
    score = evaluate(
        dataset=test_dataset,
        metrics=[context_precision, context_recall],
    )
    results[size] = score
    print(f"Chunk size {size}: Precision={score['context_precision']:.3f}, "
          f"Recall={score['context_recall']:.3f}")

# 典型结果：
# Chunk size 128:  Precision=0.92, Recall=0.61  ← 精度高但召回低
# Chunk size 256:  Precision=0.89, Recall=0.74
# Chunk size 512:  Precision=0.85, Recall=0.83  ← 最佳平衡点
# Chunk size 1024: Precision=0.78, Recall=0.86
# Chunk size 2048: Precision=0.65, Recall=0.82
# Chunk size 4096: Precision=0.41, Recall=0.73  ← Context Cliff！
```

**面试话术：**
> "分块策略我推荐 RecursiveCharacterTextSplitter 作为 baseline，512 tokens + 10% overlap。核心要注意 Context Cliff 现象——超过约 2500 tokens 后，Embedding 模型的有效注意力窗口耗尽，检索质量会断崖式下降。实际项目中我会用 RAGAS 做 A/B 测试，找到当前数据集的最佳 chunk size。语义分块虽然理论更好，但计算成本高且不够稳定，生产中我更倾向于固定分块 + 好的分隔符策略。"

---

## 5.3 检索优化

### Q: 什么是混合检索？为什么要结合向量检索和 BM25？

**混合检索（Hybrid Search）= 向量检索 + 关键词检索，取长补短。**

> 常见误解：向量检索的召回率/准确率比关键词检索"低"——实际上两者各有擅长场景，不是谁更好，而是互补。向量检索胜在语义理解，关键词检索胜在精确匹配专有名词、代码、ID 等。

```
向量检索（Semantic Search）：
  ✅ 理解语义（"JS 框架" 能匹配 "React 是一个 JavaScript 库"）
  ❌ 对精确关键词不敏感（搜 "Error code 4012" 可能匹配不到）
  ❌ 对生僻词/专有名词效果差

关键词检索（BM25）：
  ✅ 精确匹配关键词（搜 "Error 4012" 必中）
  ✅ 对专有名词/ID/代码 效果好
  ❌ 不理解语义（"JS 框架" 匹配不到 "React"）

混合检索 = 两者结合：
  ✅ 语义理解 + 精确匹配
  ✅ 互补覆盖更全面
  关键问题：如何融合两路结果的分数？→ RRF
```

**RRF（Reciprocal Rank Fusion）融合公式：**

```
RRF_score(doc) = Σ  1 / (k + rank_i(doc))

其中：
  k = 常数（通常为 60）
  rank_i(doc) = 文档在第 i 个检索器中的排名（从 1 开始）

示例：
  文档 A：向量检索排名 #1，BM25 排名 #5
  RRF(A) = 1/(60+1) + 1/(60+5) = 0.0164 + 0.0154 = 0.0318

  文档 B：向量检索排名 #3，BM25 排名 #2
  RRF(B) = 1/(60+3) + 1/(60+2) = 0.0159 + 0.0161 = 0.0320

  → 文档 B 排更前（两路都排名靠前的文档得分更高）
```

**RRF 的优势：**
- 不需要归一化（不同检索器的分数尺度不同，直接加权不公平）
- 只依赖排名，不依赖绝对分数
- 对 outlier 不敏感（k=60 平滑了排名差异）

**混合检索完整实现：**

```python
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import BM25Retriever
from langchain.vectorstores import Chroma

# 1. 向量检索器
vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings())
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# 2. BM25 关键词检索器
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 20

# 3. RRF 融合
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.6, 0.4]  # 向量权重稍高（语义理解更重要）
)

# 4. 检索
results = ensemble_retriever.invoke("React 性能优化有哪些方法？")
```

**手写 RRF 实现（面试可能要求）：**

```python
def reciprocal_rank_fusion(
    search_results_list: list[list[dict]],
    k: int = 60
) -> list[dict]:
    """
    search_results_list: 多路检索结果，每路是 [{doc_id, score}, ...]
    k: RRF 常数，默认 60
    返回：按 RRF 分数排序的文档列表
    """
    rrf_scores = {}
    doc_map = {}

    for results in search_results_list:
        for rank, item in enumerate(results, start=1):
            doc_id = item["doc_id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank)
            doc_map[doc_id] = item

    # 按 RRF 分数降序排列
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [{"doc_id": doc_id, "rrf_score": score, **doc_map[doc_id]}
            for doc_id, score in sorted_docs]
```

### Q: 什么是 HyDE？它如何解决 Query 和文档之间的语义鸿沟？

**HyDE（Hypothetical Document Embeddings）= 让 LLM 先生成一个"假设性答案"，用这个答案的 Embedding 去检索，而不是直接用用户问题去检索。**

**核心思路：**

```
传统检索的问题：
  用户问题（Query）和文档（Document）在语义空间中可能距离很远

  例如：
    Query: "如何减少 React 重渲染？"
    Document: "使用 React.memo 包裹组件可以避免不必要的 props 变化导致的渲染..."

    问题是 "问题的 Embedding" 和 "答案的 Embedding" 可能不在同一个语义区域
    → 检索时可能找不到最相关的文档

HyDE 的解法：
  1. 让 LLM 先根据 Query 生成一个假设性答案（可能不准确，但语义方向对）
  2. 用这个假设性答案的 Embedding 去检索
  3. 假设性答案和真实文档在语义空间中更接近 → 检索更准

  Query → LLM生成假设答案 → Embedding → 检索 → 真实文档 → LLM再生成
```

**HyDE 代码实现：**

```python
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
embeddings = OpenAIEmbeddings()

def hyde_retrieval(query: str, vectorstore, k: int = 5):
    """HyDE 检索：先生成假设性文档，再用其 Embedding 检索"""

    # Step 1: 生成假设性文档
    hyde_prompt = ChatPromptTemplate.from_template(
        "请写一段话来回答以下问题。不需要完全准确，"
        "但要包含相关的技术术语和概念。\n\n问题：{query}"
    )
    hypothetical_doc = llm.invoke(hyde_prompt.format(query=query)).content

    # Step 2: 用假设性文档的 Embedding 检索
    # （假设性答案和真实文档语义更接近）
    hypo_embedding = embeddings.embed_query(hypothetical_doc)
    results = vectorstore.similarity_search_by_vector(hypo_embedding, k=k)

    return results, hypothetical_doc

# 使用示例
query = "前端微服务怎么做样式隔离？"
results, hypo_doc = hyde_retrieval(query, vectorstore)

# 对比：
# 直接检索：query embedding 可能偏向"问题"语义空间
# HyDE 检索：假设性答案 "CSS Module/Shadow DOM/CSS-in-JS 可以实现样式隔离..."
#            → 更接近真实文档的语义空间
```

**HyDE 的局限：**

| 优点 | 局限 |
|------|------|
| 缩小 Query-Document 语义鸿沟 | 额外一次 LLM 调用（增加延迟和成本） |
| 对复杂/模糊问题提升明显 | 如果 LLM 生成的假设文档方向错误，检索更差 |
| 简单易实现 | 对简单/精确查询（如搜索错误码）可能反而有害 |

### Q: 什么是 Multi-Query Retrieval？它如何提升召回率？

**Multi-Query = 从不同角度改写用户问题，用多个变体分别检索，合并结果。**

```
用户原始问题："React 如何做状态管理？"

Multi-Query 改写为 3-5 个变体：
  1. "React 中有哪些常用的状态管理方案？"
  2. "Redux、MobX、Zustand 各有什么特点？"
  3. "React 组件间如何共享状态？"
  4. "前端全局状态管理的最佳实践是什么？"

每个变体分别检索 Top-K → 合并去重 → 更全面的召回
```

```python
from langchain.retrievers import MultiQueryRetriever
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    llm=llm
)

# 内部会：
# 1. 用 LLM 生成 3 个查询变体
# 2. 每个变体检索 Top-5
# 3. 合并去重，返回唯一文档列表
results = multi_query_retriever.invoke("React 如何做状态管理？")
```

### Q: Query 改写与扩展还有哪些技巧？

**Query 优化是 RAG 中性价比最高的优化手段之一。**

| 技巧 | 做法 | 适用场景 |
|------|------|----------|
| **Query Rewriting** | LLM 重写用户问题，使其更清晰 | 用户问题模糊/口语化 |
| **Step-back Prompting** | 先问一个更抽象的问题 | 需要背景知识的问题 |
| **Query Decomposition** | 将复杂问题拆成子问题 | 多跳推理问题 |
| **Query Expansion** | 添加同义词/相关术语 | 提升召回率 |

```python
# Step-back Prompting 示例
def step_back_query(query: str) -> str:
    """生成一个更抽象的 step-back 问题"""
    prompt = f"""给定以下具体问题，生成一个更通用、更基础的问题，
    这个基础问题的答案有助于回答原始问题。

    具体问题：{query}
    Step-back 问题："""
    return llm.invoke(prompt).content

# 示例：
# 原始问题："React 18 的 useTransition 和 useDeferredValue 有什么区别？"
# Step-back："React 18 的并发渲染机制是什么？"
# → 先检索并发渲染的基础知识，再结合具体 hook 的文档回答
```

**上下文语义重写（多轮对话场景）：**

多轮对话中，用户的后续问题往往省略了上下文，直接拿去检索效果很差：

```python
# 对话历史
history = [
    ("什么是 RAG？", "RAG 是检索增强生成..."),
    ("它有什么优势？", "RAG 的三大优势是..."),
]

# 用户新问题（缺少上下文）
user_query = "怎么实现它？"
# → 直接检索 "怎么实现它" 根本找不到东西

# 上下文语义重写：补全省略的指代
def contextual_rewrite(query: str, history: list) -> str:
    prompt = f"""对话历史：{history}
    用户新问题：{query}
    请将新问题改写为一个独立、完整的问题（补全省略的指代和上下文）："""
    return llm.invoke(prompt).content

rewritten = contextual_rewrite("怎么实现它？", history)
# → "如何实现 RAG 检索增强生成系统？"
# → 这个 query 检索效果好多了
```

**面试话术：**
> "检索优化我会采用三层策略：第一层是混合检索（向量+BM25+RRF），第二层是 Query 优化（Multi-Query 改写、HyDE、多轮对话上下文语义重写），第三层是 Rerank 重排序。特别是多轮对话场景，用户经常说'它怎么样'这种省略了主语的问题，必须用对话历史补全后再检索。"

---

## 5.4 Rerank 重排序

### Q: 什么是 Rerank？BiEncoder 和 CrossEncoder 有什么区别？

**Rerank = 对初步检索的结果做二次精排，提升最终返回文档的质量。**

> 常见误解：Rerank 只是"按向量相似度重新排序"——实际上向量检索已经做过相似度排序了，Rerank 做的是更精确的事：用 CrossEncoder 把"问题+chunk 实际内容"拼在一起整体理解，重新计算真实相关性。比向量检索的近似相似度准确得多，代价是更慢（无法预计算）。所以两阶段：向量检索粗召回一批候选，CrossEncoder 精排选 top-k。

**两阶段的计算方式对比：**

```
向量检索（BiEncoder）：
  问题 → 问题向量
  每个 chunk 全文 → chunk 向量（离线预计算好的）
  → 问题向量 vs 每个 chunk 向量 计算余弦相似度
  → 一次性并行算完所有 chunk，极快

CrossEncoder（Rerank）：
  [问题全文 + chunk全文] 拼在一起 → 过一遍模型 → 相关性分数
  → 每个 chunk 都要单独过一遍模型（一个一个来）
  → 无法预计算，所以慢
  → 但问题和 chunk 充分交互，打分更准确
```

"全文"和"一个一个"都是对的。

**核心思路：粗召回 + 精排（两阶段检索）**

```
阶段一：BiEncoder 粗召回（Retrieval）
  ┌─────────┐    ┌─────────┐
  │  Query   │    │  Doc_i  │
  │ Encoder  │    │ Encoder │    ← 独立编码，互不影响
  └────┬─────┘    └────┬────┘
       │               │
       ▼               ▼
   query_emb       doc_emb       ← 各自生成独立的向量
       │               │
       └───── cos ─────┘          ← 计算余弦相似度
             相似度

  特点：速度快（向量预计算 + ANN 检索），但精度有限
  从百万文档中快速找到 Top-100

阶段二：CrossEncoder 精排（Reranking）
  ┌──────────────────────────┐
  │  [CLS] Query [SEP] Doc  │    ← Query 和 Doc 拼接在一起
  │     CrossEncoder         │       做全交互注意力
  └──────────┬───────────────┘
             │
             ▼
        相关性分数（0-1）

  特点：精度高（Query 和 Doc 充分交互），但速度慢（无法预计算）
  对 Top-100 精排，选出 Top-5
```

**CrossEncoder 的计算过程（和模型推理的关系）：**

CrossEncoder 本质是把"问题+chunk"拼成一段文本，走一遍完整的 Transformer 前向传播，取 [CLS] token 的输出向量接线性层打分：

```
输入："今天天气怎么样 [SEP] 北京今天晴天28度"
↓ embedding
↓ 第1层 Self-Attention + FFN（问题和chunk的所有token互相attention）
↓ 第2层 Self-Attention + FFN
↓ ... N层
↓ 取 [CLS] token 的最终向量
↓ × 线性层 → 相关性分数 0.87
```

[CLS] 不是输入的第一个有意义的词，而是一个人为加在最前面的特殊占位 token：

```
实际输入：[CLS] 苹果是什么颜色 [SEP] 经过研究表明苹果是红色
位置：      0     1  2  3  4  5   6    7  8  9  10 ...
```

[CLS] 本身没有语义，存在的唯一目的是：经过多层 Attention 后，它的向量会 attend 整个序列所有 token，成为全文的"汇总向量"，然后接线性层打分。是 BERT 训练时专门设计的机制——[CLS] 作为"整段文本的摘要向量"用于分类/打分，而不是输入的某个具体词。

| | 模型推理（生成答案） | CrossEncoder（打分） |
|---|---|---|
| 过程 | 前向传播 + 自回归循环生成 | 只有一次前向传播 |
| 输出 | 逐 token 生成文本 | 一个标量分数 |
| 慢在哪 | 要循环生成很多 token | 每对 (问题, chunk) 都要实时过一遍模型，无法预计算 |

向量检索比 CrossEncoder 快得多的原因：向量检索根本不走 Transformer，只是两个向量做余弦相似度（纯矩阵点积），chunk 向量早已离线预计算好。

**BiEncoder vs CrossEncoder 对比：**

| 维度 | BiEncoder | CrossEncoder |
|------|-----------|--------------|
| **编码方式** | Query 和 Doc 独立编码 | Query 和 Doc 拼接后联合编码 |
| **交互程度** | 无交互（只在最后算相似度） | 全交互（Attention 层充分交互） |
| **速度** | 极快（向量可预计算 + ANN） | 慢（每对 query-doc 都要过模型） |
| **精度** | 中等 | 高（通常高 5-15%） |
| **适用规模** | 百万~十亿文档 | 几十~几百文档 |
| **典型模型** | `text-embedding-3-small` | `cross-encoder/ms-marco-MiniLM` |
| **前端类比** | 像搜索框的 `debounce` 模糊匹配 | 像 Elasticsearch 的全文精确搜索 |

**两阶段检索代码实现：**

```python
from sentence_transformers import CrossEncoder
import cohere

# === 方法一：使用 CrossEncoder 模型（本地部署） ===
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L12-v2")

def rerank_with_cross_encoder(query: str, docs: list[str], top_n: int = 5):
    """用 CrossEncoder 对检索结果精排"""
    # 构造 query-doc pairs
    pairs = [[query, doc] for doc in docs]

    # CrossEncoder 打分
    scores = reranker.predict(pairs)

    # 按分数降序排列
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]

# === 方法二：使用 Cohere Rerank API（云服务） ===
co = cohere.Client("your-api-key")

def rerank_with_cohere(query: str, docs: list[str], top_n: int = 5):
    response = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=docs,
        top_n=top_n
    )
    return [(r.document.text, r.relevance_score) for r in response.results]

# === 完整 Pipeline ===
# 1. BiEncoder 粗召回 Top-50
raw_results = vectorstore.similarity_search(query, k=50)
docs = [r.page_content for r in raw_results]

# 2. CrossEncoder 精排 Top-5
reranked = rerank_with_cross_encoder(query, docs, top_n=5)
```

**Rerank 的效果数据（典型提升）：**

```
                    Without Rerank    With Rerank    提升
MRR@10              0.68             0.81          +19%
NDCG@10             0.62             0.76          +23%
Recall@5            0.71             0.84          +18%
Answer Accuracy     0.73             0.85          +16%
```

**面试话术：**
> "Rerank 是 RAG 中性价比最高的优化之一。原理是两阶段检索：第一阶段用 BiEncoder 做粗召回，速度快，从百万文档中捞出 Top-50 到 Top-100；第二阶段用 CrossEncoder 做精排，Query 和每个候选文档拼接后做全交互注意力，精度高但速度慢。这就像前端搜索——先用 indexOf 快速过滤，再用复杂的相关性算法精排。实际项目中 Rerank 通常能带来 15-20% 的准确率提升，推荐用 Cohere Rerank 或 BGE-Reranker。"

---

## 5.5 上下文处理

### Q: 什么是 Lost in the Middle 问题？如何解决？

**Lost in the Middle = LLM 在长上下文中，倾向于关注开头和结尾，忽略中间部分的信息。**

这是 2023 年斯坦福大学论文发现的 LLM 固有缺陷：

```
Context Window 中的信息位置 vs LLM 对该信息的利用程度：

  利用程度
  100% │ ●                                           ●
       │   ●                                       ●
   80% │     ●                                   ●
       │       ●                               ●
   60% │         ●                           ●
       │           ●       ●   ●           ●
   40% │             ●   ●       ●       ●
       │               ●           ●   ●
   20% │                             ●          ← 中间位置利用率最低！
       │
    0% └─────────────────────────────────────→ 位置
       开头                 中间                结尾
```

**解决方案：**

| 策略 | 做法 | 效果 |
|------|------|------|
| **重要信息前置** | 将最相关的 chunk 放在 context 开头 | 简单有效 |
| **交叉排列** | 按相关性高-低-高-低交替排列 | 提升中间位置利用率 |
| **分段摘要** | 对每个 chunk 先 LLM 摘要再拼装 | 减少总长度，消除位置偏差 |
| **Map-Reduce** | 每个 chunk 独立回答 → 合并答案 | 彻底消除位置效应 |
| **上下文压缩** | 用 LLMLingua 等工具压缩 | 减少 token 用量 |

```python
def reorder_for_lost_in_middle(docs: list, strategy: str = "alternating"):
    """解决 Lost in the Middle：重排文档顺序"""

    if strategy == "important_first":
        # 已经按相关性排序，直接返回（最相关的在开头）
        return docs

    elif strategy == "alternating":
        # 交叉排列：最相关的放开头和结尾，次要的放中间
        # [1, 2, 3, 4, 5] → [1, 3, 5, 4, 2]
        reordered = []
        left, right = [], []
        for i, doc in enumerate(docs):
            if i % 2 == 0:
                left.append(doc)
            else:
                right.append(doc)
        return left + right[::-1]

    elif strategy == "sandwich":
        # 三明治法：最相关的放开头和结尾
        if len(docs) <= 2:
            return docs
        return [docs[0]] + docs[2:] + [docs[1]]
```

### Q: 什么是 LLMLingua？上下文压缩有什么用？

**LLMLingua = 用小模型（如 GPT-2）识别 prompt 中的"不重要" token 并删除，实现 2-20x 压缩。**

**为什么需要上下文压缩？**

```
RAG 检索返回 5 个 chunks，每个 512 tokens = 2560 tokens context
加上 System Prompt + 用户问题 ≈ 3000 tokens
问题：
  1. 成本高（GPT-4o 的 input 按 token 计费）
  2. 速度慢（更多 token → 更高延迟）
  3. Lost in the Middle（context 越长效果越差）
```

**LLMLingua 的工作原理：**

```
原始 Prompt（100 tokens）：
  "The company policy states that employees who have been working
   for more than two years are eligible for additional vacation days.
   The number of extra days is calculated based on years of service."

LLMLingua 压缩后（40 tokens）：
  "company policy: employees working >2 years eligible additional
   vacation days. extra days calculated based years service."

压缩率：60%，核心信息完整保留
```

```python
from llmlingua import PromptCompressor

compressor = PromptCompressor(
    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    use_llmlingua2=True
)

# 压缩检索到的上下文
compressed = compressor.compress_prompt(
    context=retrieved_context,       # 检索到的文档内容
    instruction="",                   # System Prompt（不压缩）
    question=user_query,              # 用户问题（不压缩）
    target_token=500,                 # 目标压缩到 500 tokens
    condition_in_question="after_condition",
    reorder_context="sort",           # 重排序上下文
    condition_compare=True,
    context_budget="+100"
)

# compressed["compressed_prompt"]  → 压缩后的 prompt
# compressed["origin_tokens"]      → 原始 token 数
# compressed["compressed_tokens"]  → 压缩后 token 数
# compressed["ratio"]              → 压缩率

print(f"压缩率: {compressed['ratio']:.1f}x")
# 典型输出: 压缩率: 3.2x
```

**面试话术：**
> "上下文处理有两个关键问题：一是 Lost in the Middle——LLM 对长 context 中间位置的信息利用率最低，解决方案是将最相关的文档放在开头和结尾，或者用 Map-Reduce 模式让每个 chunk 独立回答再合并。二是上下文压缩——可以用 LLMLingua 等工具把检索到的文档压缩 2-5 倍，降低成本的同时减少噪声。"

---

## 5.6 RAG 评估体系

### Q: RAGAS 框架的四大核心指标是什么？怎么理解？

**RAGAS（Retrieval-Augmented Generation Assessment）是 RAG 系统最主流的评估框架，定义了四大核心指标。**

```
RAGAS 四大指标：

  ┌─────────────────────────────────────────────────┐
  │                  RAG Pipeline                    │
  │                                                  │
  │  Question ──→ Retrieval ──→ Context ──→ Answer  │
  │     │             │           │          │       │
  │     │    ┌────────┘           │          │       │
  │     │    │                    │          │       │
  │     ▼    ▼                    ▼          ▼       │
  │  ┌──────────────┐   ┌──────────────────────┐    │
  │  │ 检索阶段评估   │   │    生成阶段评估       │    │
  │  │              │   │                      │    │
  │  │ Context      │   │ Faithfulness         │    │
  │  │ Precision    │   │ (忠实度)              │    │
  │  │ (上下文精度)  │   │                      │    │
  │  │              │   │ Answer Relevancy     │    │
  │  │ Context      │   │ (答案相关性)          │    │
  │  │ Recall       │   │                      │    │
  │  │ (上下文召回)  │   │                      │    │
  │  └──────────────┘   └──────────────────────┘    │
  └─────────────────────────────────────────────────┘
```

**四大指标详解：**

| 指标 | 衡量什么 | 计算方式 | 理想值 |
|------|----------|----------|--------|
| **Faithfulness** | 答案是否忠实于检索到的上下文？ | 答案中能被 context 支持的陈述比例 | → 1.0 |
| **Answer Relevancy** | 答案是否回答了用户的问题？ | 从答案反向生成问题，与原问题的相似度 | → 1.0 |
| **Context Precision** | 检索到的 context 中有多少是真正相关的？ | 相关 context 排在前面的程度（加权精度） | → 1.0 |
| **Context Recall** | 真正需要的信息是否被检索到了？ | Ground truth 中能被 context 覆盖的比例 | → 1.0 |

**直觉理解：**

```
Faithfulness（忠实度）：
  Context: "React 18 发布于 2022 年 3 月"
  Answer:  "React 18 发布于 2022 年 3 月，引入了并发渲染"
  → "发布于 2022 年 3 月" 能被 context 支持 ✅
  → "引入了并发渲染" 不在 context 中 ❌（可能是幻觉！）
  → Faithfulness = 1/2 = 0.5

Answer Relevancy（答案相关性）：
  Question: "React 的虚拟 DOM 是什么？"
  Answer:   "React 使用 Fiber 架构来实现高效的 UI 更新..."
  → 答案相关但没有直接回答虚拟 DOM 的概念
  → Relevancy ≈ 0.6

Context Precision（上下文精度）：
  检索到 5 个 chunks：[相关, 无关, 相关, 无关, 无关]
  → 相关的排在第 1、3 位
  → Precision = weighted_avg(1/1, 0, 2/3, 0, 0) ≈ 0.56

Context Recall（上下文召回）：
  Ground Truth 需要 3 个关键信息点
  检索到的 context 覆盖了其中 2 个
  → Recall = 2/3 ≈ 0.67
```

**RAGAS 代码使用：**

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

# 准备评估数据集
eval_data = {
    "question": [
        "React 的虚拟 DOM 是什么？",
        "Vue 3 的 Composition API 有什么优势？"
    ],
    "answer": [
        "虚拟 DOM 是真实 DOM 的轻量级 JS 对象表示...",
        "Composition API 提供了更好的逻辑复用能力..."
    ],
    "contexts": [
        ["虚拟 DOM（Virtual DOM）是 React 的核心概念之一..."],
        ["Vue 3 引入了 Composition API 作为 Options API 的替代..."]
    ],
    "ground_truth": [
        "虚拟 DOM 是一个 JavaScript 对象树，是真实 DOM 的抽象表示",
        "Composition API 的优势包括更好的逻辑复用、TypeScript 支持、和代码组织"
    ]
}

dataset = Dataset.from_dict(eval_data)

# 执行评估
result = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)

print(result)
# {'faithfulness': 0.87, 'answer_relevancy': 0.92,
#  'context_precision': 0.78, 'context_recall': 0.83}
```

### Q: DeepEval 如何与 CI/CD 集成？LLM-as-a-Judge 是什么？

**DeepEval 是一个可以直接集成到 pytest 的 RAG 评估框架，适合 CI/CD Pipeline。**

```python
# test_rag.py — 可以直接用 pytest 运行
import deepeval
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    HallucinationMetric
)

def test_rag_faithfulness():
    """测试 RAG 输出的忠实度"""
    test_case = LLMTestCase(
        input="公司年假政策是什么？",
        actual_output="员工入职满一年后可享受 10 天年假",
        retrieval_context=["公司规定：员工入职满一年后，可享受 10 天带薪年假"],
        expected_output="入职满一年后享受 10 天年假"
    )

    metric = FaithfulnessMetric(threshold=0.8)
    assert_test(test_case, [metric])

def test_rag_no_hallucination():
    """测试 RAG 输出无幻觉"""
    test_case = LLMTestCase(
        input="公司年假政策是什么？",
        actual_output="员工入职满一年后可享受 15 天年假",  # 错误！
        retrieval_context=["公司规定：员工入职满一年后，可享受 10 天带薪年假"],
    )

    metric = HallucinationMetric(threshold=0.5)
    # 这个测试应该 FAIL，因为 15 天 ≠ 10 天
    assert_test(test_case, [metric])

# CI/CD 中运行：
# pytest test_rag.py --deepeval  ← 自动生成报告
```

**LLM-as-a-Judge（用 LLM 评估 LLM）：**

```python
# 用一个强力 LLM（如 GPT-4o）评估 RAG 输出质量
def llm_as_judge(question: str, answer: str, context: str) -> dict:
    judge_prompt = f"""你是一个公正的评估者。请根据以下标准评估回答质量。

    问题：{question}
    参考上下文：{context}
    待评估回答：{answer}

    请从以下维度打分（1-5分）并说明理由：
    1. 准确性（回答是否与上下文一致）
    2. 完整性（是否覆盖了问题的所有要点）
    3. 相关性（是否直接回答了问题）
    4. 简洁性（是否简洁不冗余）

    输出 JSON 格式：
    {{"accuracy": X, "completeness": X, "relevancy": X, "conciseness": X, "reasoning": "..."}}
    """

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": judge_prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)
```

**三种评估方法对比：**

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **RAGAS** | 指标定义清晰、社区标准 | 需要 Ground Truth | 系统级评估、基线对比 |
| **DeepEval** | 集成 pytest、CI/CD 友好 | 依赖 LLM 评估（有成本） | 回归测试、上线门禁 |
| **LLM-as-a-Judge** | 灵活、可自定义维度 | LLM 评估也可能不准确 | 快速迭代、人工评估替代 |

**面试话术：**
> "RAG 评估我会用 RAGAS 四大指标体系：Faithfulness 检测幻觉、Answer Relevancy 检测答案相关性、Context Precision 和 Recall 评估检索质量。在 CI/CD 中用 DeepEval 集成到 pytest，每次上线前自动跑评估，设置 Faithfulness > 0.85 作为上线门禁。日常快速迭代则用 LLM-as-a-Judge，让 GPT-4o 从多个维度打分。"

---

## 5.7 RAG 进阶范式

### Q: 什么是 GraphRAG？它如何解决传统 RAG 的多跳推理问题？

**GraphRAG = 用知识图谱（Knowledge Graph）增强 RAG，解决传统 RAG 无法处理的多跳推理和全局理解问题。**

**传统 RAG 的局限：**

```
问题："张三和李四有什么共同的合作伙伴？"

传统 RAG 的困境：
  - Chunk 1: "张三与王五在 2023 年合作了项目 A"
  - Chunk 2: "李四与王五在 2024 年合作了项目 B"
  - 这两个 chunk 可能完全独立，向量检索可能只召回其中一个
  - 无法做跨 chunk 的关系推理

GraphRAG 的解法：
  构建知识图谱：
  张三 ──合作──→ 王五 ──合作──→ 李四

  图遍历就能回答：共同合作伙伴是王五
```

**GraphRAG 的核心架构（微软 GraphRAG 方法）：**

```
=== Indexing 阶段 ===

  原始文档
       │
       ▼
  ┌─────────────┐
  │ LLM 实体抽取 │  ← 提取实体和关系
  │ Entity       │    "张三" → Person
  │ Extraction   │    "项目 A" → Project
  └──────┬──────┘    "张三 --参与-→ 项目 A"
         │
         ▼
  ┌─────────────┐
  │ 构建知识图谱  │  ← 存入图数据库
  │ Build KG     │    (Neo4j / NetworkX)
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │ 社区检测     │  ← Leiden 算法
  │ Community    │    将图分成语义社区
  │ Detection    │
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │ 社区摘要     │  ← LLM 对每个社区生成摘要
  │ Community    │    提供全局理解能力
  │ Summaries    │
  └─────────────┘

=== Query 阶段 ===

  Local Search：从实体出发，遍历图的邻居节点
  Global Search：搜索社区摘要，获取全局概览
  → 两种模式互补
```

**GraphRAG 代码示例：**

```python
# 使用 LLM 从文本中抽取实体和关系
def extract_entities_and_relations(text: str) -> dict:
    prompt = f"""从以下文本中提取所有实体和它们之间的关系。

    文本：{text}

    输出 JSON 格式：
    {{
        "entities": [
            {{"name": "实体名", "type": "类型(Person/Org/Tech/...)", "description": "描述"}}
        ],
        "relations": [
            {{"source": "实体1", "target": "实体2", "relation": "关系类型", "description": "描述"}}
        ]
    }}
    """
    response = llm.invoke(prompt)
    return json.loads(response.content)

# 构建知识图谱
import networkx as nx

G = nx.DiGraph()

for chunk in chunks:
    result = extract_entities_and_relations(chunk.page_content)

    for entity in result["entities"]:
        G.add_node(entity["name"], type=entity["type"], desc=entity["description"])

    for rel in result["relations"]:
        G.add_edge(rel["source"], rel["target"],
                   relation=rel["relation"], desc=rel["description"])

# 检索时：图遍历 + 向量检索
def graph_enhanced_retrieval(query: str, G: nx.DiGraph, vectorstore):
    # 1. 向量检索找到起始实体
    initial_docs = vectorstore.similarity_search(query, k=3)

    # 2. 从起始实体出发，在图上做 2-hop 遍历
    relevant_entities = set()
    for doc in initial_docs:
        entities_in_doc = extract_entities_from_doc(doc)
        for entity in entities_in_doc:
            if entity in G:
                # 获取 1-hop 和 2-hop 邻居
                neighbors_1 = set(G.neighbors(entity))
                neighbors_2 = set()
                for n in neighbors_1:
                    neighbors_2.update(G.neighbors(n))
                relevant_entities.update({entity}, neighbors_1, neighbors_2)

    # 3. 收集所有相关实体的描述和关系
    context = []
    for entity in relevant_entities:
        node_data = G.nodes[entity]
        context.append(f"实体: {entity} ({node_data.get('type', '')}): {node_data.get('desc', '')}")
        for _, target, data in G.edges(entity, data=True):
            context.append(f"  关系: {entity} --{data['relation']}--> {target}")

    return "\n".join(context)
```

**GraphRAG vs 传统 RAG：**

| 维度 | 传统 RAG | GraphRAG |
|------|----------|----------|
| 数据结构 | 扁平 chunks | 图结构（实体 + 关系） |
| 多跳推理 | 不支持 | 原生支持（图遍历） |
| 全局理解 | 局部 chunk 视角 | 社区摘要提供全局视角 |
| 索引成本 | 低（Embedding） | 高（LLM 抽取实体+关系） |
| 适用场景 | 简单 QA | 复杂关系推理、知识密集型 |
| 维护成本 | 低 | 高（图更新、实体消歧） |

### Q: 什么是 Agentic RAG？它和传统 RAG 有什么区别？

**Agentic RAG = 用 Agent 的 think-act-observe 循环来驱动 RAG，实现多轮自适应检索。**

**传统 RAG 是"一次性"的：查一次 → 返回。Agentic RAG 是"多轮"的：查了不满意 → 换个角度再查 → 直到满意。**

```
=== 传统 RAG（Single-shot）===

  Question → Retrieve → Generate → Answer
  （一次检索，一次生成，不管质量好不好都输出）

=== Agentic RAG（Multi-turn loop）===

  Question
     │
     ▼
  ┌──────────┐
  │  Think    │  ← "这个问题需要什么信息？"
  └────┬─────┘
       │
       ▼
  ┌──────────┐
  │ Retrieve  │  ← 执行检索
  └────┬─────┘
       │
       ▼
  ┌──────────┐
  │ Evaluate  │  ← "检索到的信息够吗？质量好吗？"
  └────┬─────┘
       │
       ├──→ 不够好 ──→ Re-think（换 query/换数据源）──→ 回到 Retrieve
       │
       └──→ 够好 ──→ Generate Answer
```

**Agentic RAG 代码实现：**

```python
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import Tool

# 定义多个检索工具
tools = [
    Tool(
        name="search_docs",
        description="搜索内部文档库",
        func=lambda q: vectorstore.similarity_search(q, k=5)
    ),
    Tool(
        name="search_code",
        description="搜索代码仓库",
        func=lambda q: code_search(q)
    ),
    Tool(
        name="search_web",
        description="搜索互联网获取最新信息",
        func=lambda q: web_search(q)
    ),
    Tool(
        name="ask_clarification",
        description="当信息不足时，生成澄清问题",
        func=lambda q: generate_clarification(q)
    ),
]

# Agent 会自主决定：
# 1. 该用哪个工具检索
# 2. 检索结果是否足够
# 3. 是否需要换个角度再检索
# 4. 什么时候可以生成最终答案

agent = create_openai_tools_agent(
    llm=ChatOpenAI(model="gpt-4o"),
    tools=tools,
    prompt=agent_prompt  # 包含 "如果检索信息不足，请尝试其他工具或改写查询" 的指令
)

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=5,   # 最多 5 轮检索-思考循环
    verbose=True
)

# 执行
result = executor.invoke({
    "input": "对比 React Server Components 和 Astro Islands 的性能差异"
})
# Agent 可能的执行轨迹：
# Round 1: search_docs("React Server Components") → 找到一些内容
# Round 2: search_docs("Astro Islands architecture") → 找到一些内容
# Round 3: search_web("RSC vs Astro Islands performance benchmark 2025") → 补充最新数据
# Round 4: Generate final comparison answer
```

### Q: 什么是 Self-RAG？它如何让模型自主决定是否需要检索？

**Self-RAG（Self-Reflective RAG）= 训练模型生成"反思 token"，自主判断是否需要检索、检索是否相关、回答是否有支持。**

```
传统 RAG：每次都检索（可能浪费）
Self-RAG：模型自己决定

  Input Question
       │
       ▼
  ┌──────────────┐
  │ 生成 [Retrieve]│  ← 特殊 token
  │ Yes / No      │
  └──────┬───────┘
         │
    ┌────┴────┐
    │         │
  [Yes]     [No]
    │         │
    ▼         ▼
  检索文档   直接生成答案
    │
    ▼
  ┌──────────────┐
  │ 生成 [IsRel]  │  ← 判断检索结果是否相关
  │ Relevant /    │
  │ Irrelevant    │
  └──────┬───────┘
         │
    ┌────┴────┐
    │         │
 [Relevant] [Irrelevant]
    │         │
    ▼         └→ 丢弃，重新检索
  生成回答
    │
    ▼
  ┌──────────────┐
  │ 生成 [IsSup]  │  ← 判断回答是否被 context 支持
  │ Supported /   │
  │ Partial /     │
  │ No Support    │
  └──────┬───────┘
         │
         ▼
  输出最终答案 + 置信度
```

**Self-RAG 的四种反思 token：**

| Token | 含义 | 决策 |
|-------|------|------|
| `[Retrieval]`/`[No Retrieval]` | 是否需要检索 | [Retrieval] → 检索 / [No Retrieval] → 直接生成 |
| `[Relevant]`/`[Irrelevant]` | 检索结果是否相关 | [Relevant] → 使用 / [Irrelevant] → 丢弃 |
| `[Fully supported]`/`[Partially supported]`/`[No support]` | 答案是否被 context 支持 | [Fully supported] → 输出 / [No support] → 重新生成 |
| `[Utility:1~5]` | 答案对用户的有用程度（1-5分） | 分数高 → 输出 / 分数低 → 重新检索 |

> **注：** 以上为原论文（Asai et al., 2023）中使用的实际反思 token 名称。

### Q: 什么是 CRAG（Corrective RAG）？它如何纠错？

**CRAG = 对检索结果做三级质量评估，根据质量采取不同策略。**

```
CRAG 三级质量过滤：

  检索结果
       │
       ▼
  ┌──────────────┐
  │ 质量评估器     │  ← 轻量级模型评估检索质量
  │ Quality       │
  │ Evaluator     │
  └──────┬───────┘
         │
    ┌────┼────────┐
    │    │        │
 [Correct]  [Ambiguous]  [Incorrect]
    │         │            │
    ▼         ▼            ▼
  直接使用   知识精炼     丢弃检索结果
  检索结果   + Web搜索    用 Web 搜索替代
             补充
```

```python
def crag_pipeline(query: str, retrieved_docs: list) -> str:
    """CRAG: Corrective Retrieval-Augmented Generation"""

    # Step 1: 评估检索质量
    quality = evaluate_retrieval_quality(query, retrieved_docs)
    # quality ∈ {"correct", "ambiguous", "incorrect"}

    if quality == "correct":
        # 检索结果质量好，直接使用
        context = "\n".join([doc.page_content for doc in retrieved_docs])

    elif quality == "ambiguous":
        # 检索结果模棱两可，精炼 + Web 搜索补充
        refined = knowledge_refinement(retrieved_docs)  # 提取关键信息
        web_results = web_search(query)                  # Web 搜索补充
        context = refined + "\n" + web_results

    elif quality == "incorrect":
        # 检索结果不相关，完全依赖 Web 搜索
        web_results = web_search(query)
        context = web_results

    # Step 2: 生成回答
    answer = llm.invoke(f"根据以下信息回答问题：\n{context}\n\n问题：{query}")
    return answer

def evaluate_retrieval_quality(query: str, docs: list) -> str:
    """用 LLM 评估检索结果与 query 的相关性"""
    prompt = f"""评估以下检索结果与问题的相关性。

    问题：{query}
    检索结果：{[doc.page_content[:200] for doc in docs]}

    判断结果：
    - "correct": 检索结果直接回答了问题
    - "ambiguous": 检索结果部分相关，需要补充
    - "incorrect": 检索结果完全不相关

    只输出一个词：correct/ambiguous/incorrect
    """
    return llm.invoke(prompt).content.strip().lower()
```

### Q: 什么是 RAG-Fusion？

**RAG-Fusion = Multi-Query + RRF 融合，从多个角度检索并融合排名。**

```python
def rag_fusion(query: str, vectorstore, llm, k: int = 5) -> list:
    """RAG-Fusion: 多角度检索 + RRF 融合"""

    # Step 1: 生成多个查询变体
    variant_prompt = f"""生成 4 个不同角度的搜索查询来回答以下问题。
    每个查询应该从不同的角度出发，使用不同的关键词。

    原始问题：{query}

    输出 4 个查询，每行一个："""

    variants = llm.invoke(variant_prompt).content.strip().split("\n")
    all_queries = [query] + variants  # 原始 query + 4 个变体

    # Step 2: 每个查询独立检索
    all_results = []
    for q in all_queries:
        results = vectorstore.similarity_search_with_score(q, k=k)
        all_results.append(results)

    # Step 3: RRF 融合
    rrf_scores = {}
    doc_map = {}

    for results in all_results:
        for rank, (doc, score) in enumerate(results, start=1):
            doc_id = doc.page_content[:100]  # 用前 100 字符作为 ID
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (60 + rank)
            doc_map[doc_id] = doc

    # Step 4: 按 RRF 分数排序
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[doc_id] for doc_id, _ in sorted_docs[:k]]
```

**进阶范式对比总结：**

| 范式 | 核心思想 | 解决的问题 | 复杂度 | 适用场景 |
|------|----------|-----------|--------|----------|
| **传统 RAG** | 单次检索+生成 | 基础知识问答 | 低 | 简单 QA |
| **GraphRAG** | 知识图谱+图遍历 | 多跳推理、关系推理 | 高 | 复杂关系分析 |
| **Agentic RAG** | Agent 多轮检索 | 复杂问题需多步检索 | 中高 | 开放域复杂问答 |
| **Self-RAG** | 模型自主反思 | 不必要的检索、低质量回答 | 高（需训练） | 需要高精度的场景 |
| **CRAG** | 检索质量纠错 | 检索结果不相关 | 中 | 噪声多的知识库 |
| **RAG-Fusion** | 多角度检索+RRF | 单一查询召回不全 | 低 | 提升召回率 |

**面试话术：**
> "在进阶 RAG 范式中，我最看好 Agentic RAG 和 GraphRAG。Agentic RAG 用 Agent 的 think-retrieve-evaluate 循环实现自适应多轮检索，解决了传统 RAG '一锤子买卖' 的问题。GraphRAG 通过构建知识图谱支持多跳推理，解决了传统 RAG 无法做跨 chunk 关系推理的痛点。Self-RAG 很优雅但需要专门训练模型，生产中我倾向于用 CRAG 的思路做检索质量校验——成本低，效果好。"

---

## 5.8 RAG 生产问题

### Q: 幻觉是怎么产生的？RAG 能完全解决吗？

**幻觉产生的根本原因：** LLM 本质是预测下一个 token 的概率模型，当问题超出训练数据的知识边界时，模型会"合理地编造"一个听起来正确的答案，而不是说"我不知道"。

**RAG 不能完全解决幻觉，原因有三：**

1. **问题超出检索文档的知识边界** — 检索到的文档里根本没有答案，模型仍可能编造
2. **模型忽略检索内容** — 模型可能无视文档内容，用自己的"记忆"回答
3. **对文档内容做错误推理** — 文档内容是对的，但模型理解偏了

**其他缓解幻觉的方法：**
- 置信度标注：训练模型在不确定时说"我不知道"
- 引用溯源：强制引用原文，无法引用就不回答
- Self-RAG：模型自己判断是否需要检索，检索后再判断文档是否相关
- 微调对齐：训练模型在知识边界外主动拒绝回答

### Q: RAG 系统如何做幻觉防护？

**幻觉防护三层体系：Prompt 层 → 检测层 → 兜底层。**

```
三层防护体系：

  ┌─────────────────────────────────────────┐
  │ Layer 1: Prompt 层（预防）               │
  │ - System Prompt 中明确要求基于 context    │
  │ - 要求引用来源                           │
  │ - 禁止推测和编造                          │
  └──────────────┬──────────────────────────┘
                 │
  ┌──────────────▼──────────────────────────┐
  │ Layer 2: Detection 层（检测）            │
  │ - NLI 模型检测答案与 context 的蕴含关系   │
  │ - Faithfulness 分数低于阈值 → 拦截       │
  │ - 关键实体/数字交叉校验                   │
  └──────────────┬──────────────────────────┘
                 │
  ┌──────────────▼──────────────────────────┐
  │ Layer 3: Fallback 层（兜底）             │
  │ - 检测到幻觉时，返回 "我不确定" + 原始来源 │
  │ - 引导用户查看原始文档                     │
  │ - 记录失败案例供优化                       │
  └─────────────────────────────────────────┘
```

**完整实现：**

```python
from transformers import pipeline

# Layer 1: Prompt 层
ANTI_HALLUCINATION_PROMPT = """你是一个严谨的问答助手。请严格遵守以下规则：

1. 只基于【参考资料】中的信息回答问题
2. 如果参考资料中没有相关信息，明确说"根据现有资料，我无法回答这个问题"
3. 回答中必须标注信息来源，格式为 [来源: 文档名称]
4. 不要推测、猜测或编造任何信息
5. 数字和日期必须与参考资料完全一致

【参考资料】
{context}

【问题】
{question}
"""

# Layer 2: Detection 层
class HallucinationDetector:
    def __init__(self):
        # NLI（自然语言推断）模型
        self.nli_model = pipeline(
            "text-classification",
            model="cross-encoder/nli-deberta-v3-base"
        )

    def check_faithfulness(self, answer: str, context: str) -> dict:
        """检查答案是否忠实于上下文"""
        # 将答案拆分为独立的陈述
        statements = self._split_to_statements(answer)

        results = []
        for statement in statements:
            # NLI 判断：context 是否蕴含（entail）这个陈述
            nli_result = self.nli_model(f"{context} [SEP] {statement}")
            label = nli_result[0]["label"]  # entailment / contradiction / neutral
            score = nli_result[0]["score"]

            results.append({
                "statement": statement,
                "label": label,
                "score": score,
                "is_hallucination": label != "entailment"
            })

        # 计算整体 Faithfulness
        faithful_count = sum(1 for r in results if not r["is_hallucination"])
        faithfulness = faithful_count / len(results) if results else 0

        return {
            "faithfulness": faithfulness,
            "details": results,
            "has_hallucination": faithfulness < 0.8
        }

    def _split_to_statements(self, text: str) -> list[str]:
        """将文本拆分为独立陈述"""
        prompt = f"将以下文本拆分为独立的事实陈述，每行一个：\n{text}"
        response = llm.invoke(prompt)
        return [s.strip() for s in response.content.split("\n") if s.strip()]

# Layer 3: Fallback 层
class RAGWithHallucinationGuard:
    def __init__(self):
        self.detector = HallucinationDetector()

    def answer(self, question: str, context: str) -> dict:
        # Step 1: 生成回答（带 anti-hallucination prompt）
        prompt = ANTI_HALLUCINATION_PROMPT.format(
            context=context, question=question
        )
        answer = llm.invoke(prompt).content

        # Step 2: 检测幻觉
        check = self.detector.check_faithfulness(answer, context)

        if check["has_hallucination"]:
            # Step 3: 幻觉兜底
            hallucinated = [d["statement"] for d in check["details"]
                          if d["is_hallucination"]]

            return {
                "answer": f"我找到了一些相关信息，但无法完全确认答案的准确性。"
                          f"建议查看以下原始资料进行核实。",
                "confidence": "low",
                "source_docs": context[:500],
                "flagged_statements": hallucinated,
                "faithfulness": check["faithfulness"]
            }

        return {
            "answer": answer,
            "confidence": "high",
            "faithfulness": check["faithfulness"]
        }
```

### Q: 当不同 Chunk 包含矛盾信息时怎么办？

**Chunk 冲突是 RAG 生产中非常常见的问题——同一主题的不同文档可能给出不同信息。**

**三种冲突解决策略：**

| 策略 | 原理 | 适用场景 |
|------|------|----------|
| **时间优先** | 选最新文档 | 政策/规章更新 |
| **权威优先** | 按来源权威度排序 | 官方 vs 非官方 |
| **LLM 仲裁** | 让 LLM 分析矛盾并做出判断 | 复杂/模糊情况 |

```python
class ChunkConflictResolver:
    def __init__(self, llm):
        self.llm = llm

    def detect_conflicts(self, chunks: list[dict]) -> list[dict]:
        """检测 chunks 之间的矛盾"""
        prompt = f"""分析以下文档片段，检测其中是否存在矛盾或冲突信息。

        片段列表：
        {json.dumps([{"id": i, "content": c["content"][:300], "source": c.get("source", "unknown"),
                       "date": c.get("date", "unknown")} for i, c in enumerate(chunks)], ensure_ascii=False)}

        输出 JSON 格式：
        {{
            "has_conflict": true/false,
            "conflicts": [
                {{
                    "chunks": [0, 2],
                    "description": "关于年假天数的矛盾：片段0说10天，片段2说15天",
                    "topic": "年假政策"
                }}
            ]
        }}
        """
        return json.loads(self.llm.invoke(prompt).content)

    def resolve_by_time(self, conflicting_chunks: list[dict]) -> dict:
        """时间优先策略：选最新的"""
        return max(conflicting_chunks, key=lambda c: c.get("date", "1970-01-01"))

    def resolve_by_authority(self, conflicting_chunks: list[dict]) -> dict:
        """权威优先策略：按来源权威度"""
        authority_order = {
            "official_policy": 4,
            "internal_memo": 3,
            "meeting_notes": 2,
            "email": 1,
            "unknown": 0
        }
        return max(conflicting_chunks,
                   key=lambda c: authority_order.get(c.get("source_type", "unknown"), 0))

    def resolve_by_llm(self, query: str, conflicting_chunks: list[dict]) -> str:
        """LLM 仲裁：让 LLM 分析矛盾并给出判断"""
        prompt = f"""以下文档片段关于同一主题但存在矛盾信息。
        请分析矛盾，综合判断，给出最可能正确的答案。

        问题：{query}

        矛盾的信息片段：
        {json.dumps([{"content": c["content"], "source": c.get("source"),
                       "date": c.get("date")} for c in conflicting_chunks], ensure_ascii=False)}

        请：
        1. 指出矛盾所在
        2. 分析每个来源的可信度
        3. 给出你的判断和理由
        """
        return self.llm.invoke(prompt).content
```

### Q: RAG 系统如何做权限隔离？

**权限隔离四层架构：确保不同用户只能看到自己有权限的文档。**

```
四层权限隔离架构：

  ┌────────────────────────────────────────────┐
  │ Layer 1: Query Filter（查询过滤）           │
  │ 在检索前，根据用户角色添加过滤条件            │
  │ "只搜索 department=engineering 的文档"       │
  └──────────────┬───────────────────────────────┘
                 │
  ┌──────────────▼───────────────────────────────┐
  │ Layer 2: Chunk Labels（文档标签）             │
  │ 每个 chunk 存储元数据：department, level,     │
  │ access_group 等权限标签                       │
  └──────────────┬───────────────────────────────┘
                 │
  ┌──────────────▼───────────────────────────────┐
  │ Layer 3: Generation Guard（生成守卫）         │
  │ LLM 生成后检查回答是否泄露了无权限内容         │
  └──────────────┬───────────────────────────────┘
                 │
  ┌──────────────▼───────────────────────────────┐
  │ Layer 4: Audit Log（审计日志）                │
  │ 记录谁在什么时间访问了什么文档                  │
  └──────────────────────────────────────────────┘
```

```python
class PermissionIsolatedRAG:
    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm

    def query(self, question: str, user: dict) -> dict:
        """带权限隔离的 RAG 查询"""

        # === Layer 1: Query Filter ===
        # 根据用户角色构建过滤条件
        metadata_filter = self._build_filter(user)

        # === Layer 2: Chunk Labels ===
        # 检索时只返回有权限的 chunks
        results = self.vectorstore.similarity_search(
            question,
            k=10,
            filter=metadata_filter  # 向量数据库的元数据过滤
        )

        # 二次校验：确保每个 chunk 的权限标签匹配
        authorized_chunks = [
            r for r in results
            if self._check_access(user, r.metadata)
        ]

        context = "\n".join([c.page_content for c in authorized_chunks])

        # === Layer 3: Generation Guard ===
        answer = self.llm.invoke(
            f"根据以下信息回答：\n{context}\n\n问题：{question}"
        ).content

        # 检查答案是否泄露了不在 authorized_chunks 中的信息
        leak_check = self._check_information_leak(answer, authorized_chunks, user)
        if leak_check["has_leak"]:
            answer = "抱歉，部分信息超出了您的访问权限，无法完整回答。"

        # === Layer 4: Audit Log ===
        self._log_access(user, question, authorized_chunks, answer)

        return {"answer": answer, "sources": [c.metadata for c in authorized_chunks]}

    def _build_filter(self, user: dict) -> dict:
        """根据用户角色构建元数据过滤条件"""
        filters = {"department": {"$in": user["departments"]}}
        if user["role"] != "admin":
            filters["security_level"] = {"$lte": user["clearance_level"]}
        return filters

    def _check_access(self, user: dict, metadata: dict) -> bool:
        """二次校验 chunk 级权限"""
        # 检查部门权限
        if metadata.get("department") not in user["departments"]:
            return False
        # 检查安全等级
        if metadata.get("security_level", 0) > user.get("clearance_level", 0):
            return False
        # 检查 access group
        if metadata.get("access_group"):
            if metadata["access_group"] not in user.get("groups", []):
                return False
        return True

    def _log_access(self, user, question, chunks, answer):
        """审计日志"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user["id"],
            "user_role": user["role"],
            "question": question,
            "accessed_docs": [c.metadata.get("doc_id") for c in chunks],
            "answer_length": len(answer)
        }
        audit_logger.info(json.dumps(log_entry))
```

### Q: 什么是自愈 RAG？如何实现？

**自愈 RAG = 系统能自动检测失败、分级恢复、并验证恢复结果。**

```
自愈 RAG 工作流程：

  正常查询
       │
       ▼
  ┌──────────────┐
  │ 失败检测      │  ← 检测检索/生成是否失败
  │ (Detection)   │
  └──────┬───────┘
         │
    ┌────┼──────────┐
    │    │          │
  [检索失败]  [生成失败]  [质量不达标]
    │    │          │
    ▼    ▼          ▼
  ┌──────────────┐
  │ 分级恢复      │  ← 根据失败类型选择恢复策略
  │ (Recovery)    │
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │ 验证          │  ← 验证恢复结果是否可接受
  │ (Verify)      │
  └──────┬───────┘
         │
    ┌────┴────┐
  [通过]    [未通过]
    │         │
    ▼         └→ 升级到更高级恢复策略 / 人工介入
  返回结果
```

```python
class SelfHealingRAG:
    """自愈 RAG：自动检测失败并恢复"""

    RECOVERY_LEVELS = [
        "retry",           # Level 0: 重试
        "query_rewrite",   # Level 1: 改写 query
        "expand_search",   # Level 2: 扩大检索范围
        "fallback_web",    # Level 3: 降级到 Web 搜索
        "admit_failure"    # Level 4: 承认无法回答
    ]

    def query(self, question: str, max_recovery_level: int = 3) -> dict:
        """带自愈能力的查询"""

        for level in range(max_recovery_level + 1):
            strategy = self.RECOVERY_LEVELS[level]

            try:
                # 根据恢复级别执行不同策略
                result = self._execute_strategy(question, strategy)

                # 验证结果质量
                quality = self._verify_quality(question, result)

                if quality["score"] >= 0.7:
                    result["recovery_level"] = level
                    result["quality"] = quality
                    return result
                else:
                    # 质量不达标，升级到下一级恢复策略
                    continue

            except Exception as e:
                # 当前策略执行失败，升级
                continue

        # 所有策略都失败
        return {
            "answer": "抱歉，我目前无法准确回答这个问题。建议联系人工客服。",
            "confidence": "none",
            "recovery_level": max_recovery_level + 1,
            "escalated": True
        }

    def _execute_strategy(self, question: str, strategy: str) -> dict:
        if strategy == "retry":
            return self._standard_rag(question)

        elif strategy == "query_rewrite":
            rewritten = self._rewrite_query(question)
            return self._standard_rag(rewritten)

        elif strategy == "expand_search":
            # 扩大 k 值，降低相似度阈值
            return self._standard_rag(question, k=20, threshold=0.3)

        elif strategy == "fallback_web":
            # 降级到 Web 搜索
            web_results = self._web_search(question)
            return self._generate_with_context(question, web_results)

        elif strategy == "admit_failure":
            return {"answer": "无法回答", "confidence": "none"}

    def _verify_quality(self, question: str, result: dict) -> dict:
        """验证回答质量"""
        answer = result.get("answer", "")
        context = result.get("context", "")

        # 检查多个质量维度
        checks = {
            "not_empty": len(answer.strip()) > 10,
            "not_generic": "我不知道" not in answer and "无法" not in answer,
            "has_specifics": any(char.isdigit() for char in answer) or len(answer) > 50,
            "faithful": self._check_faithfulness(answer, context) > 0.7
        }

        score = sum(checks.values()) / len(checks)
        return {"score": score, "checks": checks}
```

### Q: 如何实现语义缓存来优化 RAG 性能？

**语义缓存（Semantic Cache）= 当新问题和之前的问题语义相似时，直接返回缓存的答案，跳过检索和生成。**

**前端类比：** 语义缓存就像 React Query / SWR 的缓存，但 key 不是 URL 字符串，而是语义向量。"React 性能优化" 和 "如何提升 React 应用速度" 虽然字面不同，但语义缓存能识别它们是同一个问题。

```python
import numpy as np
from datetime import datetime, timedelta

class SemanticCache:
    """语义缓存：基于语义相似度的 RAG 缓存"""

    def __init__(self, embeddings, similarity_threshold: float = 0.92,
                 ttl_hours: int = 24):
        self.embeddings = embeddings
        self.threshold = similarity_threshold
        self.ttl = timedelta(hours=ttl_hours)
        self.cache = []  # [{query, embedding, answer, timestamp, hit_count}]

    def get(self, query: str) -> dict | None:
        """语义匹配查找缓存"""
        query_embedding = self.embeddings.embed_query(query)

        best_match = None
        best_score = 0

        for entry in self.cache:
            # 检查是否过期
            if datetime.now() - entry["timestamp"] > self.ttl:
                continue

            # 计算余弦相似度
            similarity = self._cosine_similarity(query_embedding, entry["embedding"])

            if similarity > self.threshold and similarity > best_score:
                best_match = entry
                best_score = similarity

        if best_match:
            best_match["hit_count"] += 1
            return {
                "answer": best_match["answer"],
                "cached": True,
                "similarity": best_score,
                "original_query": best_match["query"]
            }

        return None

    def set(self, query: str, answer: str):
        """存入缓存"""
        embedding = self.embeddings.embed_query(query)
        self.cache.append({
            "query": query,
            "embedding": embedding,
            "answer": answer,
            "timestamp": datetime.now(),
            "hit_count": 0
        })

        # LRU 清理：保留最近 1000 条
        if len(self.cache) > 1000:
            self.cache.sort(key=lambda x: x["timestamp"], reverse=True)
            self.cache = self.cache[:1000]

    def _cosine_similarity(self, a, b) -> float:
        a, b = np.array(a), np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 集成到 RAG Pipeline
class CachedRAG:
    def __init__(self, rag_pipeline, embeddings):
        self.rag = rag_pipeline
        self.cache = SemanticCache(embeddings, similarity_threshold=0.92)

    def query(self, question: str) -> dict:
        # 1. 先查缓存
        cached = self.cache.get(question)
        if cached:
            print(f"Cache HIT! Similarity: {cached['similarity']:.3f}")
            return cached

        # 2. 缓存未命中，走正常 RAG Pipeline
        result = self.rag.query(question)

        # 3. 存入缓存
        self.cache.set(question, result["answer"])

        result["cached"] = False
        return result

# 效果示例：
# Query: "React 的虚拟 DOM 是什么？" → Cache MISS → 正常 RAG → 缓存
# Query: "什么是 React Virtual DOM？"  → Cache HIT! Similarity: 0.95 → 直接返回
# 节省：~2s 延迟 + ~$0.01 API 调用成本
```

**语义缓存的性能提升：**

| 指标 | 无缓存 | 有语义缓存 | 提升 |
|------|--------|-----------|------|
| 平均延迟 | 2.5s | 0.8s（含缓存命中） | 68% |
| API 成本 | $1.00/千次 | $0.35/千次 | 65% |
| 缓存命中率 | - | 40-60%（典型） | - |

**面试话术：**
> "RAG 生产中我关注五个关键问题：第一是幻觉防护，我用三层体系——Prompt 层预防、NLI 模型检测、兜底降级。第二是 Chunk 冲突，用时间优先/权威优先/LLM 仲裁三种策略。第三是权限隔离，四层架构确保数据安全。第四是自愈能力，系统能自动检测失败并分级恢复。第五是语义缓存，用语义相似度匹配减少重复计算，实测能降低 60% 以上的延迟和成本。"

---

## 5.9 RAG vs SFT 选型

### Q: 什么时候用 RAG，什么时候用 SFT（微调）？能否结合使用？

**RAG 和 SFT 不是互斥的，而是互补的——RAG 给模型"开卷考试"的能力，SFT 给模型"专业技能"。**

**决策框架：**

```
                    ┌─────────────────────────┐
                    │   你的知识需要频繁更新吗？  │
                    └──────────┬──────────────┘
                          ┌───┴───┐
                         Yes     No
                          │       │
                          ▼       ▼
                      ┌──────┐  ┌───────────────────┐
                      │ RAG  │  │ 需要改变模型的行为  │
                      │ 优先 │  │ /风格/格式吗？      │
                      └──────┘  └────────┬──────────┘
                                    ┌───┴───┐
                                   Yes     No
                                    │       │
                                    ▼       ▼
                                ┌──────┐  ┌──────┐
                                │ SFT  │  │ RAG  │
                                │ 优先 │  │ 就够 │
                                └──────┘  └──────┘
```

**详细对比：**

| 维度 | RAG | SFT（微调） | RAG + SFT |
|------|-----|-------------|-----------|
| **知识更新** | 实时（更新文档即可） | 需要重新训练 | 实时 |
| **成本** | 检索 + API 调用 | 训练 GPU 成本高 | 两者之和 |
| **延迟** | 较高（检索+生成） | 较低（直接生成） | 较高 |
| **幻觉控制** | 好（可溯源） | 差（可能编造） | 最好 |
| **个性化** | 弱 | 强（改变模型风格） | 最强 |
| **可解释性** | 高（能给出来源） | 低 | 高 |
| **数据隐私** | 好（数据不入模型） | 差（数据进入模型参数） | 好 |
| **适用场景** | 知识密集型问答 | 风格/格式定制 | 企业级 AI 应用 |

**具体场景选型：**

| 场景 | 选择 | 理由 |
|------|------|------|
| 企业知识库问答 | **RAG** | 知识频繁更新，需要溯源 |
| 客服机器人 | **RAG + SFT** | RAG 提供知识，SFT 定制回答风格 |
| 代码生成（特定框架） | **SFT** | 需要学习特定代码风格和 API |
| 医疗/法律问答 | **RAG** | 需要精确溯源，不能幻觉 |
| 翻译（特定术语体系） | **SFT** | 需要统一术语翻译 |
| 多语言文档检索 | **RAG** | 检索多语言知识库 |

```python
# RAG + SFT 结合示例：
# Step 1: SFT 微调一个专门的"客服风格"模型
# → 模型学会用温和、专业、简洁的语气回答
# → 模型学会特定的回答格式（问候→回答→确认→结束）

# Step 2: 用 RAG 给微调后的模型提供实时知识
from langchain.chains import RetrievalQA

# 使用微调后的模型（SFT），但通过 RAG 提供知识
sft_model = ChatOpenAI(
    model="ft:gpt-4o-mini-2024-07-18:company:customer-service:xxx",  # SFT 微调模型
    temperature=0.3
)

qa_chain = RetrievalQA.from_chain_type(
    llm=sft_model,       # SFT 提供回答风格
    retriever=retriever,  # RAG 提供实时知识
    chain_type="stuff"
)
# 结果：模型既有专业客服风格，又能基于最新知识回答
```

**面试话术：**
> "RAG 和 SFT 的选型我用一个简单框架：如果核心需求是'知识'——用 RAG，因为知识可以实时更新且可溯源；如果核心需求是'能力/风格'——用 SFT，因为行为模式需要通过训练内化。大多数企业级应用是两者结合：SFT 定制模型的回答风格和格式，RAG 提供实时准确的知识。类比前端：SFT 像是定制一个 UI 组件库的设计规范，RAG 像是给组件接入实时数据 API。"

---

## 5.10 2026 RAG 新趋势

### Q: 2026 年 RAG 领域有哪些新趋势？

**2026 年 RAG 正在经历从"检索增强"到"智能增强"的范式转变。**

**趋势一：Agentic RAG 成为主流**

```
2024: 传统 RAG（单次检索）
2025: Advanced RAG（混合检索 + Rerank）
2026: Agentic RAG（Agent 驱动的自适应检索）

Agentic RAG 的核心变化：
  - 检索不再是固定 Pipeline，而是 Agent 的一个 Tool
  - Agent 可以决定何时检索、检索什么、从哪里检索
  - 支持多轮检索、跨数据源检索、结果自验证
  - 与 Function Calling 和 MCP 深度集成
```

**趋势二：Memory-Augmented AI（记忆增强 AI）**

```
传统 RAG：每次查询独立，无"记忆"
Memory-Augmented AI：系统有短期/长期记忆

  ┌─────────────────────────────────────────┐
  │        Memory-Augmented Architecture     │
  │                                          │
  │  ┌──────────┐  ┌──────────┐  ┌────────┐ │
  │  │ Working   │  │ Episodic │  │ Long   │ │
  │  │ Memory    │  │ Memory   │  │ Term   │ │
  │  │ (当前对话) │  │ (历史交互) │  │ Memory │ │
  │  │           │  │           │  │ (知识库)│ │
  │  └──────────┘  └──────────┘  └────────┘ │
  │       ↑              ↑            ↑      │
  │       └──────────────┼────────────┘      │
  │                      │                   │
  │              Memory Controller           │
  │              (决定读写哪层记忆)            │
  └─────────────────────────────────────────┘

关键能力：
  - 记住用户偏好（"上次你说你更喜欢简洁的回答"）
  - 跨会话知识积累（"基于你之前问过的 React 问题..."）
  - 自动更新知识（新信息覆盖旧信息）
```

```python
# Memory-Augmented RAG 示例
class MemoryAugmentedRAG:
    def __init__(self):
        self.working_memory = []           # 当前对话
        self.episodic_memory = VectorStore()  # 历史交互（向量化存储）
        self.long_term_memory = VectorStore() # 知识库

    def query(self, question: str, user_id: str) -> str:
        # 1. 从 episodic memory 获取用户历史偏好
        user_context = self.episodic_memory.search(
            query=question,
            filter={"user_id": user_id},
            k=3
        )

        # 2. 从 long-term memory 获取知识
        knowledge = self.long_term_memory.search(question, k=5)

        # 3. 结合 working memory（当前对话）生成回答
        context = {
            "user_preferences": user_context,
            "knowledge": knowledge,
            "conversation_history": self.working_memory[-10:]  # 最近 10 轮
        }

        answer = self.llm.generate(question, context)

        # 4. 更新记忆
        self.working_memory.append({"role": "user", "content": question})
        self.working_memory.append({"role": "assistant", "content": answer})
        self.episodic_memory.add({
            "content": f"Q: {question}\nA: {answer}",
            "user_id": user_id,
            "timestamp": datetime.now()
        })

        return answer
```

**趋势三：GraphRAG 工具生态成熟**

```
2024: GraphRAG 概念提出（微软论文）
2025: 早期开源实现（neo4j-graphrag, LlamaIndex PropertyGraph）
2026: 工具链成熟，生产可用

关键进展：
  - 实体抽取从 LLM-only 转向 LLM + NER 混合（成本降低 10x）
  - 增量图更新（不需要每次全量重建）
  - 图数据库原生支持 GraphRAG（Neo4j GraphRAG package）
  - 多模态实体（图片中的实体 + 文本实体统一图谱）
```

**趋势四：Retrieval-free Reasoning（无检索推理）**

```
随着模型上下文窗口的扩大（GPT-4o: 128K, Claude: 200K, Gemini: 1M+），
部分场景开始质疑"是否还需要 RAG"：

Retrieval-free 的思路：
  - 直接把全部文档塞进超长 Context Window
  - 不需要 Chunking、Embedding、向量数据库
  - 简化架构，减少检索误差

但 RAG 仍然不可替代的场景：
  1. 文档总量 > Context Window（企业知识库可能有 TB 级别）
  2. 需要实时更新的知识
  3. 需要权限隔离
  4. 成本敏感（长 context 的 API 成本极高）
  5. 需要精确溯源
```

**RAG vs Long Context 对比：**

| 维度 | RAG | Long Context |
|------|-----|-------------|
| 文档规模限制 | 无（向量数据库可存储 TB 级） | 受 Context Window 限制 |
| 精度 | 高（检索聚焦相关段落） | 中（Lost in the Middle） |
| 成本 | 低（只处理相关 chunks） | 高（按全量 token 计费） |
| 延迟 | 中（检索 + 生成） | 高（长 context 推理慢） |
| 实时性 | 高（更新文档即可） | 低（需要重新构建 context） |
| 架构复杂度 | 高（完整 Pipeline） | 低（直接塞 context） |
| 溯源能力 | 强（可精确到 chunk） | 弱（需要额外处理） |

**趋势五：多模态 RAG**

```
2026 多模态 RAG 的关键能力：
  - 图片检索：用户传图片，检索相关文档
  - 表格理解：解析 PDF 中的表格，结构化存储
  - 视频检索：基于视频帧 + 字幕的联合检索
  - 代码检索：代码语义 Embedding + AST 结构搜索

技术栈示例：
  - 图片 Embedding：CLIP / SigLIP
  - 表格解析：Table Transformer / Unstructured.io
  - 多模态向量数据库：Weaviate（原生多模态支持）
```

**2026 RAG 技术栈推荐：**

| 层次 | 推荐方案 | 备选 |
|------|----------|------|
| **文档处理** | Unstructured.io | LlamaIndex |
| **Embedding** | OpenAI text-embedding-3 | Cohere Embed v3 / BGE-M3 |
| **向量数据库** | Weaviate / Qdrant | Pinecone / Milvus |
| **检索** | 混合检索（向量 + BM25 + RRF） | ColBERT |
| **Rerank** | Cohere Rerank v3 | BGE-Reranker-v2 |
| **框架** | LangChain / LlamaIndex | Haystack |
| **评估** | RAGAS + DeepEval | TruLens |
| **图数据库** | Neo4j（GraphRAG） | FalkorDB |
| **缓存** | 语义缓存 + Redis | GPTCache |
| **监控** | LangSmith / Phoenix | Weights & Biases |

**面试话术：**
> "2026 年 RAG 的最大趋势是 Agentic RAG 和 Memory-Augmented AI。Agentic RAG 把检索从固定 Pipeline 变成了 Agent 的动态工具调用，支持多轮自适应检索。Memory-Augmented AI 给系统加上了短期/长期记忆，能记住用户偏好和历史交互。GraphRAG 工具链成熟了，生产可用性大幅提升。不过我也关注 Retrieval-free 趋势——随着 Context Window 扩大到百万 token，简单场景可能不再需要 RAG，但企业级应用因为数据规模、权限隔离和成本考虑，RAG 仍然是核心架构。"

---

## 本章总结

```
RAG 核心要点速查：

┌──────────────┬──────────────────────────────────────────────┐
│ 主题         │ 关键记忆点                                    │
├──────────────┼──────────────────────────────────────────────┤
│ 基础 Pipeline │ Indexing(加载→分块→向量化→存储) +             │
│              │ Query(改写→检索→重排→组装→生成→后处理)         │
├──────────────┼──────────────────────────────────────────────┤
│ 分块策略      │ 固定(512 tokens) > 语义 > Late Chunking       │
│              │ Context Cliff: >2500 tokens 质量断崖下降       │
├──────────────┼──────────────────────────────────────────────┤
│ 检索优化      │ 混合检索(向量+BM25) + RRF 融合                 │
│              │ HyDE: 假设性答案 Embedding 检索                │
│              │ Multi-Query: 多角度改写提升召回                 │
├──────────────┼──────────────────────────────────────────────┤
│ Rerank       │ BiEncoder 粗召回(百万级) →                     │
│              │ CrossEncoder 精排(Top-100→Top-5)              │
├──────────────┼──────────────────────────────────────────────┤
│ 上下文处理    │ Lost in the Middle: 重要信息放开头/结尾         │
│              │ LLMLingua: 2-5x 压缩，降本提速                 │
├──────────────┼──────────────────────────────────────────────┤
│ 评估体系      │ RAGAS 四指标: Faithfulness + Relevancy +      │
│              │ Context Precision + Context Recall             │
│              │ DeepEval: CI/CD 集成 pytest                    │
├──────────────┼──────────────────────────────────────────────┤
│ 进阶范式      │ GraphRAG(知识图谱+多跳推理)                    │
│              │ Agentic RAG(Agent 多轮检索)                    │
│              │ Self-RAG(模型自主判断是否检索)                  │
│              │ CRAG(三级质量过滤)                              │
├──────────────┼──────────────────────────────────────────────┤
│ 生产问题      │ 幻觉防护: Prompt→检测→兜底 三层                │
│              │ 权限隔离: Filter→Labels→Guard→Audit 四层       │
│              │ 自愈 RAG: 检测→分级恢复→验证                    │
│              │ 语义缓存: 相似问题直接返回，节省 60%+ 成本       │
├──────────────┼──────────────────────────────────────────────┤
│ RAG vs SFT   │ 知识→RAG, 风格→SFT, 企业级→两者结合            │
├──────────────┼──────────────────────────────────────────────┤
│ 2026 趋势    │ Agentic RAG, Memory-Augmented AI,             │
│              │ GraphRAG 成熟, 多模态 RAG, Retrieval-free      │
└──────────────┴──────────────────────────────────────────────┘
```

---

## 5.11 RAG 文档更新与版本管理

### Q: RAG 知识库的文档更新后，索引如何同步？版本管理怎么做？

**核心挑战：** 知识库不是静态的——文档会增删改。如果 Embedding 索引不同步，用户检索到的是过时信息。

**三种更新策略：**

| 策略 | 做法 | 适用场景 | 延迟 |
|------|------|----------|------|
| **全量重建** | 删除旧索引，全部重新 Embedding | 数据量小（< 1 万条） | 高（分钟级） |
| **增量更新** | 只处理变化的文档（新增/修改/删除） | 数据量大，频繁更新 | 低（秒级） |
| **定时批量** | 每天凌晨全量重建一次 | 对实时性要求不高 | 固定延迟 |

**增量更新的实现：**

```python
class RAGIndexManager:
    def __init__(self, vectorstore, embedding_model):
        self.vectorstore = vectorstore
        self.embedding_model = embedding_model

    def sync_document(self, doc_id: str, new_content: str, action: str):
        if action == "create":
            # 新文档：分块 → Embedding → 写入向量库
            chunks = self.split(new_content)
            vectors = self.embedding_model.encode(chunks)
            self.vectorstore.upsert(
                ids=[f"{doc_id}_chunk_{i}" for i in range(len(chunks))],
                embeddings=vectors,
                metadatas=[{"doc_id": doc_id, "version": self.get_version(doc_id)}
                           for _ in chunks]
            )

        elif action == "update":
            # 修改文档：先删旧 chunks，再插新 chunks
            self.vectorstore.delete(filter={"doc_id": doc_id})
            self.sync_document(doc_id, new_content, "create")

        elif action == "delete":
            # 删除文档：删除该文档所有 chunks
            self.vectorstore.delete(filter={"doc_id": doc_id})

    def get_version(self, doc_id):
        """返回文档当前版本号，用于审计"""
        return hashlib.md5(doc_id.encode()).hexdigest()[:8]
```

**版本管理关键设计：**

```
每个 Chunk 的 Metadata 应包含：
{
    "doc_id": "policy-2024-001",        # 原文档 ID
    "doc_title": "员工考勤制度",         # 文档标题
    "version": "v3",                     # 版本号
    "updated_at": "2026-05-01",         # 更新时间
    "source": "hr-portal",              # 来源系统
    "chunk_index": 2,                    # 在文档中的位置
    "content_hash": "a1b2c3..."         # 内容哈希（检测是否真的变了）
}

→ 检索时可以按 version 过滤（只用最新版）
→ 审计时可以追溯答案来源的具体版本
```

**防止"幽灵文档"问题：**

```
幽灵文档 = 原文已删除，但向量库中还残留旧 chunks

解决方案：
1. 文档系统发 Webhook → RAG 同步删除
2. 定时对账：对比文档系统和向量库的 doc_id 列表
3. Metadata 加 TTL（过期时间），定期清理
```

**面试话术：**
> "RAG 文档更新我用增量同步——监听文档系统的变更事件（Webhook），有创建/修改/删除就同步处理。关键是每个 Chunk 的 Metadata 要有 doc_id、version、updated_at，这样既能按版本过滤，又能审计追溯。还要有定时对账机制，防止'幽灵文档'——原文已删但向量库残留。"

---

[← 04 - Embedding 与向量数据库](./04-embedding.md) | [目录](./README.md) | [06 - AI Agent →](./06-agent.md)
