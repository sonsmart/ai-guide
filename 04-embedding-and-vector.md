# 04 - Embedding 与向量检索

> **难度：** ⭐⭐⭐ | **定位：** RAG 和语义搜索的基础设施核心
>
> **前端类比：** Embedding 之于 AI 搜索，就像 Index 之于数据库查询——把非结构化的文本变成可快速检索的 "索引"。向量数据库就是 AI 世界的 Elasticsearch。

## 本章知识树

```
Embedding 与向量检索
├── 4.1 什么是 Embedding
├── 4.2 Embedding 模型选型（BGE、text-embedding-3、GTE）
├── 4.3 向量相似度（余弦、欧几里得、点积）
├── 4.4 向量数据库对比（Milvus / Qdrant / Pinecone / pgvector / FAISS）
├── 4.5 向量索引算法（HNSW / IVF / PQ / LSH）
├── 4.6 Embedding 微调
├── 4.7 多模态 Embedding
└── 4.8 向量数据库选型决策
```

---

## 4.1 什么是 Embedding

### Q: 什么是 Embedding？为什么 AI 应用离不开它？

**Embedding = 把人类能理解的内容（文字、图片、音频）变成机器能计算的数字向量。**

```
传统搜索（关键词匹配）：
  Query: "如何提高代码质量"
  ❌ 匹配不到: "编写高质量的软件" （没有共同关键词）

语义搜索（Embedding）：
  Query:    "如何提高代码质量"   → [0.12, -0.34, 0.56, ..., 0.78]  (1536维)
  Document: "编写高质量的软件"   → [0.11, -0.33, 0.55, ..., 0.77]  (1536维)
  → 余弦相似度 = 0.95 → 高度相关 ✅
```

**1536 维是什么意思？**

每个维度捕捉文本的一个 "语义侧面"。可以想象：

```
假设只有 3 维（实际有 1536 维）：
  维度1: 情感极性     [-1 负面 ... +1 正面]
  维度2: 主题（技术）  [-1 非技术 ... +1 技术]
  维度3: 语气（正式）  [-1 口语化 ... +1 正式]

"React 性能优化指南"  → [+0.2, +0.95, +0.8]   (正面、很技术、很正式)
"这代码写得像坨💩"    → [-0.9, +0.6,  -0.8]   (负面、偏技术、口语化)
"今天天气真好"        → [+0.7, -0.9,  -0.3]   (正面、非技术、偏口语)
```

实际的 1536 维远比这复杂，每个维度的含义不是人为定义的，而是模型从海量数据中自动学习到的。

**前端类比：** Embedding 像是给每段文本生成一个 "语义 Hash"。

```
前端的 HashMap:
  key: "user_123" → bucket: 7  （基于字符串计算，同一个字符串永远映射同一个桶）

Embedding 的 "语义 HashMap":
  key: "如何提高代码质量"  → vector: [0.12, -0.34, ...]
  key: "编写高质量的软件"  → vector: [0.11, -0.33, ...]  （语义相近 → 向量相近）
  key: "今天中午吃什么"    → vector: [0.89, 0.45, ...]   （语义无关 → 向量很远）
```

区别在于：HashMap 是精确匹配（key 完全相同才能命中），而 Embedding 是 **模糊语义匹配**（意思相近就能找到）。

**文本转向量的流程：**

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   原始文本    │ ──→ │  Embedding 模型   │ ──→ │  固定长度向量     │
│ "React 真好用"│     │ (e.g. OpenAI API)│     │ [0.12, -0.34, ...│
│              │     │                  │     │  ..., 0.78] 1536d│
└──────────────┘     └──────────────────┘     └──────────────────┘
```

**为什么需要 Embedding？**

| 场景 | 没有 Embedding | 有 Embedding |
|------|---------------|-------------|
| 搜索 | 关键词匹配，"汽车" 搜不到 "车辆" | 语义搜索，同义词/近义词都能匹配 |
| 推荐 | 基于标签 / 协同过滤 | 基于内容语义相似度推荐 |
| RAG | 无法实现 | LLM + 外部知识库检索 |
| 分类 | 需要大量标注数据 | 计算与已知类别的向量距离即可 |
| 去重 | 精确匹配 | 语义层面去重（不同表述同一含义） |

**面试话术：**
> "Embedding 是把非结构化数据变成结构化向量表示的过程。它的核心价值是让机器理解 '语义相似性'——两段文字即使用词完全不同，只要含义接近，它们的向量在高维空间中就会靠得很近。这是 RAG、语义搜索、推荐系统的基础。"

---

## 4.2 Embedding 模型选型

### Q: 主流 Embedding 模型有哪些？怎么选？

**2024-2025 主流 Embedding 模型对比：**

| 模型 | 维度 | Max Tokens | 中文 | 英文 | 多语言 | MTEB 均分 | 价格 | 开源 |
|------|------|-----------|------|------|--------|----------|------|------|
| **text-embedding-3-large** (OpenAI) | 3072 (可降维) | 8191 | ✅ 好 | ✅ 优 | ✅ | ~64.6 | $0.13/1M tokens | ❌ |
| **text-embedding-3-small** (OpenAI) | 1536 | 8191 | ✅ 中 | ✅ 好 | ✅ | ~62.3 | $0.02/1M tokens | ❌ |
| **BGE-large-zh-v1.5** (BAAI) | 1024 | 512 | ✅ 优 | ✅ 中 | ❌ | ~63.5 (C-MTEB) | 免费 | ✅ |
| **BGE-M3** (BAAI) | 1024 | 8192 | ✅ 优 | ✅ 好 | ✅ 100+语言 | ~66.1 | 免费 | ✅ |
| **GTE-large-zh** (Alibaba) | 1024 | 512 | ✅ 优 | ✅ 好 | ❌ | ~64.0 (C-MTEB) | 免费 | ✅ |
| **GTE-Qwen2-7B** (Alibaba) | 3584 | 131072 | ✅ 优 | ✅ 优 | ✅ | ~70.2 | 免费 | ✅ |
| **Voyage-3** (Voyage AI) | 1024 | 32000 | ✅ 好 | ✅ 优 | ✅ | ~67.3 | $0.06/1M tokens | ❌ |
| **jina-embeddings-v3** (Jina AI) | 1024 | 8192 | ✅ 好 | ✅ 好 | ✅ 89语言 | ~65.5 | $0.02/1M tokens | ✅ |
| **Cohere Embed v3** | 1024 | 512 | ✅ 好 | ✅ 优 | ✅ 100+语言 | ~66.0 | $0.1/1M tokens | ❌ |

> **注意：** MTEB (Massive Text Embedding Benchmark) 分数仅供参考，不同任务（检索/分类/聚类）得分差异大。中文场景建议参考 C-MTEB 排行榜。

**模型选型决策树：**

```
你的场景？
│
├─→ 纯中文场景
│   ├─→ 预算有限 / 可自部署 → BGE-large-zh-v1.5 或 GTE-large-zh
│   ├─→ 需要长文本 (>512 tokens) → BGE-M3 或 GTE-Qwen2-7B
│   └─→ 不想部署，用 API → text-embedding-3-large
│
├─→ 纯英文场景
│   ├─→ 最高质量 → Voyage-3 或 text-embedding-3-large
│   ├─→ 性价比优先 → text-embedding-3-small
│   └─→ 超长文本 → jina-embeddings-v3 (8K) 或 GTE-Qwen2-7B (128K)
│
├─→ 多语言 / 中英混合
│   ├─→ 开源自部署 → BGE-M3（最佳多语言开源）
│   ├─→ API 调用 → Cohere Embed v3 或 text-embedding-3-large
│   └─→ 极致性能 → GTE-Qwen2-7B
│
└─→ 特殊需求
    ├─→ 代码检索 → Voyage-code-3 / text-embedding-3-large
    ├─→ 多模态 → CLIP / SigLIP / jina-clip-v2
    └─→ 端侧部署 → BGE-small-zh / all-MiniLM-L6-v2 (384维)
```

**代码示例 — 调用 OpenAI Embedding API：**

```python
from openai import OpenAI

client = OpenAI()

def get_embedding(text: str, model: str = "text-embedding-3-large") -> list[float]:
    """获取文本的 Embedding 向量"""
    response = client.embeddings.create(
        input=text,
        model=model,
        dimensions=1536  # text-embedding-3 支持降维，节省存储
    )
    return response.data[0].embedding

# 批量处理（推荐，减少 API 调用次数）
def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        input=texts,  # 最多 2048 条
        model="text-embedding-3-large",
        dimensions=1536
    )
    return [item.embedding for item in response.data]

# 使用
vec = get_embedding("React 组件性能优化")
print(f"维度: {len(vec)}")  # 1536
```

**代码示例 — 使用开源 BGE-M3：**

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

sentences = [
    "如何优化 React 组件性能？",
    "React performance optimization guide",
    "今天中午吃什么？"
]

# BGE-M3 支持三种检索方式：Dense、Sparse、ColBERT
embeddings = model.encode(
    sentences,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=False
)

dense_vecs = embeddings["dense_vecs"]     # shape: (3, 1024)
sparse_vecs = embeddings["lexical_weights"]  # 稀疏向量，用于混合检索
```

**前端类比：** 选 Embedding 模型就像选前端 UI 框架——没有银弹：

```
React (text-embedding-3)    : 生态最成熟、文档最全，但要付费
Vue (BGE 系列)              : 中文社区强、上手快、免费开源
Angular (GTE-Qwen2-7B)      : 功能最全最强，但体积大、部署重
Svelte (all-MiniLM)         : 极致轻量，适合资源受限场景
```

**面试话术：**
> "模型选型我会综合考虑四个维度：语言场景（中文用 BGE，多语言用 BGE-M3）、上下文长度需求（长文本选 8K+ 的模型）、部署方式（自建用开源，快速上线用 API）、和成本预算。我通常会在自己的数据集上做 A/B 测试，因为 MTEB 排行榜的分数不一定反映你具体场景的效果。"

---

## 4.3 向量相似度

### Q: 余弦相似度、欧几里得距离、点积有什么区别？什么时候用哪个？

**三种主流相似度度量方式：**

```
假设有两个 3 维向量：
  A = [1, 2, 3]
  B = [2, 4, 6]
  C = [3, 1, -2]
```

**1. Cosine Similarity（余弦相似度）— 最常用**

```
                A · B           1×2 + 2×4 + 3×6        28
cos(A,B) = ─────────── = ──────────────────────── = ────── = 1.0
            |A| × |B|    √(1²+2²+3²) × √(2²+4²+6²)  √14×√56

范围：[-1, 1]
  1.0  → 方向完全相同（最相似）
  0.0  → 正交（无关）
 -1.0  → 方向完全相反

直觉：只看 "方向"，不看 "长度"
```

**2. Euclidean Distance（欧几里得距离）**

```
d(A,B) = √((1-2)² + (2-4)² + (3-6)²)
       = √(1 + 4 + 9)
       = √14 ≈ 3.74

范围：[0, +∞)
  0    → 完全相同
  越大  → 越不相似

直觉：两个点在空间中的 "直线距离"
```

**3. Dot Product（点积 / 内积）**

```
A · B = 1×2 + 2×4 + 3×6 = 2 + 8 + 18 = 28

范围：(-∞, +∞)
  越大 → 越相似（当向量已归一化时）

直觉：同时考虑 "方向" 和 "大小"
```

**三者关系的直观理解：**

```
               向量 B
              ╱
             ╱  θ角度 → 余弦相似度关注这个
            ╱
───────────╱───────── 向量 A
           |←─ d ──→|
           欧几里得距离关注这个

当两个向量都归一化后 (|A| = |B| = 1)：
  cosine = dot_product            （三者等价）
  euclidean² = 2 - 2 × cosine     （可以互相转换）
```

**对比总结：**

| 特性 | Cosine Similarity | Euclidean Distance | Dot Product |
|------|------------------|--------------------|-------------|
| 范围 | [-1, 1] | [0, +∞) | (-∞, +∞) |
| 是否关注向量长度 | ❌ 只看方向 | ✅ 看方向+长度 | ✅ 看方向+长度 |
| 是否需要归一化 | 自动归一化 | 建议归一化 | 必须归一化 |
| 计算速度 | 中 | 中 | 快 |
| 最适合场景 | 文本相似度 | 聚类、异常检测 | 归一化向量的快速检索 |
| 主流向量库默认 | ✅ 大多数默认 | Milvus 支持 | Pinecone / Qdrant |

**什么时候用哪个？**

```
选择策略：
│
├─→ 文本语义相似度 → Cosine（最稳定，不受文本长度影响）
│
├─→ 向量已归一化（大多数 Embedding 模型输出已归一化）
│   └─→ Dot Product（计算最快，且结果等价于 Cosine）
│
├─→ 需要考虑 "强度" / "重要性"
│   └─→ Dot Product（未归一化时，长向量表示更 "强" 的语义）
│
├─→ 聚类 / K-Means
│   └─→ Euclidean（几何距离更直观）
│
└─→ 异常检测 / 离群点发现
    └─→ Euclidean（离中心越远越异常）
```

**前端类比：**

```
Cosine   ≈ CSS 颜色对比：hsl(120, 80%, 50%) vs hsl(120, 30%, 70%)
            色相相同（方向一样）→ 相似度高，不管饱和度/亮度（长度）

Euclidean ≈ CSS position: 两个元素的像素距离
            left:100px,top:200px vs left:150px,top:250px → 距离 ≈ 70.7px

Dot Product ≈ Array.reduce((sum, a, i) => sum + a * b[i], 0)
              就是逐元素相乘再求和，JavaScript 原生就能算
```

**代码示例：**

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def euclidean_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def dot_product(a, b):
    return np.dot(a, b)

# 示例
a = np.array([0.12, -0.34, 0.56, 0.78])
b = np.array([0.11, -0.33, 0.55, 0.77])
c = np.array([0.89, 0.45, -0.12, -0.67])

print(f"A vs B (相似文本): cosine={cosine_similarity(a,b):.4f}")  # ~0.9999
print(f"A vs C (不同文本): cosine={cosine_similarity(a,c):.4f}")  # ~-0.48
```

**面试话术：**
> "三种度量各有适用场景。Cosine 最常用，因为它只关注语义方向不受向量长度干扰；Euclidean 适合聚类和异常检测等需要绝对距离的场景；Dot Product 在向量归一化后等价于 Cosine，但计算更快。实际项目中，大多数 Embedding 模型的输出已经归一化，所以用 Dot Product 就够了，它在向量数据库中检索效率也最高。"

---

## 4.4 向量数据库对比

### Q: 主流向量数据库有哪些？怎么选？

**2024-2025 主流向量数据库全面对比：**

| 特性 | Milvus | Qdrant | Pinecone | pgvector | Chroma | FAISS | Weaviate |
|------|--------|--------|----------|----------|--------|-------|----------|
| **类型** | 专用向量库 | 专用向量库 | 托管向量库 | PG 扩展 | 嵌入式 | 库(非数据库) | 专用向量库 |
| **部署** | 自建/Cloud | 自建/Cloud | 仅 Cloud | 随 PG 部署 | 嵌入进程 | 嵌入进程 | 自建/Cloud |
| **语言** | Go/C++ | Rust | - | C/SQL | Python | C++ | Go |
| **标量过滤** | ✅ 强 | ✅ 强 | ✅ 好 | ✅ SQL原生 | ✅ 基础 | ❌ 需自建 | ✅ GraphQL |
| **分布式** | ✅ 原生 | ✅ (v1.7+) | ✅ 托管 | ⚠️ 需PG方案 | ❌ | ❌ | ✅ 原生 |
| **持久化** | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ 手动 | ✅ |
| **多租户** | ✅ partition | ✅ collection | ✅ namespace | ✅ schema | ❌ | ❌ | ✅ tenant |
| **混合搜索** | ✅ | ✅ | ✅ | ✅ (tsvector) | ❌ | ❌ | ✅ BM25 |
| **最大向量数** | 数十亿 | 数亿 | 数十亿 | 数千万 | 数百万 | 数十亿 | 数亿 |
| **社区** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **价格** | 开源免费 | 开源免费 | $70/月起 | 免费(PG扩展) | 开源免费 | 免费(MIT) | 开源免费 |

**100K 条 1536 维向量的性能对比（参考值）：**

| 指标 | Milvus | Qdrant | Pinecone | pgvector | Chroma | FAISS |
|------|--------|--------|----------|----------|--------|-------|
| 插入速度 | ~15s | ~12s | ~20s | ~45s | ~30s | ~5s |
| 查询延迟 (Top-10) | ~5ms | ~3ms | ~10ms | ~30ms | ~15ms | ~1ms |
| 内存占用 | ~800MB | ~600MB | N/A | ~1.2GB | ~900MB | ~580MB |
| Recall@10 | 0.98 | 0.99 | 0.97 | 0.99* | 0.97 | 0.99 |

> \* pgvector 使用 HNSW 索引时。IVFFlat 索引的 recall 约 0.92-0.95。

**各数据库的核心优势：**

```
FAISS：
  ✅ 最快（纯内存计算，Facebook 出品）
  ✅ 研发阶段首选
  ❌ 不是数据库，没有 CRUD / 持久化 / 分布式
  → 适合：原型验证、离线计算、百万级以下数据

pgvector：
  ✅ 不用额外运维（复用已有 PostgreSQL）
  ✅ SQL 原生支持，JOIN / WHERE 过滤自然
  ❌ 性能较差（非专门优化）
  → 适合：数据量 < 500 万、已有 PG 基础设施、想减少技术栈

Chroma：
  ✅ 安装简单（pip install chromadb）
  ✅ LangChain / LlamaIndex 集成好
  ❌ 不支持分布式，大数据量性能差
  → 适合：本地开发、Demo、学习用途

Qdrant：
  ✅ Rust 编写，性能优秀
  ✅ 过滤 + 向量搜索的混合查询很强
  ✅ API 设计优雅，文档清晰
  → 适合：中等规模生产环境、需要复杂过滤条件

Milvus：
  ✅ 原生分布式，支持十亿级向量
  ✅ Zilliz Cloud 托管服务
  ❌ 部署复杂（依赖 etcd、MinIO、Pulsar）
  → 适合：大规模生产环境、企业级

Pinecone：
  ✅ 全托管，零运维
  ❌ 最贵，锁定供应商，数据不可导出
  → 适合：预算充足、不想运维、快速上线

Weaviate：
  ✅ 内置向量化模块（可以直接存文本）
  ✅ GraphQL 查询接口
  → 适合：需要 Graph + Vector 混合查询
```

**前端类比：**

```
向量数据库选型 ≈ 前端状态管理选型：

FAISS       ≈ 手写 useState      → 最快最灵活，但啥都要自己做
Chroma      ≈ Zustand            → 轻量好上手，小项目足够
pgvector    ≈ 在 localStorage 里存状态 → 复用现有基础设施，但性能有限
Qdrant      ≈ Jotai              → 设计优雅、性能好、中等规模首选
Milvus      ≈ Redux + Saga       → 功能最强最完整，但配置复杂
Pinecone    ≈ Firebase Realtime  → 全托管零运维，但贵且锁定
```

**代码示例 — 使用 Qdrant：**

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# 1. 连接
client = QdrantClient(host="localhost", port=6333)

# 2. 创建 Collection
client.create_collection(
    collection_name="articles",
    vectors_config=VectorParams(
        size=1536,           # 向量维度
        distance=Distance.COSINE  # 相似度度量
    )
)

# 3. 插入数据
points = [
    PointStruct(
        id=1,
        vector=[0.12, -0.34, ...],  # 1536维向量
        payload={                    # 元数据（用于过滤）
            "title": "React 性能优化",
            "category": "frontend",
            "date": "2025-01-15"
        }
    ),
    # ...更多数据
]
client.upsert(collection_name="articles", points=points)

# 4. 搜索（带过滤）
from qdrant_client.models import Filter, FieldCondition, MatchValue

results = client.search(
    collection_name="articles",
    query_vector=[0.11, -0.33, ...],  # 查询向量
    query_filter=Filter(
        must=[
            FieldCondition(key="category", match=MatchValue(value="frontend"))
        ]
    ),
    limit=10
)

for result in results:
    print(f"Score: {result.score:.4f} | {result.payload['title']}")
```

**代码示例 — 使用 pgvector：**

```sql
-- 安装扩展
CREATE EXTENSION vector;

-- 创建表
CREATE TABLE articles (
    id SERIAL PRIMARY KEY,
    title TEXT,
    category TEXT,
    embedding vector(1536)  -- 1536维向量
);

-- 创建 HNSW 索引（推荐）
CREATE INDEX ON articles
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);

-- 插入数据
INSERT INTO articles (title, category, embedding)
VALUES ('React 性能优化', 'frontend', '[0.12, -0.34, ...]');

-- 向量搜索 + 标量过滤（SQL 原生！）
SELECT title, category,
       1 - (embedding <=> '[0.11, -0.33, ...]') AS similarity
FROM articles
WHERE category = 'frontend'
ORDER BY embedding <=> '[0.11, -0.33, ...]'
LIMIT 10;
```

**面试话术：**
> "向量数据库选型要看三个因素：数据规模（百万级以下 pgvector 或 Qdrant 就够，十亿级需要 Milvus）、运维能力（不想运维选 Pinecone 或 Zilliz Cloud）、现有技术栈（已有 PG 就加 pgvector 扩展最省事）。我们项目用的是 Qdrant，因为它性能好、API 设计优雅、部署简单，对于我们千万级数据完全够用。"

---

## 4.5 向量索引算法

### Q: HNSW、IVF、PQ、LSH 这些索引算法有什么区别？

**核心问题：** 在百万/十亿级向量中，如何快速找到最相似的 Top-K？

暴力搜索 (Brute-force) 需要逐一计算距离，O(n) 复杂度，百万级数据就要几秒。所以需要 **近似最近邻搜索 (ANN, Approximate Nearest Neighbor)** 算法，用少量精度损失换取巨大的速度提升。

**1. HNSW (Hierarchical Navigable Small World) — 最常用**

```
原理：在多层 "跳表" 式图结构中快速导航

Layer 2 (最稀疏):  A ─────────────── D
                    │                 │
Layer 1 (中等):     A ──── B ──── C ── D ──── E
                    │     │     │    │     │
Layer 0 (最密集):   A - B - C - D - E - F - G - H - I

搜索过程（查找与 Query 最近的节点）：
1. 从 Layer 2 的随机入口点 A 开始
2. 在 Layer 2 中找到最近的节点 D（大步跳转）
3. 下降到 Layer 1，从 D 出发继续搜索，找到 E
4. 下降到 Layer 0，从 E 出发精细搜索，找到最终结果 F

类比：高铁 → 动车 → 公交 → 步行，逐层细化
```

**HNSW 关键参数：**

| 参数 | 含义 | 建议值 | 影响 |
|------|------|--------|------|
| `M` | 每个节点的最大连接数 | 16-64 | 越大 → 精度越高，内存越多 |
| `ef_construction` | 建图时搜索范围 | 128-256 | 越大 → 建图越慢，质量越高 |
| `ef_search` | 搜索时搜索范围 | 64-256 | 越大 → 搜索越慢，精度越高 |

**2. IVF (Inverted File Index) — 分区搜索**

```
原理：先把向量空间分成 N 个区域（Voronoi cells），搜索时只查最相关的几个区域

建索引：用 K-Means 将所有向量分成 nlist 个簇

    ┌─────────────┐
    │ ·  ·        │
    │   · C1  ·   │    C1, C2, C3 = 三个聚类中心
    │  ·    ──────┼──────────┐
    │      ·  ·   │·  ·      │
    ├─────────────┤   C2  ·  │
    │ ·   ·       │ ·   ·    │
    │  C3   ·     │    ·     │
    │    ·  ·     │          │
    └─────────────┴──────────┘

搜索过程：
1. 计算 Query 与所有聚类中心的距离
2. 选择最近的 nprobe 个簇
3. 只在这些簇内做精确搜索

nlist=100, nprobe=10 → 只搜索 10% 的数据 → 速度提升 ~10x
```

**3. PQ (Product Quantization) — 压缩向量**

```
原理：把高维向量切成子空间，每个子空间独立量化，压缩存储

原始向量 (1536维, float32, 6KB):
[0.12, -0.34, 0.56, ..., 0.78, -0.23, 0.45, ..., 0.11]
 |←── 子空间1 (192维) ──→|←── 子空间2 ──→| ... |←── 子空间8 ──→|

每个子空间用 K-Means 聚类（通常 K=256）：
  子空间1: [0.12, -0.34, ...] → 最近的聚类中心编号 = 42  (1 byte)
  子空间2: [0.78, -0.23, ...] → 最近的聚类中心编号 = 187 (1 byte)
  ...
  子空间8: [0.34, 0.11, ...]  → 最近的聚类中心编号 = 95  (1 byte)

压缩后 (8 bytes):
[42, 187, ..., 95]

压缩比: 6144 bytes → 8 bytes = 768x 压缩！
```

**4. LSH (Locality-Sensitive Hashing) — 哈希分桶**

```
原理：设计特殊的 Hash 函数，使得相似的向量大概率落在同一个桶

    传统 Hash：相似输入 → 不同桶（散列均匀）
    LSH Hash：相似输入 → 同一个桶（局部敏感）

                Random Hyperplane
                     /
    · ·  ·          /      ·   ·
      ·    ·       /    ·    ·  ·
    · ·  ·  ·     /      ·  ·
     Hash=0       /      Hash=1

多个 Hash 函数组合，减少误判：
  向量A: hash1=0, hash2=1, hash3=1 → 桶 "011"
  向量B: hash1=0, hash2=1, hash3=1 → 桶 "011"  ← 同桶，可能相似
  向量C: hash1=1, hash2=0, hash3=0 → 桶 "100"  ← 不同桶，可能不相似
```

**四种算法对比：**

| 特性 | HNSW | IVF | PQ | LSH |
|------|------|-----|----|-----|
| 搜索速度 | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| 搜索精度 (Recall) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| 内存占用 | ⭐ (大) | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ (小) | ⭐⭐⭐ |
| 建索引速度 | ⭐⭐ (慢) | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 支持增量插入 | ✅ | ❌ 需重建 | ❌ 需重建 | ✅ |
| 适合数据规模 | 百万~千万 | 百万~亿 | 亿级+ | 百万 |
| 常见组合 | 单独使用 | IVF + PQ | PQ + OPQ | 单独使用 |

**实际使用中的常见组合：**

```
小数据量 (< 100 万)：
  → HNSW（精度最高，速度够快）
  → 大多数向量数据库的默认选择

中等数据量 (100 万 ~ 1 亿)：
  → IVF + HNSW（IVF 做粗筛，HNSW 做精排）
  → IVF + PQ（需要控制内存时）

大数据量 (> 1 亿)：
  → IVF + PQ + OPQ（极致压缩，适合内存有限场景）
  → ScaNN (Google) / DiskANN (Microsoft)（基于磁盘的混合索引）

实时插入场景：
  → HNSW（支持增量插入，不需重建索引）
```

**前端类比：**

```
HNSW  ≈ Virtual DOM Diff：通过分层结构高效定位变化节点
IVF   ≈ Code Splitting：把大包拆成小 chunk，只加载需要的部分
PQ    ≈ 图片压缩 (WebP/AVIF)：有损压缩，体积小很多，质量略降
LSH   ≈ Content-Hash 文件名：内容相似的文件分到同一个 hash bucket
```

**代码示例 — FAISS 中使用不同索引：**

```python
import faiss
import numpy as np

d = 1536     # 向量维度
n = 1000000  # 100万条向量
xb = np.random.random((n, d)).astype('float32')  # 数据库向量
xq = np.random.random((10, d)).astype('float32')  # 查询向量

# 1. Brute-force（精确搜索，做 baseline）
index_flat = faiss.IndexFlatL2(d)
index_flat.add(xb)
D, I = index_flat.search(xq, 10)  # 搜索 Top-10

# 2. IVF（分区搜索）
nlist = 100  # 100 个簇
quantizer = faiss.IndexFlatL2(d)
index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist)
index_ivf.train(xb)     # 需要训练（K-Means 聚类）
index_ivf.add(xb)
index_ivf.nprobe = 10   # 搜索最近的 10 个簇
D, I = index_ivf.search(xq, 10)

# 3. HNSW
index_hnsw = faiss.IndexHNSWFlat(d, 32)  # M=32
index_hnsw.hnsw.efConstruction = 200
index_hnsw.hnsw.efSearch = 128
index_hnsw.add(xb)  # 不需要训练
D, I = index_hnsw.search(xq, 10)

# 4. IVF + PQ（极致压缩）
m = 192      # 分成 192 个子空间
nbits = 8    # 每个子空间用 8 bits 编码 (256 个聚类)
index_ivfpq = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
index_ivfpq.train(xb)
index_ivfpq.add(xb)
index_ivfpq.nprobe = 10
D, I = index_ivfpq.search(xq, 10)

# 比较内存占用
print(f"FlatL2:  {index_flat.ntotal * d * 4 / 1e9:.2f} GB")   # ~5.7 GB
print(f"HNSW:    ~{index_flat.ntotal * d * 4 / 1e9 * 1.5:.2f} GB")  # 额外图结构
print(f"IVF+PQ:  ~{index_flat.ntotal * m * 1 / 1e9:.2f} GB")  # ~0.18 GB (32x压缩)
```

**面试话术：**
> "索引算法的选择核心是在 '搜索速度、精度、内存' 三者间做 trade-off。HNSW 精度最高、支持增量插入，是大多数场景的首选；IVF 适合大数据量、需要控制内存的场景；PQ 是终极压缩手段，可以将内存占用减少 30-50 倍但会牺牲精度。实际项目中，我通常先用 HNSW 作为 baseline，如果内存扛不住再考虑 IVF+PQ 的组合。"

---

## 4.6 Embedding 微调

### Q: 什么时候需要微调 Embedding 模型？怎么做？

**什么时候需要微调？**

```
用通用 Embedding → 直接用  ← 大多数场景够用了

需要微调的信号：
├─→ 领域术语多：医疗 / 法律 / 金融，通用模型不理解行业黑话
├─→ 检索 Recall 低：通用模型在你的数据集上效果差
├─→ 特殊语义关系：公司内部的缩写 / 代号 / 特定概念
└─→ 跨语言对齐：中英文档需要在同一语义空间
```

**微调的核心思路：**

```
通用 Embedding 模型的语义空间：

    技术 ──────────────────────────────────
    ↑    "React"  ·
    |        "Vue" ·          "心脏病" ·
    |   "JavaScript" ·    "高血压" ·
    |              "TypeScript" ·  "糖尿病" ·
    |                              "冠心病" ·
    ↓
    非技术 ────────────────────────────────

    医疗术语彼此接近，但没有 "主诉→诊断" 的方向关系

微调后（医疗场景）：

    "胸口疼" ────→ "冠心病"、"心肌梗塞"
    "头晕恶心" ──→ "高血压"、"脑供血不足"
    "血糖高" ───→ "糖尿病"、"胰岛素抵抗"

    微调让模型学会了 "症状→疾病" 的语义对应关系
```

**微调方法 — Contrastive Learning（对比学习）：**

```
训练数据格式（三元组）：

(anchor, positive, negative)
("如何优化 React 渲染",   "React 性能优化指南",    "Python 爬虫入门")
("胸口疼怎么办",         "冠心病的早期症状",      "感冒发烧吃什么药")

训练目标：
  让 anchor 与 positive 的距离更近
  让 anchor 与 negative 的距离更远

Loss Function: Triplet Loss / InfoNCE Loss
  L = max(0, d(anchor, positive) - d(anchor, negative) + margin)
```

**代码示例 — 使用 sentence-transformers 微调：**

```python
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses
)
from datasets import Dataset

# 1. 加载预训练模型
model = SentenceTransformer("BAAI/bge-large-zh-v1.5")

# 2. 准备训练数据
train_data = {
    "anchor": [
        "如何优化 React 性能",
        "前端状态管理方案",
        "CSS Grid 布局教程",
    ],
    "positive": [
        "React 渲染性能提升指南",
        "Redux vs Zustand 对比",
        "Grid 网格布局实战",
    ],
    "negative": [
        "Python 机器学习入门",
        "数据库索引优化",
        "Java Spring Boot 教程",
    ],
}
dataset = Dataset.from_dict(train_data)

# 3. 定义 Loss
loss = losses.TripletLoss(model=model)

# 4. 配置训练参数
args = SentenceTransformerTrainingArguments(
    output_dir="./finetuned-bge",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=True,
    eval_strategy="steps",
    eval_steps=100,
    save_steps=500,
    logging_steps=50,
)

# 5. 训练
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    loss=loss,
)
trainer.train()

# 6. 保存模型
model.save_pretrained("./finetuned-bge")

# 7. 使用微调后的模型
finetuned_model = SentenceTransformer("./finetuned-bge")
embeddings = finetuned_model.encode(["React 性能优化"])
```

**更高效的方案 — MatryoshkaLoss（俄罗斯套娃损失）：**

```python
from sentence_transformers import losses

# Matryoshka Loss 让模型在多个维度都能工作
# 比如 1024维、512维、256维、128维 都保持较好效果
# 部署时可以灵活选择维度，平衡精度和速度

matryoshka_loss = losses.MatryoshkaLoss(
    model=model,
    loss=losses.MultipleNegativesRankingLoss(model),
    matryoshka_dims=[1024, 512, 256, 128]
)
```

**微调效果评估：**

```python
from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator
)

# 准备评估数据
queries = {"q1": "如何优化首屏加载速度"}
corpus = {
    "d1": "前端性能优化：减少首屏白屏时间",
    "d2": "React 代码分割与懒加载",
    "d3": "数据库查询优化",
}
relevant_docs = {"q1": {"d1", "d2"}}  # q1 的正确答案

evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    name="custom-eval"
)

# 评估
results = evaluator(model)
print(f"MRR@10: {results['custom-eval_mrr@10']:.4f}")
print(f"NDCG@10: {results['custom-eval_ndcg@10']:.4f}")
print(f"Recall@10: {results['custom-eval_recall@10']:.4f}")
```

**微调 Checklist：**

| 步骤 | 要点 | 建议 |
|------|------|------|
| 数据准备 | 需要 (query, positive, negative) 三元组 | 至少 1000+ 对，越多越好 |
| 负样本策略 | Hard Negative 比随机负样本效果好 | 用 BM25 取 top-50 中非相关的作为 hard negative |
| 基础模型 | 选与目标语言匹配的模型 | 中文选 BGE-large-zh，多语言选 BGE-M3 |
| 学习率 | 太大会遗忘预训练知识 | 2e-5 ~ 5e-5 |
| 训练轮数 | 过多会过拟合 | 3-5 轮通常够 |
| 评估 | 在验证集上跑 MRR / NDCG / Recall | 对比微调前后的差异 |

**面试话术：**
> "通用 Embedding 模型在大多数场景够用了，微调主要针对领域特定的语义理解需求。我的做法是：先用通用模型做 baseline，评估检索的 Recall@10 和 MRR。如果效果不达标，再准备领域数据用 Contrastive Learning 微调。关键是 Hard Negative Mining——用 BM25 或通用模型检索出的 '看似相关但实际不相关' 的文档作为负样本，这比随机负样本效果好很多。"

---

## 4.7 多模态 Embedding

### Q: 什么是多模态 Embedding？怎么实现图文跨模态检索？

**多模态 Embedding = 把不同模态（文本、图片、音频、视频）映射到同一个向量空间。**

```
单模态 Embedding：
  文本 → 文本向量空间
  图片 → 图片向量空间
  ❌ 两个空间不互通，无法跨模态搜索

多模态 Embedding：
  文本 "一只在草地上奔跑的金毛犬" → [0.12, -0.34, ...] ─┐
                                                          ├→ 同一个向量空间！
  图片 [金毛犬在草地奔跑的照片]     → [0.11, -0.33, ...] ─┘

  → 文字搜图片 ✅  图片搜文字 ✅  图片搜图片 ✅
```

**主流多模态 Embedding 模型：**

| 模型 | 支持模态 | 维度 | 特点 |
|------|---------|------|------|
| **CLIP** (OpenAI) | 文本+图片 | 512/768 | 开山之作，生态最好 |
| **SigLIP** (Google) | 文本+图片 | 768/1152 | CLIP 改进版，Sigmoid Loss |
| **jina-clip-v2** (Jina) | 文本+图片 | 1024 | 多语言支持好，89种语言 |
| **ImageBind** (Meta) | 文本+图片+音频+视频+深度+热力 | 1024 | 6种模态统一空间 |
| **ONE-PEACE** (Alibaba) | 文本+图片+音频 | 1536 | 三模态统一 |
| **Sentence Transformers v3+** | 文本+图片 | 可变 | Python 生态集成好 |

**CLIP 的工作原理：**

```
训练阶段（对比学习）：

  Text Encoder                Image Encoder
  ┌──────────┐               ┌──────────┐
  │ "a dog"  │──→ T1         │  🐕 照片  │──→ I1
  │ "a cat"  │──→ T2         │  🐱 照片  │──→ I2
  │ "a car"  │──→ T3         │  🚗 照片  │──→ I3
  └──────────┘               └──────────┘

  Contrastive Matrix（对比矩阵）：
         I1    I2    I3
  T1   [✅高  ❌低  ❌低]    "a dog" 应该和狗的图片最近
  T2   [❌低  ✅高  ❌低]    "a cat" 应该和猫的图片最近
  T3   [❌低  ❌低  ✅高]    "a car" 应该和车的图片最近

  训练目标：最大化对角线（匹配对），最小化非对角线（非匹配对）
```

**代码示例 — 使用 CLIP 做图文检索：**

```python
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# 1. 加载模型
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# 2. 准备数据
images = [
    Image.open("dog.jpg"),
    Image.open("cat.jpg"),
    Image.open("car.jpg"),
]
texts = ["一只可爱的金毛犬", "一辆红色跑车", "日落时的海滩"]

# 3. 获取 Embedding
inputs = processor(
    text=texts,
    images=images,
    return_tensors="pt",
    padding=True
)

with torch.no_grad():
    outputs = model(**inputs)
    image_embeds = outputs.image_embeds   # (3, 768)
    text_embeds = outputs.text_embeds     # (3, 768)

# 4. 归一化
image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

# 5. 计算相似度矩阵
similarity = text_embeds @ image_embeds.T  # (3, 3)
print(similarity)
# tensor([[0.92, 0.15, 0.08],   # "金毛犬" 和 dog.jpg 最像
#         [0.11, 0.06, 0.89],   # "红色跑车" 和 car.jpg 最像
#         [0.05, 0.12, 0.18]])  # "海滩" 和所有图片都不太像

# 6. 文字搜图片
query = "一只在草地上的狗"
query_inputs = processor(text=[query], return_tensors="pt", padding=True)
with torch.no_grad():
    query_embed = model.get_text_features(**query_inputs)
    query_embed = query_embed / query_embed.norm(dim=-1, keepdim=True)

scores = (query_embed @ image_embeds.T).squeeze()
best_match = scores.argmax().item()
print(f"最匹配的图片索引: {best_match}")  # 0 (dog.jpg)
```

**使用 Sentence Transformers 的多模态 Embedding：**

```python
from sentence_transformers import SentenceTransformer
from PIL import Image

# 加载支持多模态的模型
model = SentenceTransformer("clip-ViT-L-14")

# 文本 Embedding
text_emb = model.encode(["一只金毛犬在草地上奔跑"])

# 图片 Embedding
img = Image.open("dog.jpg")
img_emb = model.encode([img])

# 计算跨模态相似度
from sentence_transformers.util import cos_sim
similarity = cos_sim(text_emb, img_emb)
print(f"文图相似度: {similarity.item():.4f}")
```

**多模态 Embedding 的应用场景：**

```
1. 电商以图搜图 / 以文搜图
   用户上传照片 → 找到相似商品
   用户描述 "红色连衣裙" → 检索商品图片

2. 内容审核
   上传图片 → 与违规内容向量库比对

3. 多模态 RAG
   知识库包含文档 + 图表 + 流程图
   → 用户提问时可以检索到相关图表

4. 视频搜索
   对视频逐帧提取 Embedding → 用文字搜索视频片段
```

**前端类比：**

```
多模态 Embedding ≈ CSS 的 "统一单位系统"

不同模态 = 不同 CSS 单位：
  文本 = px（像素）
  图片 = rem（相对单位）
  音频 = vw（视口单位）

多模态 Embedding 就像 calc() 一样，
把所有单位转换到同一个坐标系下进行计算：
  calc(16px + 1rem + 2vw) → 统一成像素值
  CLIP(文本 + 图片)       → 统一成向量
```

**面试话术：**
> "多模态 Embedding 的核心是通过 Contrastive Learning，把不同模态的数据映射到同一个向量空间。CLIP 是这个领域的开山之作，它用 4 亿个图文对训练，让文本和图片共享语义空间。实际应用中，我会选择 jina-clip-v2 做中文场景的图文检索，因为它对中文支持更好。如果需要更多模态，Meta 的 ImageBind 支持 6 种模态的统一表示。"

---

## 4.8 向量数据库选型决策

### Q: 给你一个新项目，你会怎么选向量数据库？

**完整的选型决策树：**

```
项目阶段？
│
├─→ 原型验证 / Demo / PoC
│   └─→ Chroma（pip install 一行搞定，本地运行）
│       或 FAISS（纯计算，最快上手）
│
├─→ MVP / 小型项目（< 100 万向量）
│   │
│   ├─→ 已有 PostgreSQL？
│   │   └─→ pgvector（零额外基础设施，SQL 原生查询）
│   │
│   ├─→ 需要复杂过滤 + 向量搜索？
│   │   └─→ Qdrant（过滤性能最佳）
│   │
│   └─→ 不想运维？
│       └─→ Pinecone Free Tier 或 Qdrant Cloud
│
├─→ 生产环境（100 万 ~ 1 亿向量）
│   │
│   ├─→ 有 DevOps 团队？
│   │   ├─→ 性能优先 → Qdrant（Rust，轻量高性能）
│   │   └─→ 功能全面 → Milvus（分布式，功能丰富）
│   │
│   ├─→ 没有运维能力？
│   │   ├─→ 预算充足 → Pinecone（全托管）
│   │   ├─→ 预算有限 → Zilliz Cloud（Milvus 托管版）
│   │   └─→ 最省钱 → Qdrant Cloud Free Tier + 扩容
│   │
│   └─→ 需要混合搜索（向量 + 全文检索）？
│       ├─→ Weaviate（内置 BM25 + Vector）
│       └─→ Elasticsearch 8.x（原生向量搜索支持）
│
└─→ 大规模生产（> 1 亿向量）
    │
    ├─→ 自建 → Milvus 集群（分布式原生，久经考验）
    ├─→ 托管 → Zilliz Cloud / Pinecone Enterprise
    └─→ 已有 Elasticsearch → ES 8.x 向量搜索
```

**按团队背景的推荐：**

```
前端 / 全栈工程师（你）：
  ├─→ 入门学习 → Chroma（Python API 简单，LangChain 教程多）
  ├─→ 个人项目 → pgvector（Supabase 一键开启）
  ├─→ 公司项目 → Qdrant（性能好，API 设计直觉，Docker 一键部署）
  └─→ 大厂项目 → 听后端架构师的 :)

后端 / 架构师：
  ├─→ 需要评估 → Qdrant vs Milvus 做 Benchmark
  ├─→ 已有 PG → 先试 pgvector，不够再迁移
  └─→ 十亿级别 → Milvus 分布式集群
```

**一个完整的技术选型评估模板：**

| 评估维度 | 权重 | Qdrant | Milvus | pgvector | Pinecone |
|---------|------|--------|--------|----------|----------|
| 查询性能 (QPS) | 25% | 9 | 8 | 6 | 7 |
| 部署复杂度 | 20% | 8 | 5 | 9 | 10 |
| 运维成本 | 15% | 7 | 5 | 8 | 10 |
| 功能丰富度 | 15% | 8 | 9 | 7 | 7 |
| 社区生态 | 10% | 7 | 8 | 9 | 6 |
| 价格 | 10% | 9 | 9 | 10 | 4 |
| 扩展性 | 5% | 7 | 9 | 5 | 9 |
| **加权总分** | 100% | **8.05** | **7.15** | **7.55** | **7.70** |

> 注：评分仅供参考，实际项目中需要结合具体场景调整权重。

**实际项目中的架构示例：**

```
典型的 RAG 应用架构：

┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   前端 UI     │────→│   API 网关    │────→│  RAG Service │
│  (Next.js)   │     │   (Nginx)    │     │  (FastAPI)   │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                     ┌────────────────────────────┤
                     │                            │
              ┌──────▼───────┐           ┌───────▼────────┐
              │ Vector DB     │           │   LLM API      │
              │ (Qdrant)      │           │  (OpenAI /     │
              │               │           │   本地 Ollama)  │
              │ - 文档 chunks  │           └────────────────┘
              │ - Embedding   │
              │ - Metadata    │
              └──────┬────────┘
                     │
              ┌──────▼────────┐
              │  PostgreSQL    │
              │  (业务数据)     │
              │  - 用户信息     │
              │  - 对话历史     │
              │  - 文档管理     │
              └───────────────┘

前端工程师在这个架构中的角色：
1. 构建对话 UI（流式输出 SSE）
2. 实现文件上传（拖拽上传、预览、分 chunk 显示）
3. 搜索结果展示（高亮、来源引用、相关度排序）
4. 调用 RAG API（传入 query，展示检索结果 + LLM 回答）
```

**代码示例 — 完整的文档检索 Pipeline：**

```python
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# 初始化
openai_client = OpenAI()
qdrant = QdrantClient(host="localhost", port=6333)

# 1. 创建 Collection
qdrant.recreate_collection(
    collection_name="docs",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)

# 2. 文档入库
def index_documents(documents: list[dict]):
    """将文档列表转为向量并存入 Qdrant"""
    texts = [doc["content"] for doc in documents]

    # 批量生成 Embedding
    response = openai_client.embeddings.create(
        input=texts,
        model="text-embedding-3-large",
        dimensions=1536
    )

    points = [
        PointStruct(
            id=i,
            vector=response.data[i].embedding,
            payload={
                "content": doc["content"],
                "title": doc.get("title", ""),
                "source": doc.get("source", ""),
            }
        )
        for i, doc in enumerate(documents)
    ]

    qdrant.upsert(collection_name="docs", points=points)
    print(f"已索引 {len(points)} 个文档块")

# 3. 语义搜索
def semantic_search(query: str, top_k: int = 5) -> list[dict]:
    """语义搜索，返回最相关的文档"""
    # 生成查询向量
    response = openai_client.embeddings.create(
        input=query,
        model="text-embedding-3-large",
        dimensions=1536
    )
    query_vector = response.data[0].embedding

    # 搜索
    results = qdrant.search(
        collection_name="docs",
        query_vector=query_vector,
        limit=top_k
    )

    return [
        {
            "content": r.payload["content"],
            "title": r.payload["title"],
            "score": r.score,
        }
        for r in results
    ]

# 4. RAG：检索 + 生成
def rag_query(question: str) -> str:
    """RAG: 检索相关文档 + LLM 生成回答"""
    # 检索
    docs = semantic_search(question, top_k=3)
    context = "\n\n".join([
        f"[来源: {d['title']}] (相关度: {d['score']:.2f})\n{d['content']}"
        for d in docs
    ])

    # 生成
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"根据以下参考资料回答用户问题。\n\n{context}"},
            {"role": "user", "content": question}
        ]
    )

    return response.choices[0].message.content

# 使用
documents = [
    {"title": "React 性能优化", "content": "使用 React.memo 和 useMemo 减少不必要的重渲染..."},
    {"title": "Vue 3 Composition API", "content": "Composition API 提供了更灵活的逻辑复用方式..."},
    {"title": "Next.js SSR", "content": "Server-Side Rendering 可以提升首屏加载速度..."},
]

index_documents(documents)
answer = rag_query("如何优化前端应用的首屏加载速度？")
print(answer)
```

**数据迁移注意事项：**

```
从 A 向量库迁移到 B：

  1. Embedding 模型必须一致！
     ❌ 用 text-embedding-ada-002 生成的向量 → 不能直接导入用 BGE 的库
     ✅ 需要用同一个模型重新生成所有向量

  2. 相似度度量必须一致！
     ❌ 原来用 Cosine，新库用 Euclidean → 结果会不对
     ✅ 保持相同的 Distance 设置

  3. 向量维度必须一致！
     ❌ 原来 1536 维，新库配置 768 维 → 直接报错
     ✅ Collection 配置相同维度

  4. 导出数据格式：
     大多数向量库支持 JSON / Parquet 导出
     建议同时导出 payload（元数据）
```

**面试话术：**
> "向量数据库选型是个典型的架构决策，没有银弹。我的方法论是：先确定约束条件（数据规模、预算、运维能力、现有技术栈），然后用加权评分模型做客观评估，最后在候选方案上做 Benchmark（重点看 QPS、Recall@10、P99 延迟）。我之前项目的经验是：初期用 Chroma 或 pgvector 快速验证，确认 RAG 方案可行后，再迁移到 Qdrant 或 Milvus 做生产部署。记住一点：Embedding 模型的选择比向量数据库的选择更重要——模型决定了检索质量的天花板，数据库只是决定了性能和运维成本。"

---

[← 上一章：03 - Prompt Engineering](./03-prompt-engineering.md) | [下一章：05 - RAG 检索增强生成 →](./05-rag.md)