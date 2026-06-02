# 12 - 生产部署与运维

> **难度：** ⭐⭐⭐⭐⭐ | **定位：** 从"Demo 跑通"到"百万用户稳定运行"的最后一公里，高级 AI 工程师必考
>
> **前端类比：** 如果前面的章节是"写代码"，本章就是"上线运维"——部署策略 ≈ Vercel vs 自建服务器，成本优化 ≈ CDN + 图片压缩 + 代码分割，可观测性 ≈ Sentry + DataDog + Lighthouse，系统设计 ≈ 前端八股里的"设计一个 XX 系统"。

## 本章知识树

```
生产部署与运维
├── 12.1 模型部署策略（云 API / 自部署 / 混合）
├── 12.2 成本优化（模型路由、语义缓存、Token 压缩、批处理）
├── 12.3 可观测性体系（Tracing / Metrics / Logging）
├── 12.4 Agent 监控与 SLA 设计
├── 12.5 AI 系统设计：百万 DAU 客服系统
├── 12.6 AI 系统设计：企业知识库 RAG 平台
├── 12.7 AI 系统设计：LLM API Gateway
└── 12.8 AI 系统设计：AI 内容审核系统
```

---

## 12.1 模型部署策略

### Q: 云 API、自部署、混合部署三种方案怎么选？各有什么优缺点？

**没有万能的部署方案——选型取决于数据安全、成本、延迟、可控性四个维度。云 API 适合快速启动，自部署适合数据敏感场景，混合部署是大多数企业的最终选择。**

**三种部署策略对比：**

| 维度 | 云 API（OpenAI / Claude） | 自部署（vLLM / Ollama） | 混合部署 |
|------|-------------------------|----------------------|---------|
| **启动成本** | $0（按量付费） | $10K-100K+（GPU 服务器） | 中等 |
| **单次成本** | 较高（按 Token 计费） | 低（固定硬件成本均摊） | 最优 |
| **数据安全** | 数据出境风险 | ✅ 数据不出内网 | 敏感走内网，普通走云 |
| **延迟** | 50-200ms（网络 + 排队） | 10-50ms（本地推理） | 按需路由 |
| **模型选择** | GPT-4o / Claude / Gemini | 开源模型（Llama / Qwen） | 两者都有 |
| **运维复杂度** | ⭐ 零运维 | ⭐⭐⭐⭐⭐ GPU 运维 | ⭐⭐⭐ |
| **扩展性** | ⭐⭐⭐⭐⭐ 自动扩容 | ⭐⭐ 需手动扩容 | ⭐⭐⭐⭐ |
| **前端类比** | Vercel / Netlify 托管 | 自建 K8s 集群 | 混合部署 |

**混合部署架构（生产推荐）：**

```
┌──────────────────────────────────────────────────────┐
│                   AI Gateway（统一入口）               │
│                                                      │
│  ┌─────────────────────────────────────────────────┐ │
│  │              路由层（Model Router）                │ │
│  │                                                 │ │
│  │  规则 1: 数据包含 PII → 内网自部署模型             │ │
│  │  规则 2: 简单问题（分类/提取）→ 小模型（4o-mini）   │ │
│  │  规则 3: 复杂推理 → GPT-4o / Claude               │ │
│  │  规则 4: 代码生成 → Claude Sonnet                  │ │
│  └──────────┬──────────────────┬────────────────────┘ │
│             │                  │                      │
│      ┌──────▼──────┐   ┌──────▼──────┐               │
│      │  自部署集群   │   │   云 API     │               │
│      │  vLLM +      │   │  OpenAI     │               │
│      │  Qwen-72B    │   │  Claude     │               │
│      │  (敏感数据)   │   │  Gemini     │               │
│      └─────────────┘   └─────────────┘               │
└──────────────────────────────────────────────────────┘
```

**自部署技术栈选型：**

| 工具 | 定位 | 适合场景 |
|------|------|---------|
| **vLLM** | 高性能推理引擎（PagedAttention） | 生产环境、高吞吐 |
| **Ollama** | 本地模型管理工具（一键下载运行） | 开发调试、个人使用 |
| **TGI** | HuggingFace 推理服务 | HF 生态集成 |
| **SGLang** | 高性能推理（RadixAttention） | 复杂 prompt 复用场景 |
| **Triton** | NVIDIA 推理服务器 | 多模型混合部署 |

**面试话术：**

> "部署策略选型看四个维度：数据安全、成本、延迟、运维复杂度。大多数企业最终走混合部署——敏感数据走内网自部署（vLLM + 开源模型），非敏感走云 API（GPT-4o / Claude）。中间加一层 AI Gateway 做路由、限流、降级。这就像前端的 CDN 策略——静态资源走 CDN，动态请求回源。"

---

## 12.2 成本优化

### Q: LLM 应用的 Token 成本太高，有哪些优化策略？

**LLM 成本优化的核心思想：能不调就不调（缓存），必须调就用小的（路由），调了就压短（压缩），多条一起调（批处理）。四板斧可以砍掉 60-80% 的成本。**

**成本优化四大策略：**

```
成本优化金字塔（从上到下优先级递减）：

          ╱╲
         ╱  ╲     语义缓存
        ╱ $0 ╲    命中缓存 → 不调 LLM → 成本为 0
       ╱──────╲
      ╱        ╲   模型路由
     ╱  $0.15   ╲  简单问题 → 小模型（成本 1/30）
    ╱────────────╲
   ╱              ╲  Prompt 压缩
  ╱    $0.50       ╲ 压缩 Token 数量 → 减少 30-50%
 ╱──────────────────╲
╱                    ╲ 批处理
╱      $1.00          ╲ 批量请求 → 享受 50% 折扣
╱──────────────────────╲
```

**策略 1：模型路由（简单→复杂分流，省 70%）**

```python
import re
from enum import Enum

class TaskComplexity(Enum):
    SIMPLE = "simple"      # 分类、提取、简单 QA
    MEDIUM = "medium"      # 摘要、翻译、一般对话
    COMPLEX = "complex"    # 推理、代码生成、多步规划

# 模型路由表
MODEL_ROUTER = {
    TaskComplexity.SIMPLE:  {"model": "gpt-4o-mini",  "cost_per_1k": 0.15},   # 定价以官网为准，随时变化
    TaskComplexity.MEDIUM:  {"model": "gpt-4o",       "cost_per_1k": 2.50},   # 定价以官网为准，随时变化
    TaskComplexity.COMPLEX: {"model": "claude-sonnet", "cost_per_1k": 3.00},   # 定价以官网为准，随时变化
}

async def classify_complexity(query: str) -> TaskComplexity:
    """用规则 + 小模型判断任务复杂度"""
    # 第一层：规则快速判断
    simple_patterns = [
        r"分类|归类|提取|是什么|翻译成",
        r"classify|extract|translate|what is",
    ]
    for pattern in simple_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return TaskComplexity.SIMPLE

    # 第二层：小模型判断
    response = await mini_llm.ainvoke(
        f"判断以下问题的复杂度(simple/medium/complex):\n{query}"
    )
    return TaskComplexity(response.content.strip())

async def smart_llm_call(query: str, messages: list) -> str:
    """智能路由调用"""
    complexity = await classify_complexity(query)
    config = MODEL_ROUTER[complexity]

    response = await client.chat.completions.create(
        model=config["model"],
        messages=messages,
    )
    return response.choices[0].message.content
```

**策略 2：语义缓存（相似问题直接返回，省 100%）**

```python
import numpy as np
from redis import asyncio as aioredis

class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.92):
        self.threshold = similarity_threshold
        self.redis = aioredis.from_url("redis://localhost:6379")

    async def get(self, query: str) -> str | None:
        """语义匹配缓存"""
        query_embedding = await get_embedding(query)

        # 从 Redis 获取所有缓存的 embedding
        cached_keys = await self.redis.keys("cache:*")

        for key in cached_keys:
            cached_data = await self.redis.hgetall(key)
            cached_embedding = np.frombuffer(cached_data[b"embedding"])

            # 余弦相似度
            similarity = np.dot(query_embedding, cached_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(cached_embedding)
            )

            if similarity >= self.threshold:
                return cached_data[b"response"].decode()

        return None

    async def set(self, query: str, response: str, ttl: int = 3600):
        """缓存响应"""
        embedding = await get_embedding(query)
        key = f"cache:{hash(query)}"
        await self.redis.hset(key, mapping={
            "embedding": embedding.tobytes(),
            "response": response,
            "query": query,
        })
        await self.redis.expire(key, ttl)
```

**策略 3：Prompt 压缩（减少 Token，省 30-50%）**

| 方法 | 描述 | Token 节省 |
|------|------|-----------|
| **LLMLingua** | 微软的 Prompt 压缩工具，去掉冗余 Token | 30-50% |
| **摘要代替全文** | 先让小模型摘要，再把摘要给大模型 | 50-70% |
| **动态 Context** | 只送最相关的 Top-3 而不是 Top-10 | 60% |
| **System Prompt 缓存** | OpenAI Prompt Caching（相同前缀缓存） | 50%（缓存命中时） |

**策略 4：批处理（Batch API，省 50%）**

```python
# OpenAI Batch API —— 24 小时内完成，价格减半
from openai import OpenAI

client = OpenAI()

# 1. 准备批量请求（JSONL 格式）
requests = [
    {"custom_id": f"req-{i}", "method": "POST", "url": "/v1/chat/completions",
     "body": {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": f"总结第{i}段"}]}}
    for i in range(1000)
]

# 2. 上传文件
batch_file = client.files.create(file=open("requests.jsonl", "rb"), purpose="batch")

# 3. 创建批处理任务
batch = client.batches.create(
    input_file_id=batch_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",  # 24 小时内完成
)
# 成本：比实时 API 便宜 50%！
```

**成本优化效果总览：**

| 策略 | 成本节省 | 延迟影响 | 实现复杂度 |
|------|---------|---------|-----------|
| 语义缓存 | 100%（命中时） | 更低 | ⭐⭐ |
| 模型路由 | 60-70% | 无 | ⭐⭐ |
| Prompt 压缩 | 30-50% | 略增 | ⭐⭐⭐ |
| Batch API | 50% | 大幅增加（异步） | ⭐ |
| 组合使用 | **60-80%** | - | - |

**面试话术：**

> "LLM 成本优化四板斧：第一，语义缓存——相似问题直接返回缓存，成本归零；第二，模型路由——简单任务用 4o-mini（成本只有 GPT-4o 的 1/16），复杂任务才用大模型；第三，Prompt 压缩——用 LLMLingua 去掉冗余 Token，或只送 Top-3 Context；第四，批处理——非实时任务用 Batch API 省 50%。四个加起来可以砍掉 60-80% 的成本。"

---

## 12.3 可观测性体系

### Q: AI 应用的可观测性和传统应用有什么不同？需要监控哪些指标？

**AI 应用的可观测性 = 传统可观测性 + LLM 特有指标。传统应用只需监控延迟和错误率，AI 应用还要监控 Token 消耗、输出质量、幻觉率、检索相关性等"AI 指标"。三大支柱不变：Tracing、Metrics、Logging。**

**AI 可观测性三大支柱：**

```
┌─────────────────────────────────────────────────────────┐
│                  AI Observability 体系                    │
│                                                         │
│  Tracing（链路追踪）                                     │
│  ├── 用户请求 → Retrieval → Rerank → LLM → 后处理       │
│  ├── 每一步的输入/输出/耗时/Token 数                      │
│  └── 工具：LangSmith / Arize Phoenix / Langfuse          │
│                                                         │
│  Metrics（指标监控）                                      │
│  ├── 传统指标：延迟 / 吞吐 / 错误率 / 可用性              │
│  ├── AI 指标：Token 使用量 / 成本 / 缓存命中率            │
│  ├── 质量指标：幻觉率 / 相关性 / 用户满意度               │
│  └── 工具：Prometheus + Grafana / DataDog                │
│                                                         │
│  Logging（日志记录）                                      │
│  ├── 请求/响应日志（脱敏后）                               │
│  ├── Prompt 版本追踪                                     │
│  └── 工具：ELK / Loki                                   │
└─────────────────────────────────────────────────────────┘
```

**AI 应用必须监控的指标：**

| 类别 | 指标 | 告警阈值参考 | 工具 |
|------|------|------------|------|
| **延迟** | P50 / P95 / P99 TTFT（首 Token 时间） | P95 > 3s 告警 | Prometheus |
| **延迟** | P50 / P95 端到端延迟 | P95 > 10s 告警 | Prometheus |
| **吞吐** | QPS / 并发数 | 接近限流阈值告警 | Grafana |
| **错误** | 5xx 率 / LLM 429 限流率 | > 1% 告警 | PagerDuty |
| **Token** | 每请求平均 Token 数 | 突增 50% 告警 | 自定义 |
| **成本** | 每日/每小时 Token 费用 | 超预算告警 | Billing API |
| **缓存** | 语义缓存命中率 | < 20% 告警（缓存失效？） | Redis metrics |
| **质量** | 用户反馈（点赞/点踩率） | 点踩率 > 10% 告警 | 自定义 |
| **质量** | 幻觉率（抽样评估） | > 5% 告警 | LangSmith Eval |
| **RAG** | 检索相关性分数 | 平均 < 0.7 告警 | Arize Phoenix |

**LangSmith 集成（推荐的 AI 可观测性平台）：**

```python
# 1. 环境变量配置
import os
os.environ["LANGSMITH_API_KEY"] = "ls_..."
os.environ["LANGSMITH_PROJECT"] = "my-ai-app"
os.environ["LANGSMITH_TRACING"] = "true"

# 2. 自动追踪（LangChain 自动集成）
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 只要设了环境变量，所有 LangChain 调用自动上报 Trace
chain = ChatPromptTemplate.from_template("总结：{text}") | ChatOpenAI()
result = chain.invoke({"text": "..."})  # 自动记录到 LangSmith

# 3. 手动追踪（非 LangChain 代码）
from langsmith import traceable

@traceable(run_type="chain", name="my_rag_pipeline")
async def rag_query(question: str) -> str:
    docs = await retrieve(question)      # 自动记录子 Span
    context = format_context(docs)
    answer = await llm_generate(context, question)
    return answer

# 4. 用户反馈关联
from langsmith import Client

ls_client = Client()
ls_client.create_feedback(
    run_id="run-xxx",          # Trace ID
    key="user_rating",
    score=1,                   # 1=点赞, 0=点踩
    comment="回答很准确",
)
```

**OpenTelemetry 集成（通用方案）：**

```python
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider

tracer = trace.get_tracer("ai-app")
meter = metrics.get_meter("ai-app")

# 自定义 AI 指标
token_counter = meter.create_counter("llm.tokens.total", description="Total tokens used")
latency_hist = meter.create_histogram("llm.latency", description="LLM call latency")
cache_hit_counter = meter.create_counter("cache.hits", description="Semantic cache hits")

@tracer.start_as_current_span("llm_call")
async def traced_llm_call(messages: list) -> str:
    span = trace.get_current_span()
    span.set_attribute("llm.model", "gpt-4o")
    span.set_attribute("llm.prompt_tokens", len(str(messages)))

    start = time.time()
    response = await client.chat.completions.create(model="gpt-4o", messages=messages)

    # 记录指标
    latency_hist.record(time.time() - start)
    token_counter.add(response.usage.total_tokens, {"model": "gpt-4o"})

    span.set_attribute("llm.completion_tokens", response.usage.completion_tokens)
    return response.choices[0].message.content
```

**面试话术：**

> "AI 可观测性在传统三大支柱（Tracing/Metrics/Logging）基础上，增加了 AI 特有指标：Token 消耗、成本追踪、输出质量（幻觉率、相关性）、缓存命中率。推荐 LangSmith 做 AI 链路追踪（自动记录每一步的输入输出和耗时），Prometheus + Grafana 做指标监控，OpenTelemetry 做通用 Trace 集成。"

---

## 12.4 Agent 监控与 SLA 设计

### Q: Agent 的 SLA 怎么定义？和传统 API 的 SLA 有什么区别？

**Agent SLA 的核心挑战：Agent 执行步骤不确定（可能 2 步完成，也可能 20 步），每一步都可能调用不同的 Tool，延迟方差极大。不能像传统 API 一样简单定义"P99 < 500ms"，需要多维度 SLA。**

**Agent SLA vs 传统 API SLA：**

| 维度 | 传统 API SLA | Agent SLA |
|------|-------------|-----------|
| **延迟** | P99 < 500ms（确定性） | P95 < 30s（高方差） |
| **步骤数** | 固定（1 步） | 不确定（1-N 步，需设上限） |
| **成本** | 固定（1 次 DB 查询） | 不确定（N 次 LLM 调用） |
| **正确性** | 精确（查数据库） | 概率性（LLM 可能出错） |
| **可用性** | 99.99% | 99.9%（依赖多个外部服务） |

**Agent SLA 设计框架：**

```
┌─────────────────────────────────────────────────┐
│              Agent SLA 四维指标                    │
│                                                 │
│  1. 可用性（Availability）                        │
│     └── 目标：99.9%（月停机 < 43 分钟）            │
│         降级策略：Agent 不可用时切换到规则引擎       │
│                                                 │
│  2. 延迟（Latency）                               │
│     ├── TTFT（首 Token 时间）：P95 < 2s            │
│     ├── 端到端完成时间：P95 < 30s                  │
│     └── 超时熔断：单步 > 10s 或 总计 > 60s 中止     │
│                                                 │
│  3. 质量（Quality）                               │
│     ├── 任务完成率：> 85%                          │
│     ├── 幻觉率：< 3%                              │
│     └── 用户满意度：点赞率 > 80%                    │
│                                                 │
│  4. 成本（Cost）                                  │
│     ├── 单次 Agent 调用平均成本 < $0.10             │
│     ├── 单次最大步骤数 ≤ 10（防止无限循环）          │
│     └── 单次最大 Token ≤ 50K                       │
└─────────────────────────────────────────────────┘
```

**Agent 监控必须实现的安全机制：**

```python
import asyncio
import time

class AgentGuardrails:
    """Agent 运行时防护"""

    def __init__(
        self,
        max_steps: int = 10,            # 最大步骤数
        max_time: float = 60.0,         # 最大运行时间（秒）
        max_tokens: int = 50_000,       # 最大 Token 消耗
        max_cost: float = 0.10,         # 最大成本（美元）
        step_timeout: float = 10.0,     # 单步超时
    ):
        self.max_steps = max_steps
        self.max_time = max_time
        self.max_tokens = max_tokens
        self.max_cost = max_cost
        self.step_timeout = step_timeout

        # 运行时计数器
        self.steps = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.start_time = time.time()

    def check(self) -> str | None:
        """每一步之前检查——返回 None 表示可以继续，否则返回中止原因"""
        if self.steps >= self.max_steps:
            return f"exceeded max steps ({self.max_steps})"
        if time.time() - self.start_time > self.max_time:
            return f"exceeded max time ({self.max_time}s)"
        if self.total_tokens > self.max_tokens:
            return f"exceeded max tokens ({self.max_tokens})"
        if self.total_cost > self.max_cost:
            return f"exceeded max cost (${self.max_cost})"
        return None

    async def execute_step(self, step_fn, *args):
        """带超时和防护的单步执行"""
        reason = self.check()
        if reason:
            raise AgentTerminated(reason)

        self.steps += 1
        result = await asyncio.wait_for(step_fn(*args), timeout=self.step_timeout)
        return result
```

**面试话术：**

> "Agent SLA 和传统 API 最大的区别是'不确定性'——步骤数不确定、延迟方差大、成本不可控。所以 Agent SLA 要四维定义：可用性（99.9% + 降级兜底）、延迟（TTFT P95 < 2s，端到端 P95 < 30s）、质量（完成率 > 85%，幻觉率 < 3%）、成本（单次 < $0.10）。同时必须实现 Guardrails——最大步骤数、最大运行时间、最大 Token 消耗的硬限制，防止 Agent 无限循环烧钱。"

---

## 12.5 AI 系统设计：百万 DAU 客服系统

### Q: 设计一个支持百万 DAU 的 AI 客服系统，要求首响时间 < 2s、成本可控。

**核心思路：三层分流（FAQ → RAG → Agent → 人工），90% 的请求在前两层解决，只有 10% 进入 Agent/人工。**

**系统架构图：**

```
┌──────────────────────────────────────────────────────────────────┐
│                        用户接入层                                 │
│  Web / App / 微信 / 飞书 / API                                   │
│  ├── WebSocket 长连接管理                                         │
│  └── 负载均衡（Nginx / ALB）                                      │
└────────────────────────┬─────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│                      智能分流层                                    │
│                                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐      │
│  │ L1: FAQ  │──▶│ L2: RAG  │──▶│ L3: Agent│──▶│ L4: 人工  │      │
│  │ 精准匹配  │   │ 知识检索  │   │ 多轮对话  │   │ 复杂/投诉 │      │
│  │ ~50%     │   │ ~30%     │   │ ~15%     │   │ ~5%      │      │
│  │ <100ms   │   │ <1s      │   │ <5s      │   │ 排队等待  │      │
│  │ 成本: $0 │   │ 成本: 低  │   │ 成本: 中  │   │ 成本: 高  │      │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘      │
└──────────────────────────────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│                      基础设施层                                    │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ Redis       │  │ PostgreSQL  │  │ Milvus      │              │
│  │ ·FAQ 缓存   │  │ ·对话历史   │  │ ·知识向量库  │              │
│  │ ·语义缓存   │  │ ·用户画像   │  │ ·FAQ 向量   │              │
│  │ ·Session    │  │ ·工单系统   │  │             │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐             │
│  │ Kafka       │  │ Prometheus  │  │ LangSmith    │             │
│  │ ·消息队列   │  │ + Grafana   │  │ ·LLM Trace   │             │
│  │ ·异步处理   │  │ ·监控告警   │  │ ·质量评估    │             │
│  └─────────────┘  └─────────────┘  └──────────────┘             │
└──────────────────────────────────────────────────────────────────┘
```

**容量估算（百万 DAU）：**

| 指标 | 数值 | 计算 |
|------|------|------|
| DAU | 1,000,000 | 需求 |
| 日均对话数 | 200,000 | 20% 用户发起客服对话 |
| 峰值 QPS | ~50 | 200K / 86400 * 20（峰值倍数） |
| 日均 LLM 调用 | 100,000 | 50% FAQ 命中 + 30% RAG（各 1 次 LLM）+ 15% Agent（平均 3 次 LLM） |
| 日均 Token | ~50M | 平均每次 500 Token |
| 日均成本 | ~$75 | 70% 用 4o-mini ($0.15/1K) + 30% 用 4o ($2.5/1K) |

**关键设计决策：**

| 决策点 | 选择 | 原因 |
|--------|------|------|
| L1 FAQ 匹配 | 向量相似度 + BM25 混合 | 精准召回，语义泛化 |
| L2 RAG 模型 | GPT-4o-mini | 知识检索场景够用，成本 1/16 |
| L3 Agent 模型 | GPT-4o（降级到 4o-mini） | 多轮推理需要强模型 |
| 缓存策略 | 语义缓存 + FAQ 精确缓存 | 高频问题命中率 > 50% |
| 转人工触发 | 情绪检测 angry + 3 轮未解决 | 防止用户流失 |

**面试话术：**

> "百万 DAU 客服系统的核心是四层分流：L1 FAQ 精确匹配（50%请求，<100ms，成本为零），L2 RAG 知识检索（30%，用小模型 + 语义缓存），L3 Agent 多轮对话（15%，用大模型但加 Guardrails 控制成本），L4 转人工（5%，愤怒用户或 Agent 3 轮解决不了自动转接）。通过三层分流把 95% 的请求用低成本方案解决，日均成本控制在 $75 左右。"

---

## 12.6 AI 系统设计：企业知识库 RAG 平台

### Q: 设计一个企业级 RAG 知识库平台，要求支持多数据源、权限隔离、持续更新。

**企业 RAG 的三大挑战：数据源多样（PDF/Confluence/数据库/代码仓库）、权限复杂（部门隔离 + 角色控制）、知识时效性（文档更新后索引要同步）。**

**系统架构图：**

```
┌───────────────────────────────────────────────────────────────┐
│                        接入层                                  │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐         │
│  │ Web UI  │  │ Slack   │  │ 飞书Bot  │  │  API    │         │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘         │
└───────┼────────────┼────────────┼────────────┼───────────────┘
        │            │            │            │
┌───────▼────────────▼────────────▼────────────▼───────────────┐
│                      查询处理层                                │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │ 权限校验  │→│ Query    │→│ 混合检索  │→│ Rerank   │     │
│  │ (RBAC)   │  │ 改写/扩展 │  │ 向量+BM25│  │ 精排     │     │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘     │
│                                                    │         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐         │         │
│  │ Context  │←│ 权限过滤  │←│ 结果融合  │←────────┘         │
│  │ 组装+生成 │  │ (行级)   │  │ (RRF)    │                   │
│  └──────────┘  └──────────┘  └──────────┘                   │
└──────────────────────────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────┐
│                     数据处理层（离线）                          │
│                                                              │
│  数据源连接器                          索引管道                 │
│  ┌──────────┐  ┌──────────┐    ┌──────────┐  ┌──────────┐   │
│  │Confluence│  │ Notion   │    │ 文档解析  │→│ 智能分块  │   │
│  │ PDF/Word │  │ GitHub   │───▶│ (Unstructured)│ (语义分块) │   │
│  │ Database │  │ Google   │    └──────────┘  └─────┬────┘   │
│  └──────────┘  │ Drive    │                        │         │
│                └──────────┘                        ▼         │
│                                            ┌──────────┐      │
│  增量更新                                   │ Embedding│      │
│  ┌──────────────────────────┐              │ + 权限标签│      │
│  │ Webhook / 定时扫描        │              └─────┬────┘      │
│  │ → 变更检测（哈希比对）     │                    │           │
│  │ → 增量重索引              │                    ▼           │
│  └──────────────────────────┘              ┌──────────┐      │
│                                            │ Milvus / │      │
│                                            │ Qdrant   │      │
│                                            └──────────┘      │
└──────────────────────────────────────────────────────────────┘
```

**权限隔离——四层架构：**

| 层级 | 策略 | 实现 |
|------|------|------|
| **数据源层** | 只同步有权限的文档 | OAuth 授权 + API 权限继承 |
| **索引层** | 每个 Chunk 打权限标签 | `metadata: {department: "engineering", role: ["admin", "dev"]}` |
| **检索层** | 查询时带权限过滤 | `filter: {"department": user.dept, "role": {"$in": user.roles}}` |
| **展示层** | 脱敏 + 链接回源文档 | 敏感字段 mask，引用链接检查权限 |

```python
# 检索时的权限过滤（Milvus 示例）
async def search_with_permission(query: str, user: User) -> list[Document]:
    query_embedding = await embed(query)

    # 权限过滤表达式
    permission_filter = (
        f'department == "{user.department}" '
        f'or access_level == "public" '
        f'or "{user.role}" in allowed_roles'
    )

    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=20,
        expr=permission_filter,  # 向量搜索 + 权限过滤同时执行
        output_fields=["content", "source", "department"],
    )
    return results
```

**增量更新策略：**

```
全量重建 vs 增量更新：

全量重建（简单但慢）：
  每天凌晨：删除所有索引 → 重新抓取 → 重新 Embedding → 重新索引
  耗时：10 万文档 ≈ 2-4 小时
  问题：索引期间数据不一致

增量更新（推荐）：
  Webhook 触发 / 定时扫描 → 文档哈希比对 → 仅处理变更文档
  ├── 新增文档 → 解析 + 分块 + Embedding + 插入
  ├── 修改文档 → 删除旧 Chunk → 重新处理 + 插入
  └── 删除文档 → 删除对应 Chunk
  耗时：变更量通常 < 1%，秒级完成
```

**面试话术：**

> "企业 RAG 平台的核心挑战是三个：多数据源（用连接器 + Unstructured 统一解析）、权限隔离（四层——数据源权限继承、Chunk 打标签、检索时过滤、展示时脱敏）、持续更新（Webhook + 哈希比对做增量索引，不需要全量重建）。架构上分接入层（多端）、查询处理层（改写→检索→重排→生成）、数据处理层（连接器→解析→分块→索引）三层。"

---

## 12.7 AI 系统设计：LLM API Gateway

### Q: 设计一个 LLM API Gateway，统一管理多个 LLM Provider 的调用。

**LLM API Gateway = 传统 API Gateway + LLM 特有能力。它是所有 LLM 调用的统一入口，解决多 Provider 管理、成本控制、限流降级、可观测性四大问题。**

**架构图：**

```
┌────────────────────────────────────────────────────────────┐
│                    LLM API Gateway                          │
│                                                            │
│  ┌───────────────────────────────────────────────────┐     │
│  │  接入层                                            │     │
│  │  ├── 统一 API 格式（OpenAI 兼容）                   │     │
│  │  ├── API Key 认证 + 租户隔离                        │     │
│  │  └── 请求校验（Pydantic Schema）                    │     │
│  └─────────────────────┬─────────────────────────────┘     │
│                        │                                    │
│  ┌─────────────────────▼─────────────────────────────┐     │
│  │  策略层                                            │     │
│  │  ├── 语义缓存 → 相似请求直接返回                     │     │
│  │  ├── 限流控制 → 令牌桶 / 滑动窗口（按租户）          │     │
│  │  ├── 模型路由 → 按任务复杂度/成本/延迟路由           │     │
│  │  ├── 内容审核 → 输入输出合规检查                     │     │
│  │  └── Prompt 注入防护 → 检测恶意 Prompt              │     │
│  └─────────────────────┬─────────────────────────────┘     │
│                        │                                    │
│  ┌─────────────────────▼─────────────────────────────┐     │
│  │  调度层                                            │     │
│  │  ├── Provider 池管理（OpenAI / Claude / Gemini）    │     │
│  │  ├── 负载均衡（加权轮询 / 最少连接）                 │     │
│  │  ├── 熔断降级（主 Provider 挂了切备用）              │     │
│  │  ├── 重试（指数退避 + Jitter）                      │     │
│  │  └── 流式转发（SSE 透传）                           │     │
│  └─────────────────────┬─────────────────────────────┘     │
│                        │                                    │
│  ┌─────────────────────▼─────────────────────────────┐     │
│  │  可观测层                                          │     │
│  │  ├── 请求日志（输入/输出/耗时/Token/成本）           │     │
│  │  ├── Metrics（Prometheus）                         │     │
│  │  ├── Tracing（OpenTelemetry）                      │     │
│  │  └── 成本仪表盘（按租户/模型/日期）                  │     │
│  └───────────────────────────────────────────────────┘     │
└────────────────────────────────────────────────────────────┘
          │              │              │
    ┌─────▼─────┐  ┌────▼─────┐  ┌────▼─────┐
    │  OpenAI   │  │  Claude  │  │  Gemini  │
    │  GPT-4o   │  │  Sonnet  │  │  Pro     │
    │  4o-mini  │  │  Haiku   │  │  Flash   │
    └───────────┘  └──────────┘  └──────────┘
```

**核心功能实现——统一 API 格式 + Provider 适配：**

```python
from abc import ABC, abstractmethod
from pydantic import BaseModel

# 统一请求格式（OpenAI 兼容）
class UnifiedChatRequest(BaseModel):
    model: str                          # "gpt-4o" / "claude-sonnet" / "gemini-pro"
    messages: list[dict]
    temperature: float = 0.7
    max_tokens: int = 4096
    stream: bool = False

# Provider 适配器接口
class LLMProvider(ABC):
    @abstractmethod
    async def chat(self, request: UnifiedChatRequest) -> str: ...

    @abstractmethod
    async def chat_stream(self, request: UnifiedChatRequest): ...

class OpenAIProvider(LLMProvider):
    async def chat(self, request: UnifiedChatRequest) -> str:
        response = await self.client.chat.completions.create(
            model=request.model,
            messages=request.messages,
            temperature=request.temperature,
        )
        return response.choices[0].message.content

class ClaudeProvider(LLMProvider):
    async def chat(self, request: UnifiedChatRequest) -> str:
        # 转换 message 格式（system 消息处理不同）
        system = next((m["content"] for m in request.messages if m["role"] == "system"), "")
        messages = [m for m in request.messages if m["role"] != "system"]

        response = await self.client.messages.create(
            model=self._map_model(request.model),
            system=system,
            messages=messages,
            max_tokens=request.max_tokens,
        )
        return response.content[0].text

# Gateway 核心
class LLMGateway:
    def __init__(self):
        self.providers = {
            "openai": OpenAIProvider(),
            "claude": ClaudeProvider(),
            "gemini": GeminiProvider(),
        }
        self.cache = SemanticCache()
        self.rate_limiter = RateLimiter()

    async def chat(self, request: UnifiedChatRequest, tenant_id: str) -> str:
        # 1. 限流检查
        await self.rate_limiter.check(tenant_id)

        # 2. 语义缓存
        cached = await self.cache.get(str(request.messages))
        if cached:
            return cached

        # 3. 路由到 Provider
        provider = self._route(request.model)

        # 4. 调用（带重试和熔断）
        response = await self._call_with_resilience(provider, request)

        # 5. 缓存结果
        await self.cache.set(str(request.messages), response)

        return response
```

**面试话术：**

> "LLM API Gateway 分四层：接入层（统一 OpenAI 兼容 API + 认证）、策略层（语义缓存 + 限流 + 路由 + 内容审核）、调度层（多 Provider 负载均衡 + 熔断降级 + 重试）、可观测层（日志 + Metrics + Tracing + 成本仪表盘）。核心价值是让业务方只对接一个 API，Gateway 内部管理多个 Provider 的切换、降级和成本优化。"

---

## 12.8 AI 系统设计：AI 内容审核系统

### Q: 设计一个 AI 内容审核系统，要求支持文本/图片审核，误判率 < 1%。

**AI 内容审核 = 多模态分类 + 规则引擎 + 人工复审的三层架构。单靠 LLM 审核误判率太高（5-10%），必须组合多种策略把误判率压到 < 1%。**

**三层审核架构：**

```
用户提交内容（文本 / 图片 / 视频帧）
         │
         ▼
┌────────────────────────────────────────────────────────┐
│  L1: 快速过滤层（< 10ms，过滤 70% 的明显违规）           │
│  ├── 关键词黑名单 / 正则匹配                             │
│  ├── URL / 电话号码 / 二维码检测                         │
│  ├── 图片 hash 比对（已知违规图库，pHash）                │
│  └── 结果：通过 / 拒绝 / 不确定 → 下一层                 │
└──────────────────────┬─────────────────────────────────┘
                       │ 不确定（约 30%）
                       ▼
┌────────────────────────────────────────────────────────┐
│  L2: AI 模型层（< 500ms，精确分类）                      │
│  ├── 文本分类模型（fine-tuned BERT / 自训练模型）         │
│  ├── 图片分类模型（NSFW 检测 / 暴力检测）                │
│  ├── LLM 审核（GPT-4o 做复杂语境理解）                   │
│  ├── 多模型投票（2/3 同意才判违规）                       │
│  └── 结果：通过 / 拒绝 / 低置信度 → 下一层               │
└──────────────────────┬─────────────────────────────────┘
                       │ 低置信度（约 5%）
                       ▼
┌────────────────────────────────────────────────────────┐
│  L3: 人工复审层                                         │
│  ├── 审核员队列（优先级排序）                             │
│  ├── 审核工具（高亮可疑部分 + AI 辅助建议）               │
│  ├── 结果反馈 → 模型训练数据                             │
│  └── SLA：4 小时内完成复审                               │
└────────────────────────────────────────────────────────┘
```

**多模型投票机制（降低误判率的关键）：**

```python
from enum import Enum
from pydantic import BaseModel

class ModerationResult(Enum):
    SAFE = "safe"
    UNSAFE = "unsafe"
    UNCERTAIN = "uncertain"

class ModerationVote(BaseModel):
    result: ModerationResult
    confidence: float          # 0-1
    category: str | None       # "hate", "violence", "sexual", etc.
    reason: str

async def multi_model_moderation(content: str) -> ModerationVote:
    """多模型投票审核"""

    # 并发调用多个审核模型
    votes = await asyncio.gather(
        openai_moderation(content),       # OpenAI Moderation API（免费）
        custom_bert_classifier(content),   # 自训练 BERT 分类器
        llm_moderation(content),           # GPT-4o-mini 审核
    )

    # 投票逻辑
    unsafe_count = sum(1 for v in votes if v.result == ModerationResult.UNSAFE)

    if unsafe_count >= 2:
        # 2/3 判违规 → 违规
        return ModerationVote(
            result=ModerationResult.UNSAFE,
            confidence=unsafe_count / len(votes),
            category=_merge_categories(votes),
            reason=_merge_reasons(votes),
        )
    elif unsafe_count == 1:
        # 1/3 判违规 → 不确定 → 人工复审
        return ModerationVote(
            result=ModerationResult.UNCERTAIN,
            confidence=0.5,
            category=votes[0].category,
            reason="模型投票分歧，需人工复审",
        )
    else:
        return ModerationVote(
            result=ModerationResult.SAFE,
            confidence=min(v.confidence for v in votes),
            category=None,
            reason="所有模型判定安全",
        )
```

**关键指标与告警：**

| 指标 | 目标 | 告警阈值 |
|------|------|---------|
| 误判率（False Positive） | < 1% | > 1.5% |
| 漏判率（False Negative） | < 0.1% | > 0.5% |
| L1 处理延迟 | < 10ms | > 50ms |
| L2 处理延迟 | < 500ms | > 2s |
| L3 人工复审时效 | < 4h | > 8h |
| 日均审核量 | 可扩展到 1M+ | - |
| 人工复审占比 | < 5% | > 10% |

**降低误判率的策略总结：**

| 策略 | 效果 | 描述 |
|------|------|------|
| **多模型投票** | 误判率 -60% | 2/3 同意才判违规 |
| **置信度阈值** | 误判率 -30% | 低置信度转人工，不自动拒绝 |
| **类别细分** | 误判率 -20% | 不同类别用不同模型和阈值 |
| **上下文理解** | 误判率 -40% | 用 LLM 理解语境（讽刺、引用、教育） |
| **持续训练** | 误判率 -15%/月 | 人工复审结果反馈到模型训练 |

**面试话术：**

> "AI 内容审核系统用三层架构：L1 快速过滤（关键词 + 哈希，<10ms，过滤 70% 明显违规），L2 AI 模型（多模型投票——OpenAI Moderation + 自训练 BERT + LLM，2/3 同意才判违规），L3 人工复审（低置信度的 5% 转人工，结果反馈到训练数据）。误判率控制在 1% 以下的关键是多模型投票 + 置信度阈值分流——不确定的不自动拒绝，而是转人工。"

---

## 本章总结

```
┌──────────────────────────────────────────────────────────────┐
│                  生产部署与运维知识体系                          │
│                                                              │
│  部署层                                                       │
│  ├── 三种策略 → 云 API / 自部署(vLLM) / 混合部署               │
│  └── 选型关键 → 数据安全 / 成本 / 延迟 / 运维复杂度            │
│                                                              │
│  优化层                                                       │
│  ├── 语义缓存 → 相似问题零成本                                 │
│  ├── 模型路由 → 简单→小模型，复杂→大模型                       │
│  ├── Prompt 压缩 → LLMLingua / 动态 Context                   │
│  └── 批处理 → Batch API 省 50%                                │
│                                                              │
│  运维层                                                       │
│  ├── 可观测性 → LangSmith + Prometheus + OpenTelemetry         │
│  └── Agent SLA → 四维（可用性/延迟/质量/成本）+ Guardrails     │
│                                                              │
│  系统设计                                                     │
│  ├── 百万 DAU 客服 → 四层分流 (FAQ→RAG→Agent→人工)            │
│  ├── 企业 RAG → 多数据源 + 权限四层 + 增量更新                 │
│  ├── LLM Gateway → 统一 API + 策略 + 调度 + 可观测             │
│  └── 内容审核 → 三层架构 + 多模型投票 + 人工复审                │
└──────────────────────────────────────────────────────────────┘
```

---

## 12.9 MLOps 与 CI/CD

### Q: AI 应用的 MLOps 完整流程是什么？和传统 DevOps 有什么区别？

**MLOps = Machine Learning + DevOps，管理 AI 应用从开发到部署的全生命周期。**

**AI 应用的 CI/CD 与传统的区别：**

| 维度 | 传统 DevOps | AI MLOps |
|------|------------|----------|
| 测试对象 | 代码逻辑 | 代码 + 模型质量 + Prompt 效果 |
| 回归标准 | 单元测试通过 | 单元测试 + RAGAS 评分 ≥ 阈值 |
| 部署产物 | 代码包 | 代码 + 模型权重 + Prompt 版本 + 向量索引 |
| 监控 | CPU/内存/QPS | + Token 成本 / 幻觉率 / Faithfulness |
| 回滚 | 代码回滚 | 代码 + Prompt + 模型权重 全部回滚 |

**AI 应用 CI/CD Pipeline：**

```
┌─ CI（持续集成）───────────────────────────────────────┐
│  1. 代码检查（lint, type check）                       │
│  2. 单元测试（Mock LLM，测业务逻辑）                    │
│  3. Prompt 回归测试                                   │
│     → 跑评估数据集，RAGAS Faithfulness ≥ 0.85          │
│     → DeepEval pytest 风格：assert score > threshold   │
│  4. 集成测试（真实 API 调用，抽样 10%）                 │
└───────────────────────────────────────────────────────┘
        ↓ 全部通过
┌─ CD（持续部署）───────────────────────────────────────┐
│  5. 灰度发布（5% 流量 → 观察 1 小时）                   │
│  6. 监控告警（幻觉率 > 5%？成本异常？延迟飙升？）        │
│  7. 全量发布 or 回滚                                   │
└───────────────────────────────────────────────────────┘
```

```python
# DeepEval CI/CD 示例（放在 pytest 中）
from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

def test_rag_quality():
    test_case = LLMTestCase(
        input="公司年假政策是什么？",
        actual_output=rag_pipeline("公司年假政策是什么？"),
        retrieval_context=["员工满一年享有5天年假..."]
    )
    metric = FaithfulnessMetric(threshold=0.85)
    metric.measure(test_case)
    assert metric.score >= 0.85, f"Faithfulness {metric.score} < 0.85"
```

**面试话术：**
> "AI 应用的 CI/CD 比传统多一步——Prompt 回归测试。每次改 prompt 或模型版本，自动跑评估数据集，用 DeepEval 检查 Faithfulness ≥ 0.85。低于阈值就阻止上线。灰度发布时监控幻觉率和成本，异常就自动回滚。"

---

## 12.10 Prompt Caching（API 成本优化）

### Q: 什么是 Prompt Caching？它能省多少钱？

**Prompt Caching = API 层面缓存 System Prompt 的前缀，避免重复计算。**

```
没有 Prompt Caching：
  请求 1：[System Prompt 2000 tokens] + [User: 你好]    → 计费 2050 tokens
  请求 2：[System Prompt 2000 tokens] + [User: 谢谢]    → 计费 2050 tokens
  → 每次都重新处理 System Prompt，重复付费

有 Prompt Caching：
  请求 1：[System Prompt 2000 tokens] + [User: 你好]    → 计费 2050 tokens（首次全价）
  请求 2：[缓存命中 2000 tokens] + [User: 谢谢]         → 计费 50 tokens（省 90%！）
```

**主流 API 的 Prompt Caching 支持：**

| 提供商 | 功能 | 折扣 | 最小缓存长度 |
|--------|------|------|-------------|
| Anthropic | 自动缓存 | 输入价格 90% 折扣 | 1024 tokens |
| OpenAI | 自动（GPT-4o） | 输入价格 50% 折扣 | 1024 tokens |
| Google | Context Caching | 按缓存存储时间计费 | 32K tokens |

**使用建议：**
- System Prompt 超过 1024 tokens → 自动获益
- RAG 场景：把固定的 System Prompt + 格式要求放前面，检索内容放后面
- 多轮对话：历史消息在前，新消息在后

**面试话术：**
> "Prompt Caching 是 2024-2025 年 API 成本优化的重大突破。原理是缓存不变的 prompt 前缀，Anthropic 给 90% 折扣，OpenAI 给 50%。我在 RAG 系统中把 2000 tokens 的 System Prompt 设计为固定前缀，配合语义缓存，总成本降了 60%。"

---

## 12.11 AI 四层黄金架构

### Q: 什么是企业级 AI 四层黄金架构（RAG → Agent → MCP → A2A）？

**2026 年企业 AI 应用的标准分层架构：**

```
┌────────────────────────────────────────────────────┐
│  Layer 4: A2A（Agent-to-Agent）                     │
│  Agent 之间的通信协议                                │
│  → 多 Agent 协作、任务委托、能力发现                  │
├────────────────────────────────────────────────────┤
│  Layer 3: MCP（Model Context Protocol）             │
│  Agent 与外部工具/数据源的连接协议                    │
│  → 统一工具调用、资源访问、权限控制                    │
├────────────────────────────────────────────────────┤
│  Layer 2: AI Agent                                  │
│  自主决策和任务执行                                   │
│  → ReAct / Planning / Reflection / Memory           │
├────────────────────────────────────────────────────┤
│  Layer 1: RAG                                       │
│  知识检索与增强生成                                   │
│  → Embedding / 检索 / Rerank / 生成                  │
└────────────────────────────────────────────────────┘
```

**四层如何协同：**

```
用户提问："对比我们公司和竞品的 Q1 营收"

Layer 1 (RAG)：检索公司内部财务文档
Layer 2 (Agent)：规划任务 → 需要内部数据 + 外部数据
Layer 3 (MCP)：调用内部数据库 MCP Server + 外部搜索 MCP Server
Layer 4 (A2A)：财务分析 Agent 请求数据采集 Agent 协助
→ 返回结构化对比报告
```

**面试话术：**
> "企业级 AI 应用的四层架构是 RAG→Agent→MCP→A2A。RAG 提供知识基础，Agent 做自主决策，MCP 统一工具连接，A2A 实现多 Agent 协作。不是每个项目都需要四层——简单问答只要 L1（RAG），复杂任务加 L2（Agent），多工具加 L3（MCP），多团队协作才需要 L4（A2A）。"

---

## 12.12 系统设计：企业级 AI Agent 平台（类 Coze/Dify）

### Q: 如何设计一个企业级 AI Agent 平台？让非技术人员也能搭建 Agent？

**核心需求：** 像 Coze/Dify 一样，让业务人员通过拖拽/配置创建 AI Agent，不写代码。

**四层架构：**

```
┌──────────────────────────────────────────────────────────┐
│  用户层：可视化 Agent Builder（拖拽工作流编辑器）          │
│  → React Flow / 画布式 UI / Prompt 模板库                │
├──────────────────────────────────────────────────────────┤
│  编排层：Agent Runtime Engine（执行引擎）                 │
│  → 工作流解释器 / ReAct 循环 / 状态机执行                │
│  → 支持 Chain（线性）和 Loop（循环）两种模式               │
├──────────────────────────────────────────────────────────┤
│  能力层：模型 + 工具 + 知识库                             │
│  → 多模型路由（GPT/Claude/Qwen 切换）                    │
│  → MCP Server 市场（工具插件化）                         │
│  → RAG 知识库（上传文档即用）                            │
├──────────────────────────────────────────────────────────┤
│  基础层：多租户 + 权限 + 计费 + 监控                      │
│  → 租户隔离 / RBAC / Token 用量计费 / LangSmith 可观测    │
└──────────────────────────────────────────────────────────┘
```

**关键设计决策：**

| 决策 | 选项 | 建议 |
|------|------|------|
| 工作流引擎 | LangGraph / 自研 DAG | LangGraph（生态好），复杂场景自研 |
| 模型接入 | 直连 API / 统一 Gateway | Gateway（统一限流、计费、切换） |
| 工具扩展 | 硬编码 / MCP 协议 | MCP（标准化，用户可自建 Server） |
| 知识库 | 内置 RAG / 外接 | 内置（降低用户门槛） |
| 多租户 | Schema 隔离 / DB 隔离 | Schema 隔离（成本低），大客户 DB 隔离 |

**前端类比：** 这就是给 AI 做一个 "低代码平台"——和前端低代码平台（如 Retool）的架构思路完全一样：可视化编辑器 + 组件市场 + 执行引擎 + 多租户。

**面试话术：**
> "我会设计四层架构：顶层是 React Flow 可视化编辑器让用户拖拽工作流；编排层用 LangGraph 执行 Agent 逻辑；能力层通过 MCP 协议插件化接入工具，RAG 知识库内置；基础层做多租户隔离和 Token 计费。核心思路是把 Agent 开发从'写代码'变成'搭积木'。"

---

**面试高频考点 Top 5：**

| 排名 | 考点 | 关键答题点 |
|------|------|-----------|
| 1 | 成本优化 | 四大策略（缓存→路由→压缩→批处理），省 60-80% |
| 2 | 系统设计：客服 | 四层分流、容量估算、成本计算 |
| 3 | 可观测性 | AI 特有指标（Token/成本/幻觉率）、LangSmith |
| 4 | 部署策略 | 混合部署、vLLM vs Ollama、AI Gateway |
| 5 | Agent SLA | 四维指标、Guardrails、降级策略 |

---

**章节导航：**
[上一章：11-ai-engineering](./11-ai-engineering.md) | [目录](./README.md)