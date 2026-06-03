# 11 - AI 应用工程化

> **难度：** ⭐⭐⭐⭐ | **定位：** 从"能跑 Demo"到"能上线"的关键跳跃，AI 工程师必考
>
> **前端类比：** 如果 LLM 是 React，那本章就是 Next.js + Express + Zod + Jest 的全家桶。SSE 你在 ChatGPT 每天都在用，Pydantic ≈ Zod，FastAPI ≈ Express，LangChain ≈ 中间件模式，asyncio ≈ Promise.all。前端转 AI 工程，本章是你最容易拿分的章节。

## 本章知识树

```
AI 应用工程化
├── 11.1 流式输出（SSE vs WebSocket，前端实现）
├── 11.2 LLM API 调用最佳实践（重试、超时、降级）
├── 11.3 LangChain 核心组件与 LCEL
├── 11.4 LangGraph 状态机工作流
├── 11.5 低代码 AI 平台（Dify / Coze / n8n）
├── 11.6 Python 异步编程（asyncio for AI apps）
├── 11.7 Pydantic v2 结构化数据验证
├── 11.8 FastAPI + SSE 实现 AI 接口
├── 11.9 AI 应用测试策略
└── 11.10 NL2SQL（自然语言转 SQL）
```

---

## 11.1 流式输出（SSE vs WebSocket）

### Q: SSE 和 WebSocket 在 AI 流式输出场景下如何选择？请分别给出后端（Go/Python）和前端（JS）的实现。

**SSE（Server-Sent Events）是 AI 流式输出的首选方案。它是单向的（服务器 → 客户端），基于 HTTP，天然适合"LLM 逐 Token 推送"的场景。WebSocket 是双向的，更适合聊天室、协同编辑等需要客户端频繁发消息的场景。**

**SSE vs WebSocket 对比：**

| 维度 | SSE | WebSocket |
|------|-----|-----------|
| **方向** | 单向（Server → Client） | 双向 |
| **协议** | HTTP/1.1（或 HTTP/2） | ws:// / wss:// |
| **重连** | 浏览器自动重连 | 需要手动实现 |
| **数据格式** | 纯文本（`text/event-stream`） | 文本或二进制 |
| **代理兼容性** | 好（走标准 HTTP） | 差（部分代理不支持升级） |
| **AI 场景适配** | ✅ 完美（逐 Token 推送） | 过重（不需要双向） |
| **前端类比** | `EventSource` API（原生） | `new WebSocket()` |

**为什么 ChatGPT / Claude 都用 SSE？**

```
SSE 的优势场景：
  1. LLM 生成是单向的——模型逐 Token 吐出，客户端只需接收
  2. 基于 HTTP，CDN / Nginx / 云网关 零配置支持
  3. 浏览器原生 EventSource API，自动重连 + Last-Event-ID
  4. 不需要维护长连接状态，服务端无状态更易扩展
```

**Go 后端 SSE 实现：**

```go
// Go 标准库实现 SSE
func sseHandler(w http.ResponseWriter, r *http.Request) {
    w.Header().Set("Content-Type", "text/event-stream")
    w.Header().Set("Cache-Control", "no-cache")
    w.Header().Set("Connection", "keep-alive")

    flusher, ok := w.(http.Flusher)
    if !ok {
        http.Error(w, "Streaming not supported", 500)
        return
    }

    // 模拟 LLM 逐 Token 输出
    tokens := []string{"Hello", " ", "World", "!", " ", "How", " are", " you?"}
    for i, token := range tokens {
        fmt.Fprintf(w, "id: %d\n", i)
        fmt.Fprintf(w, "data: %s\n\n",
            fmt.Sprintf(`{"token": "%s", "done": %v}`, token, i == len(tokens)-1))
        flusher.Flush()
        time.Sleep(100 * time.Millisecond)
    }
}
```

**Python 后端 SSE 实现（FastAPI）：**

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio, json

app = FastAPI()

async def generate_tokens(prompt: str):
    """模拟 LLM 流式输出"""
    async for chunk in llm.astream(prompt):
        yield f"data: {json.dumps({'token': chunk.content, 'done': False})}\n\n"
    yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    return StreamingResponse(
        generate_tokens(request.prompt),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}  # 关闭 Nginx 缓冲
    )
```

**前端 JS 实现（两种方式）：**

```javascript
// 方式 1：原生 EventSource（仅支持 GET）
const source = new EventSource('/chat/stream?prompt=hello');
source.onmessage = (event) => {
    const { token, done } = JSON.parse(event.data);
    document.getElementById('output').textContent += token;
    if (done) source.close();
};

// 方式 2：fetch + ReadableStream（支持 POST，推荐）
async function streamChat(prompt) {
    const response = await fetch('/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt }),
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop();  // 保留不完整的部分

        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const data = JSON.parse(line.slice(6));
                onToken(data.token);  // 逐 Token 渲染
            }
        }
    }
}
```

**面试话术：**

> "AI 流式输出首选 SSE，因为 LLM 生成是单向的，不需要 WebSocket 的双向能力。SSE 基于标准 HTTP，代理和 CDN 零配置支持，浏览器还有原生 EventSource API 自动重连。前端用 fetch + ReadableStream 来处理 POST 请求的 SSE，就像处理一个会持续 emit 的 Node.js Stream。"

---

## 11.2 LLM API 调用最佳实践

### Q: 如何实现 LLM API 的指数退避重试？生产环境还需要哪些防护措施？

**LLM API 调用三大杀手：网络抖动（5xx）、限流（429 Rate Limit）、超时（模型推理慢）。生产环境必须实现指数退避重试 + 超时控制 + 降级兜底的三层防护。**

**指数退避重试（Exponential Backoff with Jitter）：**

```
为什么要加 Jitter（随机抖动）？

不加 Jitter（雷群效应 Thundering Herd）：
  Time 0s: 1000 个请求同时失败
  Time 1s: 1000 个请求同时重试 → 再次失败
  Time 2s: 1000 个请求同时重试 → 服务器雪崩

加 Jitter（打散重试时间）：
  Time 0s:   1000 个请求失败
  Time 0.5s: 200 个重试
  Time 1.2s: 300 个重试
  Time 1.8s: 250 个重试
  Time 2.5s: 250 个重试
  → 压力均匀分散，服务器可正常恢复
```

**Python 完整实现：**

```python
import asyncio
import random
from openai import AsyncOpenAI, APIError, RateLimitError, APITimeoutError

client = AsyncOpenAI()

async def llm_call_with_retry(
    messages: list,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    timeout: float = 30.0,
    fallback_model: str = "gpt-4o-mini",  # 降级模型
):
    """带重试、超时、降级的 LLM 调用"""

    model = "gpt-4o"  # 主模型
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.7,
                ),
                timeout=timeout,
            )
            return response.choices[0].message.content

        except RateLimitError as e:
            last_error = e
            # 429 特殊处理：尊重 Retry-After 头
            retry_after = float(e.response.headers.get("Retry-After", base_delay))
            await asyncio.sleep(retry_after)

        except (APIError, APITimeoutError) as e:
            last_error = e
            if attempt < max_retries:
                # 指数退避 + Full Jitter
                delay = min(base_delay * (2 ** attempt), max_delay)
                jitter = random.uniform(0, delay)  # Full Jitter
                print(f"Retry {attempt+1}/{max_retries} after {jitter:.1f}s: {e}")
                await asyncio.sleep(jitter)

        except Exception as e:
            # 不可重试的错误（如 400 Bad Request）
            raise

    # 所有重试失败 → 降级到小模型
    print(f"Primary model failed, falling back to {fallback_model}")
    try:
        response = await client.chat.completions.create(
            model=fallback_model,
            messages=messages,
        )
        return response.choices[0].message.content
    except Exception:
        raise last_error  # 降级也失败，抛出原始错误
```

**生产环境完整防护体系：**

| 层级 | 策略 | 实现 |
|------|------|------|
| **重试层** | 指数退避 + Full Jitter | `delay = random(0, min(base * 2^n, max))` |
| **超时层** | 单次请求超时 + 总超时 | `asyncio.wait_for()` + 总计时器 |
| **限流层** | 客户端限流（令牌桶） | `asyncio.Semaphore` / `aiolimiter` |
| **降级层** | 主模型 → 小模型 → 缓存 → 兜底文案 | GPT-4o → GPT-4o-mini → 缓存 → 默认回复 |
| **熔断层** | 连续失败 N 次自动熔断 | `circuitbreaker` 库 / 自实现 |
| **缓存层** | 相同 Query 命中缓存 | 语义缓存（Embedding 相似度） |

**三种 Jitter 策略对比：**

```
Full Jitter:    delay = random(0, min(base * 2^n, cap))     ← 推荐
Equal Jitter:   delay = min(base * 2^n, cap)/2 + random(0, min(base * 2^n, cap)/2)
Decorr Jitter:  delay = min(cap, random(base, prev_delay * 3))
```

**面试话术：**

> "LLM API 调用的核心防护是三层：重试（指数退避 + Full Jitter 防雷群效应）、超时（asyncio.wait_for 控制单次请求时间）、降级（主模型挂了切小模型，小模型也挂了走语义缓存或兜底文案）。这跟前端的 fetch 重试 + AbortController 超时 + Fallback UI 是同一个思路。"

---

## 11.3 LangChain 核心组件与 LCEL

### Q: LangChain 有哪些核心组件？LCEL 管道语法是怎么回事？

**LangChain 是 AI 应用开发的"中间件框架"，LCEL（LangChain Expression Language）是它的管道语法，用 `|` 操作符把组件串联起来，就像 Unix Pipe 或 RxJS 的 `.pipe()`。**

**LangChain 核心组件表：**

| 组件 | 作用 | 前端类比 |
|------|------|---------|
| **ChatModel** | 封装 LLM 调用（OpenAI / Claude / Gemini） | `fetch('/api/llm')` |
| **PromptTemplate** | 构造 Prompt 模板，支持变量插值 | 模板字符串 `${variable}` |
| **OutputParser** | 解析 LLM 输出为结构化数据（JSON / Pydantic） | `response.json()` |
| **Retriever** | 从向量数据库检索文档 | 数据库查询层 |
| **Chain** | 组合多个组件为一个工作流 | Express 中间件链 |
| **Tool** | 给 Agent 使用的外部函数 | API Route |
| **Memory** | 对话历史管理 | Redux Store |
| **Callback** | 链路追踪、日志、监控 | Express 中间件 `(req, res, next)` |

**LCEL 管道语法（核心！）：**

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LCEL：用 | 连接组件，形成管道
chain = (
    ChatPromptTemplate.from_messages([
        ("system", "你是一个{role}专家"),
        ("human", "{question}"),
    ])
    | ChatOpenAI(model="gpt-4o", temperature=0.7)
    | StrOutputParser()
)

# 同步调用
result = chain.invoke({"role": "Python", "question": "什么是 GIL?"})

# 流式调用
for chunk in chain.stream({"role": "Python", "question": "什么是 GIL?"}):
    print(chunk, end="")

# 异步调用
result = await chain.ainvoke({"role": "Python", "question": "什么是 GIL?"})

# 批量调用
results = chain.batch([
    {"role": "Python", "question": "什么是 GIL?"},
    {"role": "Go", "question": "什么是 Goroutine?"},
])
```

**LCEL 底层原理：**

```python
# LCEL 的 | 操作符本质上是 RunnableSequence
# prompt | model | parser
# 等价于：
from langchain_core.runnables import RunnableSequence
chain = RunnableSequence(first=prompt, middle=[model], last=parser)

# 每个组件实现 Runnable 接口：
class Runnable:
    def invoke(self, input) -> output       # 同步
    async def ainvoke(self, input) -> output # 异步
    def stream(self, input) -> Iterator     # 流式
    def batch(self, inputs) -> list         # 批量
```

**LCEL 进阶用法——并行与条件分支：**

```python
from langchain_core.runnables import RunnableParallel, RunnableLambda

# 并行执行（类似 Promise.all）
chain = RunnableParallel(
    summary=prompt_summary | model | parser,
    keywords=prompt_keywords | model | parser,
    sentiment=prompt_sentiment | model | parser,
)
result = chain.invoke({"text": "今天天气真好..."})
# result = {"summary": "...", "keywords": "...", "sentiment": "..."}

# 条件分支（类似三元表达式）
from langchain_core.runnables import RunnableBranch

route_chain = RunnableBranch(
    (lambda x: "代码" in x["question"], code_chain),
    (lambda x: "翻译" in x["question"], translate_chain),
    default_chain,  # 兜底
)
```

**面试话术：**

> "LangChain 的核心价值是组件化和标准化。LCEL 用 `|` 管道把 Prompt → Model → Parser 串起来，每个组件都实现同一个 Runnable 接口，天然支持同步/异步/流式/批量四种调用模式。这就像 Express 中间件 `app.use(a).use(b).use(c)`，数据从左到右流过每个中间件。RunnableParallel 就是 `Promise.all`，RunnableBranch 就是路由。"

---

## 11.4 LangGraph 状态机工作流

### Q: LangGraph 是什么？它和 LangChain 的 Chain 有什么区别？

**LangGraph 是 LangChain 团队出品的工作流引擎，核心思想是把 AI 工作流建模为状态图（StateGraph）。Chain 是线性管道（A→B→C），LangGraph 是有向图（支持循环、分支、条件跳转），专门用来构建 Agent 和复杂工作流。**

```
Chain（线性管道）：
  Input → Step A → Step B → Step C → Output
  ❌ 不能循环、不能条件分支、不能人工介入

LangGraph（状态图）：
  Input → Step A → [条件] → Step B ←→ Step C → [人工审核] → Output
                      ↓                              ↑
                   Step D ──────────────────────────┘
  ✅ 循环、分支、人工介入、持久化状态
```

**LangGraph 核心概念：**

| 概念 | 作用 | 前端类比 |
|------|------|---------|
| **State** | 全局共享状态，贯穿整个工作流 | Redux Store |
| **Node** | 工作流中的一个处理步骤（函数） | React 组件 |
| **Edge** | 节点之间的连接 | React Router 路由跳转 |
| **Conditional Edge** | 根据状态决定下一步走哪个节点 | 条件渲染 `{flag && <Comp/>}` |
| **Checkpointer** | 状态持久化，支持暂停恢复 | localStorage / SessionStorage |

**LangGraph 状态机示例——AI 客服工作流：**

```python
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# 1. 定义状态
class CustomerState(TypedDict):
    messages: Annotated[list, add_messages]  # 对话历史（自动追加）
    category: str          # 问题分类
    sentiment: str         # 用户情绪
    needs_human: bool      # 是否需要人工
    resolution: str        # 解决方案

# 2. 定义节点（每个节点是一个函数，输入 State，输出 State 更新）
async def classify_intent(state: CustomerState) -> dict:
    """意图分类"""
    response = await llm.ainvoke(
        f"将以下问题分类为 billing/technical/complaint：\n{state['messages'][-1].content}"
    )
    return {"category": response.content}

async def analyze_sentiment(state: CustomerState) -> dict:
    """情绪分析"""
    response = await llm.ainvoke(
        f"分析用户情绪（positive/neutral/negative/angry）：\n{state['messages'][-1].content}"
    )
    sentiment = response.content
    return {
        "sentiment": sentiment,
        "needs_human": sentiment == "angry",
    }

async def auto_resolve(state: CustomerState) -> dict:
    """自动解决"""
    response = await llm.ainvoke(f"作为客服，解决此{state['category']}问题：{state['messages']}")
    return {"resolution": response.content}

async def escalate_to_human(state: CustomerState) -> dict:
    """转人工"""
    return {"resolution": "已转接人工客服，请稍候..."}

# 3. 定义路由（条件边）
def route_by_sentiment(state: CustomerState) -> Literal["auto_resolve", "escalate"]:
    if state["needs_human"]:
        return "escalate"
    return "auto_resolve"

# 4. 构建图
graph = StateGraph(CustomerState)

# 添加节点
graph.add_node("classify", classify_intent)
graph.add_node("sentiment", analyze_sentiment)
graph.add_node("auto_resolve", auto_resolve)
graph.add_node("escalate", escalate_to_human)

# 添加边
graph.set_entry_point("classify")
graph.add_edge("classify", "sentiment")
graph.add_conditional_edges("sentiment", route_by_sentiment, {
    "auto_resolve": "auto_resolve",
    "escalate": "escalate",
})
graph.add_edge("auto_resolve", END)
graph.add_edge("escalate", END)

# 编译并运行
app = graph.compile()
result = await app.ainvoke({
    "messages": [HumanMessage(content="你们的破产品又崩了！退钱！")],
})
```

**面试话术：**

> "LangGraph 把 AI 工作流从'线性管道'升级为'状态图'。核心三件套：State（共享状态，类似 Redux Store）、Node（处理节点，类似 React 组件）、Edge（连接和路由，类似 React Router）。它的杀手锏是支持循环（Agent 的 Reason→Act→Observe 循环）和人工介入（暂停等待审批后恢复），这是普通 Chain 做不到的。"

---

## 11.5 低代码 AI 平台

### Q: Dify、Coze、n8n 三大低代码 AI 平台怎么选？各有什么优缺点？

**三大平台定位不同：Dify 是开源的 AI 应用开发平台（侧重 RAG + Agent），Coze 是字节跳动的 Bot 构建平台（侧重发布到社交平台），n8n 是通用自动化工作流平台（加了 AI 节点）。**

**核心对比：**

| 维度 | Dify | Coze | n8n |
|------|------|------|-----|
| **定位** | AI 应用开发平台 | AI Bot 构建平台 | 通用工作流自动化 |
| **开源** | ✅ 开源（可私有部署） | ❌ 闭源 | ✅ 开源 |
| **核心能力** | RAG + Agent + Workflow | Bot + Plugin + 发布渠道 | 500+集成（社区含1800+） + AI 节点 |
| **模型支持** | OpenAI / Claude / 本地模型 | 主要字节系（豆包）+ OpenAI | 通过 HTTP 接入任意模型 |
| **RAG** | ⭐⭐⭐⭐⭐ 内置完整 RAG Pipeline | ⭐⭐⭐ 基础知识库 | ⭐⭐ 需自己拼 |
| **工作流** | ⭐⭐⭐⭐ 可视化编排 | ⭐⭐⭐ 简单流程 | ⭐⭐⭐⭐⭐ 最强工作流 |
| **发布渠道** | API / Web App | 微信/飞书/抖音/Discord | Webhook / API |
| **适合团队** | AI 应用开发团队 | 快速做 Bot 的产品团队 | 已有 n8n 的运维/自动化团队 |
| **私有部署** | ✅ Docker Compose 一键部署 | ❌ | ✅ |
| **前端类比** | Vercel（全栈开发平台） | Wix（建站工具） | Zapier（自动化平台） |

**选型决策树：**

```
需要私有部署？
├── 是 → 需要强 RAG？
│       ├── 是 → Dify ✅
│       └── 否 → n8n ✅
└── 否 → 主要发布到社交平台？
        ├── 是 → Coze ✅
        └── 否 → 需要复杂工作流？
                ├── 是 → n8n ✅
                └── 否 → Dify Cloud ✅
```

**面试话术：**

> "三选一的话：如果做企业 AI 应用（RAG / Agent / 私有部署），选 Dify；如果做社交平台 Bot（飞书 / 微信 / 抖音），选 Coze；如果已有自动化基建、想加 AI 能力，选 n8n。Dify 是 AI-native 的，RAG 和 Agent 做得最好；Coze 胜在发布渠道多；n8n 胜在通用工作流集成最强。"

---

## 11.6 Python 异步编程（asyncio for AI apps）

### Q: 为什么 AI 应用必须用异步编程？如何用 asyncio.Semaphore 控制 LLM 并发？

**AI 应用 95% 的时间在等 I/O（等 LLM API 响应、等向量数据库查询、等网络请求）。同步编程在等待期间线程被阻塞，一个线程只能处理一个请求。异步编程让一个线程在等待 I/O 时去处理其他请求，吞吐量提升 10-50 倍。**

**前端类比：** Python asyncio 就是 JavaScript 的 Event Loop。`await` 是一样的，`asyncio.gather()` ≈ `Promise.all()`，`asyncio.Semaphore` ≈ 手写的并发池（p-limit 库）。

```
同步（一个一个等）：
  Request 1: [--发送--|=========等 LLM 3s========|--处理--]
  Request 2:                                              [--发送--|=========等 3s========|--处理--]
  Request 3:                                                                                       [--发送--|=========等 3s========|--处理--]
  总耗时：9s+

异步（同时等）：
  Request 1: [--发送--|=========等 LLM 3s========|--处理--]
  Request 2: [--发送--|=========等 LLM 3s========|--处理--]
  Request 3: [--发送--|=========等 LLM 3s========|--处理--]
  总耗时：3s+（≈1 个请求的时间）
```

**asyncio.Semaphore 控制 LLM 并发——防止打爆 API 限流：**

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()

# 信号量：最多同时 5 个 LLM 请求（防止触发 Rate Limit）
semaphore = asyncio.Semaphore(5)

async def llm_call(prompt: str) -> str:
    """带并发限制的 LLM 调用"""
    async with semaphore:  # 最多 5 个并发
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

async def batch_process(prompts: list[str]) -> list[str]:
    """批量并发处理，自动限流"""
    tasks = [llm_call(prompt) for prompt in prompts]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 分离成功和失败
    successes = [r for r in results if not isinstance(r, Exception)]
    failures = [r for r in results if isinstance(r, Exception)]

    if failures:
        print(f"⚠️ {len(failures)} 个请求失败: {failures[0]}")

    return results

# 使用：100 个 prompt 同时发出，但最多 5 个并发
prompts = [f"总结第 {i} 段文档" for i in range(100)]
results = asyncio.run(batch_process(prompts))
```

**asyncio 核心 API 速查表：**

| API | 作用 | JS 等价 |
|-----|------|---------|
| `await coroutine` | 等待一个异步操作 | `await promise` |
| `asyncio.gather(*tasks)` | 并发执行多个任务 | `Promise.all([...])` |
| `asyncio.wait_for(coro, timeout)` | 超时控制 | `Promise.race([p, timeout])` |
| `asyncio.Semaphore(n)` | 限制并发数 | `p-limit(n)` |
| `asyncio.Queue()` | 异步队列 | 无直接等价（可用 Channel） |
| `asyncio.TaskGroup()` | 任务组（Python 3.11+） | `Promise.all()（失败时取消其余任务，比Promise.all更严格）` |

> **注：** `Promise.allSettled()` 的 Python 对应是 `asyncio.gather(return_exceptions=True)`

**面试话术：**

> "AI 应用本质是 I/O 密集型——95% 的时间在等 LLM API 响应。用 asyncio 可以在等待期间处理其他请求，吞吐量提升数十倍。Semaphore 用来限流，防止并发太高打爆 API 的 Rate Limit。这就像前端的 `Promise.all()` + `p-limit` 并发池。"

---

## 11.7 Pydantic v2 结构化数据验证

### Q: 为什么 AI 应用需要 Pydantic？如何用它验证 LLM 输出？

**LLM 输出是非结构化的字符串，但下游系统需要结构化数据（JSON / 数据库 / API 响应）。Pydantic 就是 LLM 输出的"类型守卫"——定义 Schema，验证输出，类型不对就报错重试。对前端来说，Pydantic ≈ Zod。**

```
LLM 输出（不可控）：
  "用户情感分析结果：正面的，置信度大概 0.95 左右"  ← 非结构化，无法直接用

Pydantic 验证后（结构化）：
  SentimentResult(sentiment="positive", confidence=0.95, reason="...")  ← 类型安全
```

**Pydantic v2 用于 LLM 输出验证：**

```python
from pydantic import BaseModel, Field, field_validator
from openai import OpenAI

# 1. 定义输出 Schema（类似前端的 Zod Schema）
class MovieReview(BaseModel):
    """电影评论的结构化分析结果"""
    title: str = Field(description="电影名称")
    sentiment: str = Field(description="情感：positive / neutral / negative")
    score: float = Field(ge=0, le=10, description="评分 0-10")
    keywords: list[str] = Field(min_length=1, max_length=5, description="关键词")
    summary: str = Field(max_length=200, description="一句话总结")

    @field_validator("sentiment")
    @classmethod
    def validate_sentiment(cls, v):
        allowed = {"positive", "neutral", "negative"}
        if v not in allowed:
            raise ValueError(f"sentiment 必须是 {allowed} 之一，但收到 '{v}'")
        return v

# 2. 让 LLM 输出符合 Schema（OpenAI Structured Output）
client = OpenAI()

def analyze_review(review_text: str) -> MovieReview:
    completion = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "分析以下电影评论，返回结构化结果。"},
            {"role": "user", "content": review_text},
        ],
        response_format=MovieReview,  # Pydantic Model 直接传入！
    )
    return completion.choices[0].message.parsed  # 已经是 MovieReview 实例

# 3. 使用
result = analyze_review("《沙丘2》视觉震撼，剧情紧凑，Hans Zimmer 配乐封神")
print(result.sentiment)  # "positive"
print(result.score)      # 8.5
print(result.keywords)   # ["视觉", "剧情", "配乐"]
```

**Pydantic vs Zod 对比（帮前端快速理解）：**

| Pydantic (Python) | Zod (TypeScript) | 作用 |
|---|---|---|
| `BaseModel` | `z.object({})` | 定义对象 Schema |
| `Field(ge=0, le=10)` | `z.number().min(0).max(10)` | 字段约束 |
| `field_validator` | `.refine()` | 自定义验证 |
| `model.model_dump()` | `schema.parse()` | 序列化 |
| `model.model_json_schema()` | `zodToJsonSchema()` | 生成 JSON Schema |

**面试话术：**

> "LLM 输出是不可控的字符串，但下游 API、数据库都需要结构化数据。Pydantic 就是 Python 的 Zod——定义 Schema，强制验证 LLM 输出的类型和约束。OpenAI 的 Structured Output 直接支持传入 Pydantic Model，输出保证符合 Schema。验证失败可以自动重试，形成 LLM → Parse → Validate → Retry 的闭环。"

---

## 11.8 FastAPI + SSE 实现 AI 接口

### Q: 如何用 FastAPI 实现一个生产级的 AI 聊天接口？要支持流式输出。

**FastAPI 是 Python AI 应用的首选 Web 框架——异步原生、类型安全（Pydantic 集成）、自动生成 API 文档、性能媲美 Go/Node.js。对前端来说，FastAPI ≈ Express + Zod + Swagger 的合体。**

**完整的生产级 AI 聊天接口：**

```python
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
import asyncio
import json
import time
import uuid

app = FastAPI(title="AI Chat API")

# CORS（前端跨域）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = AsyncOpenAI()

# ========== 请求/响应 Schema ==========
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = None
    model: str = Field(default="gpt-4o-mini")
    stream: bool = Field(default=True)

class ChatResponse(BaseModel):
    id: str
    content: str
    model: str
    usage: dict

# ========== 流式接口 ==========
async def generate_stream(request: ChatRequest):
    """SSE 生成器"""
    request_id = str(uuid.uuid4())
    start_time = time.time()
    full_content = ""

    try:
        # 发送开始事件
        yield f"data: {json.dumps({'event': 'start', 'id': request_id})}\n\n"

        stream = await client.chat.completions.create(
            model=request.model,
            messages=[{"role": "user", "content": request.message}],
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                full_content += token
                yield f"data: {json.dumps({'event': 'token', 'data': token})}\n\n"

        # 发送完成事件
        latency = time.time() - start_time
        yield f"data: {json.dumps({'event': 'done', 'id': request_id, 'latency': round(latency, 2)})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"

@app.post("/v1/chat")
async def chat(request: ChatRequest):
    if request.stream:
        return StreamingResponse(
            generate_stream(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # Nginx 关闭缓冲
            },
        )
    else:
        # 非流式
        response = await client.chat.completions.create(
            model=request.model,
            messages=[{"role": "user", "content": request.message}],
        )
        return ChatResponse(
            id=str(uuid.uuid4()),
            content=response.choices[0].message.content,
            model=request.model,
            usage=dict(response.usage),
        )

# ========== 健康检查 ==========
@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}
```

**前端对接代码：**

```typescript
// React Hook: useStreamChat
function useStreamChat() {
  const [content, setContent] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const sendMessage = async (message: string) => {
    setIsLoading(true);
    setContent('');

    const response = await fetch('/v1/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, stream: true }),
    });

    const reader = response.body!.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const text = decoder.decode(value);
      for (const line of text.split('\n\n')) {
        if (!line.startsWith('data: ')) continue;
        const event = JSON.parse(line.slice(6));
        if (event.event === 'token') {
          setContent(prev => prev + event.data);
        }
      }
    }
    setIsLoading(false);
  };

  return { content, isLoading, sendMessage };
}
```

**面试话术：**

> "FastAPI 是 AI 应用的 Express——异步原生，Pydantic 做请求验证（等于自带 Zod），自动生成 Swagger 文档。流式输出用 StreamingResponse + async generator 实现 SSE。前端用 fetch + ReadableStream 接收。生产环境要加 CORS、请求限流、错误处理、健康检查。"

---

## 11.9 AI 应用测试策略

### Q: AI 应用的输出不确定，怎么测试？有哪些测试策略？

**AI 应用测试的核心挑战是"输出不确定性"——同样的输入，每次输出可能不同。解决思路：不测精确值，测属性（格式、长度、是否包含关键信息）。分三层：Mock 测试（不调 LLM）、属性测试（调 LLM 但验证属性）、评估测试（LLM-as-a-Judge）。**

**AI 测试金字塔：**

```
            /\
           /  \   评估测试（Eval）
          / E  \  LLM-as-a-Judge, RAGAS
         /──────\
        /        \  集成测试（属性测试）
       /   IT     \ 调真实 LLM，验证输出属性
      /────────────\
     /              \ 单元测试（Mock）
    /      UT        \ Mock LLM 响应，测业务逻辑
   /──────────────────\
```

**层级 1：单元测试（Mock LLM，不花钱）**

```python
import pytest
from unittest.mock import AsyncMock, patch

# 被测函数
async def classify_sentiment(text: str) -> str:
    response = await llm.ainvoke(f"分类情感：{text}")
    return response.content.strip().lower()

# Mock 测试——不调真实 LLM
@pytest.mark.asyncio
async def test_classify_sentiment_positive():
    with patch("app.llm.ainvoke", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value.content = "positive"

        result = await classify_sentiment("I love this product!")

        assert result == "positive"
        mock_llm.assert_called_once()

@pytest.mark.asyncio
async def test_classify_handles_llm_error():
    """测试 LLM 报错时的降级逻辑"""
    with patch("app.llm.ainvoke", side_effect=Exception("API Error")):
        result = await classify_sentiment("test")
        assert result == "unknown"  # 应该返回默认值，不能崩
```

**层级 2：属性测试（调真实 LLM，验证格式和约束）**

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_movie_review_output_format():
    """验证 LLM 输出符合 Pydantic Schema"""
    result = await analyze_review("这部电影太好看了，强烈推荐！")

    # 不测精确值，测属性
    assert isinstance(result, MovieReview)
    assert result.sentiment in {"positive", "neutral", "negative"}
    assert 0 <= result.score <= 10
    assert len(result.keywords) >= 1
    assert len(result.summary) <= 200

@pytest.mark.integration
@pytest.mark.asyncio
async def test_rag_contains_source():
    """验证 RAG 回答引用了知识库文档"""
    result = await rag_query("公司退货政策是什么？")

    assert result.answer is not None
    assert len(result.sources) > 0  # 必须有引用来源
    assert any("退货" in s.content for s in result.sources)
```

**层级 3：评估测试（LLM-as-a-Judge）**

```python
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

def test_rag_quality():
    """用 DeepEval 做 RAG 质量评估"""
    test_case = LLMTestCase(
        input="什么是 RAG？",
        actual_output=rag_system("什么是 RAG？"),
        expected_output="RAG 是检索增强生成...",
        retrieval_context=["RAG = Retrieval-Augmented Generation..."],
    )

    metrics = [
        AnswerRelevancyMetric(threshold=0.7),  # 回答相关性
        FaithfulnessMetric(threshold=0.8),     # 忠实度（不编造）
    ]

    evaluate(test_cases=[test_case], metrics=metrics)
```

**面试话术：**

> "AI 测试分三层：单元测试 Mock 掉 LLM 只测业务逻辑（不花钱、速度快）；集成测试调真实 LLM 但只验证输出属性（格式对不对、字段全不全）；评估测试用 LLM-as-a-Judge 打分（相关性、忠实度）。核心原则是'不测精确值，测属性'，就像前端 Snapshot 测试不看每个像素，而是看整体结构。"

---

## 11.10 NL2SQL（自然语言转 SQL）

### Q: NL2SQL 的完整 Pipeline 是什么？如何提高准确率？

**NL2SQL = Natural Language to SQL，让用户用自然语言查询数据库。核心挑战：LLM 不知道你的表结构，生成的 SQL 可能语法错、逻辑错、甚至产生安全问题。需要一套完整的 Pipeline 来保障准确性和安全性。**

**NL2SQL 完整 Pipeline：**

```
用户问题："上个月销售额最高的前5个产品是什么？"
    │
    ▼
┌──────────────────┐
│ 1. Schema 检索    │  ← 从 Schema Registry 获取相关表结构
│ (哪些表相关？)    │     embedding 匹配 or 规则路由
└────────┬─────────┘
         │  tables: products, orders, order_items
         ▼
┌──────────────────┐
│ 2. SQL 生成       │  ← LLM + Prompt（包含表结构 + 示例）
│ (Text → SQL)     │
└────────┬─────────┘
         │  SELECT p.name, SUM(oi.amount) as total ...
         ▼
┌──────────────────┐
│ 3. SQL 验证       │  ← 语法检查 + 安全检查（防注入/防 DROP）
│ (合法？安全？)    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 4. SQL 执行       │  ← 只读连接 + 超时 + 行数限制
│ (安全执行)       │
└────────┬─────────┘
         │  [(iPhone 15, 500000), (MacBook, 350000), ...]
         ▼
┌──────────────────┐
│ 5. 结果解读       │  ← LLM 将查询结果转为自然语言
│ (SQL结果 → 回答)  │
└──────────────────┘
    │
    ▼
"上个月销售额 Top 5 产品：1. iPhone 15（50万）2. MacBook（35万）..."
```

**核心实现：**

```python
from pydantic import BaseModel, Field
from openai import OpenAI

client = OpenAI()

# Schema 信息（实际项目中从数据库元数据自动生成）
SCHEMA_CONTEXT = """
Table: products (id INT PK, name VARCHAR, category VARCHAR, price DECIMAL)
Table: orders (id INT PK, user_id INT, created_at TIMESTAMP, status VARCHAR)
Table: order_items (id INT PK, order_id INT FK→orders.id, product_id INT FK→products.id, quantity INT, amount DECIMAL)

示例:
Q: 每个类目的平均价格？
SQL: SELECT category, AVG(price) as avg_price FROM products GROUP BY category;
"""

class SQLResult(BaseModel):
    sql: str = Field(description="生成的 SQL 语句")
    explanation: str = Field(description="SQL 逻辑解释")

def generate_sql(question: str) -> SQLResult:
    completion = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"""你是 SQL 专家。根据用户问题生成 PostgreSQL 查询。

数据库 Schema：
{SCHEMA_CONTEXT}

规则：
1. 只生成 SELECT 语句，禁止 INSERT/UPDATE/DELETE/DROP
2. 必须使用 LIMIT（最大 1000 行）
3. 日期函数使用 PostgreSQL 语法
4. 使用表别名提高可读性"""},
            {"role": "user", "content": question},
        ],
        response_format=SQLResult,
    )
    return completion.choices[0].message.parsed

def validate_sql(sql: str) -> bool:
    """SQL 安全验证"""
    dangerous = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "EXEC", "--", ";"]
    sql_upper = sql.upper()
    for keyword in dangerous:
        if keyword in sql_upper:
            if keyword == ";" and sql_upper.count(";") == 1 and sql_upper.strip().endswith(";"):
                continue  # 允许末尾的分号
            return False
    return True
```

**提高 NL2SQL 准确率的 5 大策略：**

| 策略 | 描述 | 效果 |
|------|------|------|
| **Few-shot 示例** | 在 Prompt 中放 5-10 个同类型的 Q→SQL 示例 | +15-20% 准确率 |
| **Schema 精简** | 只传相关表的 Schema，不是全部表 | 减少干扰，+10% |
| **Self-correction** | 执行失败后把错误信息反馈给 LLM 重新生成 | 修复 30-50% 的语法错误 |
| **Query 分解** | 复杂问题拆分为多个子查询 | 提升复杂查询准确率 |
| **Schema 增强** | 给字段加 description、示例值、业务含义 | +10-15% |

**面试话术：**

> "NL2SQL 的 Pipeline 是五步：Schema 检索（找到相关表）→ SQL 生成（LLM + 表结构 + Few-shot）→ SQL 验证（防注入、只允许 SELECT）→ 安全执行（只读连接 + 超时 + 行数限制）→ 结果解读（LLM 把查询结果翻译成自然语言）。提高准确率的核心是 Few-shot 示例和 Schema 精简——给 LLM 看几个例子比写一堆规则有效得多。"

---

## 本章总结

```
┌──────────────────────────────────────────────────────────────┐
│                   AI 应用工程化知识体系                         │
│                                                              │
│  通信层                                                       │
│  ├── SSE 流式输出 → Go/Python 后端 + JS 前端                   │
│  └── LLM API 防护 → 重试 + 超时 + 降级 + 熔断                  │
│                                                              │
│  框架层                                                       │
│  ├── LangChain LCEL → 管道语法 | 组件化                        │
│  ├── LangGraph → 状态图 + 循环 + 人工介入                      │
│  └── 低代码平台 → Dify(RAG) / Coze(Bot) / n8n(工作流)         │
│                                                              │
│  工程层                                                       │
│  ├── asyncio → Semaphore 限流 + gather 并发                   │
│  ├── Pydantic → LLM 输出结构化验证                             │
│  ├── FastAPI → SSE + 类型安全 + 自动文档                       │
│  └── 测试 → Mock / 属性测试 / LLM-as-a-Judge                  │
│                                                              │
│  应用层                                                       │
│  └── NL2SQL → Schema 检索 + SQL 生成 + 安全验证 + 结果解读      │
└──────────────────────────────────────────────────────────────┘
```

---

## 11.11 LlamaIndex vs LangChain

### Q: LlamaIndex 和 LangChain 有什么区别？什么时候选哪个？

| 维度 | LangChain | LlamaIndex |
|------|-----------|------------|
| **定位** | 通用 LLM 应用框架 | RAG 专用框架 |
| **核心能力** | Chain 编排、Agent、Memory、Tools | 数据连接、索引构建、查询引擎 |
| **数据加载** | 基础（需要第三方 Loader） | 强大（160+ 内置 Loader：PDF/HTML/DB/API） |
| **索引类型** | 只有 VectorStore | Vector / List / Tree / Keyword / Knowledge Graph |
| **查询优化** | 需手动实现 | 内置 Sub-Question、Router、HyDE |
| **Agent** | 强（AgentExecutor、LangGraph） | 基础（ReAct Agent） |
| **学习曲线** | 陡（抽象多） | 相对平缓（聚焦 RAG） |

**选型决策：**

```
你的项目核心是什么？
├── RAG / 知识库问答 → LlamaIndex（专为 RAG 优化）
├── Agent / 工具调用 → LangChain + LangGraph
├── 都需要 → LlamaIndex 做数据层 + LangChain 做编排层
└── 简单项目 → 直接用 OpenAI SDK，不用框架
```

**面试话术：**
> "LangChain 是通用 LLM 应用框架，什么都能做但抽象多。LlamaIndex 专注 RAG，数据连接和查询优化做得更深。我的建议是 RAG 重的项目用 LlamaIndex，Agent 重的用 LangChain + LangGraph，简单项目直接用 SDK。"

---

## 11.12 OpenAI Assistants API

### Q: OpenAI Assistants API 是什么？Thread / Run / File Search / Code Interpreter 怎么用？

**Assistants API = OpenAI 提供的托管 Agent 方案，内置了记忆、工具调用、文件处理。**

**核心概念：**

| 概念 | 作用 | 类比 |
|------|------|------|
| **Assistant** | 定义 Agent 的角色、工具、模型 | 组件定义 |
| **Thread** | 一次对话的上下文（自动管理历史） | 会话 Session |
| **Run** | 一次执行（触发 Agent 思考和行动） | API 请求 |
| **File Search** | 内置 RAG（上传文件自动索引检索） | 向量检索 |
| **Code Interpreter** | 沙箱执行 Python（数据分析、画图） | 在线 REPL |

```python
from openai import OpenAI
client = OpenAI()

# 1. 创建 Assistant
assistant = client.beta.assistants.create(
    name="数据分析师",
    instructions="你是数据分析专家，用 Python 分析数据并画图",
    model="gpt-4o",
    tools=[
        {"type": "file_search"},        # 内置 RAG
        {"type": "code_interpreter"},    # 沙箱 Python
    ]
)

# 2. 创建 Thread + 发消息
thread = client.beta.threads.create()
client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="分析上传的销售数据，画出月度趋势图"
)

# 3. Run（触发执行）
run = client.beta.threads.runs.create_and_poll(
    thread_id=thread.id,
    assistant_id=assistant.id
)

# 4. 获取结果
messages = client.beta.threads.messages.list(thread_id=thread.id)
```

**Assistants API vs 自建 Agent：**

| 维度 | Assistants API | 自建（LangChain/LangGraph） |
|------|---------------|---------------------------|
| 开发速度 | 快（托管服务） | 慢（需自己搭建） |
| 灵活性 | 低（受 API 限制） | 高（完全自定义） |
| 成本 | 较高（额外计费） | 可控 |
| 可观测性 | 弱（黑盒） | 强（自定义日志） |
| 适用 | 快速原型、简单 Agent | 生产级、复杂 Agent |

**面试话术：**
> "Assistants API 的优势是开箱即用——内置了 RAG（File Search）和代码执行（Code Interpreter），Thread 自动管理对话历史。但生产环境我更倾向自建，因为可观测性和灵活性更好。快速验证 idea 时用 Assistants API，上线时迁移到 LangGraph。"

---

**面试高频考点 Top 5：**

| 排名 | 考点 | 关键答题点 |
|------|------|-----------|
| 1 | SSE 流式输出 | SSE vs WebSocket 选型、fetch + ReadableStream 实现 |
| 2 | LLM API 重试 | 指数退避 + Full Jitter、降级链路、Semaphore 限流 |
| 3 | LangChain LCEL | 管道语法 `\|`、Runnable 接口、RunnableParallel |
| 4 | Pydantic 验证 | LLM 输出 Schema 定义、OpenAI Structured Output |
| 5 | NL2SQL | 五步 Pipeline、安全验证、Few-shot 提升准确率 |

---

**章节导航：**
[上一章：10-multimodal](./10-multimodal.md) | [目录](./README.md) | [下一章：12-production](./12-production.md)