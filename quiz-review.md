# 五题复盘：RAG / 推理优化 / Agent / 流式输出

> 日期：2026-07-20
>
> 用途：记录一次自测后的高频薄弱点。题干保留在 `quiz-history.md`，这里整理成可复习版本。

---

## 1. RAG 分块策略：固定分块、语义分块、Late Chunking 的区别

### 题目

RAG 中固定分块、语义分块、Late Chunking 分别是什么？各自适合什么场景？

### 答案

**固定分块**是按字符数或 token 数切分文档，比如 `512 tokens + 10% overlap`。它简单、稳定、可控，适合作为生产 baseline；缺点是可能切断语义，比如把一个定义、一个代码函数或一个代词指代切到两个 chunk 里。

**语义分块**是先按句子或段落计算语义相似度，在语义断裂处切分。它更容易保留完整语义，适合非结构化长文本；缺点是需要额外 embedding 计算，成本更高，而且切分结果受阈值影响，稳定性不如固定分块。

**Late Chunking**是先让长上下文 embedding 模型读取完整文档，得到每个 token 的上下文化向量，再按照 chunk 边界对 token 向量做 pooling。它的关键不是把全文塞进每个 chunk 文本里，而是让每个 token 在编码阶段已经通过 self-attention 看过全文上下文。

普通 chunking：

```text
先切块 -> 每个 chunk 单独 embedding
```

Late Chunking：

```text
完整文档 -> token-level contextual embeddings -> 按 chunk 边界 mean pooling
```

例如：

```text
张三加入了 OpenAI。他后来负责多模态模型。
```

如果“他后来负责多模态模型”单独 embedding，模型可能不知道“他”是谁。Late Chunking 先把完整文档送进模型，“他”这个 token 的向量已经 attention 过前文“张三”，所以之后即使按 chunk 聚合，chunk embedding 也携带了全局上下文。

### 记忆钩子

固定分块看长度，语义分块看语义断点，Late Chunking 先全局编码再局部聚合。

### 关联章节

- `05-rag.md`：5.2 文档处理与分块策略

---

## 2. Rerank：BiEncoder 粗召回和 CrossEncoder 精排的区别

### 题目

Rerank 是为了解决什么问题？BiEncoder 粗召回和 CrossEncoder 精排的核心区别是什么？

### 答案

Rerank 解决的是初步召回结果里 chunk 过多、噪声较多、相关度排序不够准的问题。典型做法是先用向量检索从大规模文档中召回 Top-50 或 Top-100，再用 Reranker 精排，选出 Top-5 或 Top-10 放进上下文。

**BiEncoder** 是双塔结构：

```text
query -> query embedding
chunk -> chunk embedding
query embedding 和 chunk embedding 计算相似度
```

chunk embedding 可以离线预计算，所以速度快，适合从百万级文档中做粗召回。缺点是 query 和 chunk 在编码时没有充分交互，只是在最后用向量相似度比较，精度有限。

**CrossEncoder** 是把 query 原文和 chunk 原文拼在一起，让模型联合理解后输出相关性分数：

```text
[CLS] query [SEP] chunk [SEP]
  -> Transformer Encoder
  -> 取 [CLS] 最终向量
  -> 分类/回归 head
  -> relevance score
```

CrossEncoder 更准，是因为 query token 和 chunk token 在 Transformer 层中可以互相 attention，模型能判断 chunk 是否真的回答了问题，而不是只判断两个整体向量像不像。

CrossEncoder 更慢，是因为每个 `(query, chunk)` pair 都要实时跑一遍模型，不能像 BiEncoder 一样提前算好所有 chunk 向量。

### 记忆钩子

BiEncoder 是“分别编码后相似度搜索”，CrossEncoder 是“拼在一起整体理解后打分”。

### 关联章节

- `05-rag.md`：5.4 Rerank 重排序

---

## 3. PagedAttention 如何解决 KV Cache 内存碎片化

### 题目

PagedAttention 是什么？KV Cache 这里是否必须地址连续？为什么强调“逻辑连续，物理不连续”？

### 答案

PagedAttention 是 vLLM 提出的 KV Cache 管理方式，借鉴操作系统虚拟内存分页。它把 GPU 显存中的 KV Cache 切成固定大小的 block，通过 block table 维护逻辑 token 位置到物理 block 的映射。

传统 KV Cache 管理的问题不是“没有连续地址就一定存不进去”，而是高效推理通常希望每个请求拥有一段逻辑连续的 KV 序列。传统实现往往给每个请求预分配一大段连续空间，例如按照最大生成长度 2048 tokens 分配；如果请求实际只生成 300 tokens，剩下空间就浪费了。

PagedAttention 的做法：

```text
逻辑上：
请求 A 的 KV Cache 是连续的 token 序列
token 0, token 1, token 2, ...

物理上：
这些 token 的 KV 分散在多个 block 中
block 7, block 23, block 5, ...

Block Table：
逻辑 block 0 -> 物理 block 7
逻辑 block 1 -> 物理 block 23
逻辑 block 2 -> 物理 block 5
```

这样模型视角里仍然是连续序列，但底层显存可以按 block 动态分配。新 token 生成到需要新 block 时再申请，不需要一开始按最大长度分满。

它主要解决：

- 预分配浪费：不知道最终输出多长，不必按 max tokens 一次性分配。
- 内存碎片：固定大小 block 更容易复用空闲空间。
- 并发利用率低：显存能容纳更多请求的 KV Cache。
- 前缀共享：相同 system prompt 或共享前缀可以通过 Copy-on-Write 复用 block。

### 记忆钩子

PagedAttention 不是让 KV Cache 不连续，而是让“逻辑连续”从“物理连续”里解耦出来。

### 关联章节

- `09-inference-optimization.md`：9.5 PagedAttention

---

## 4. Agent 为什么容易出现死循环

### 题目

Agent 在生产环境中为什么容易死循环？应该设计哪些保护机制？

### 答案

Agent 容易死循环，是因为它不是一次性回答，而是一个开放目标下的循环执行系统：

```text
Thought -> Action -> Observation -> Thought -> Action -> Observation ...
```

每一轮都由 LLM 根据当前状态决定下一步。如果目标、工具结果、完成条件或错误处理不清晰，就可能原地打转。

常见原因：

1. **目标没有清晰完成条件**
   例如“帮我优化系统”，如果没有定义验收标准，Agent 不知道什么时候该停止。

2. **工具返回缺少有效信息**
   工具返回空结果、失败、超时，Agent 又没有换策略，就会反复调用同一个工具。

3. **重复尝试同一动作**
   同一个 tool 和同一组参数已经失败，模型仍然误以为再试一次可能成功。

4. **规划粒度太粗**
   没有明确阶段状态，Agent 不知道当前任务走到哪一步，也不知道哪些子任务已经完成。

5. **Observation 没有被正确利用**
   工具已经返回“权限不足”“文件不存在”“参数错误”，但 Agent 没把它当作终止或换策略信号。

6. **Multi-Agent 乒乓循环**
   Agent A 把任务交给 B，B 又交给 A，形成反复转交。

生产保护机制：

- `max_iterations`：硬性迭代上限。
- `max_time`：超时中断。
- 重复动作检测：对 `tool + params` 做签名，同一动作多次出现就终止或换策略。
- token / cost budget：超出预算停止或降级。
- 进度检测：连续多步没有新信息或状态变化，就触发降级、总结或人工介入。
- Human-in-the-Loop：高风险、不确定或无进展时让人审批。

### 记忆钩子

Agent 死循环的本质是“开放目标 + 循环执行 + 概率决策”，所以必须有终止条件、预算和进度判断。

### 关联章节

- `06-ai-agent.md`：6.13 Agent 防死循环与容错

---

## 5. AI 流式输出场景下 SSE 和 WebSocket 如何选择

### 题目

AI 聊天的流式输出场景下，SSE 和 WebSocket 应该怎么选？为什么很多 AI 产品优先用 SSE？

### 答案

AI 聊天流式输出通常优先用 **SSE（Server-Sent Events）**，因为 LLM 生成结果主要是服务端向客户端单向推送 token。

SSE 的特点：

- 单向：服务器 -> 客户端，正好匹配 token streaming。
- 基于 HTTP：代理、CDN、网关兼容性好。
- 浏览器原生支持 `EventSource`。
- 自动重连能力更友好。
- 服务端实现相对轻量。

WebSocket 的特点：

- 双向通信。
- 适合客户端和服务端都频繁发送消息的场景。
- 连接状态管理更重，代理兼容性和运维复杂度更高。

选择建议：

```text
普通 AI Chat / 文本生成 / 服务端逐 token 推送 -> SSE
协同编辑 / 实时游戏 / 多端同步 / 高频双向控制 -> WebSocket
```

面试中不要绝对说“SSE 性能一定更好”。更稳的说法是：SSE 更轻量、更贴合 AI token streaming 的单向输出场景。

### 记忆钩子

LLM 流式回答是“服务器说、客户端听”，所以 SSE 通常够用；需要双方频繁说话，再上 WebSocket。

### 关联章节

- `11-ai-engineering.md`：11.1 流式输出（SSE vs WebSocket）
