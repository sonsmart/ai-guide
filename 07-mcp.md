# 07 - MCP 协议与工具系统

> **难度：** ⭐⭐⭐⭐ | **定位：** AI 应用与外部世界的连接层，Agent 开发必考
>
> **前端类比：** MCP 之于 AI，就像 REST API 标准之于前端——在 REST 出现之前，每个后端都有自定义协议（SOAP、XML-RPC），前端要对接 N 个后端就要写 N 套适配代码。MCP 做的事情一模一样：统一 AI 与工具之间的通信协议。MCP Server ≈ Express.js middleware，接收标准化请求，执行具体逻辑，返回标准化结果。

## 本章知识树

```
MCP 协议与工具系统
├── 7.1 MCP 基础（定义、为什么是 "AI 世界的 USB"）
├── 7.2 MCP 架构（Client / Server / Transport 三层）
├── 7.3 MCP vs Function Calling vs Tools API
├── 7.4 MCP Server 开发实战
├── 7.5 MCP 安全与权限管理
└── 7.6 2026 MCP 生态与趋势
```

---

## 7.1 MCP 基础

### Q: 什么是 MCP？为什么说它是 "AI 世界的 USB"？

**MCP（Model Context Protocol）是 Anthropic 于 2024 年 11 月开源的标准化协议，定义了 AI 模型与外部工具、数据源之间的通信方式。**

```
MCP 出现之前（混乱时代）：

  ChatGPT ──── 自定义 Plugin 协议 ──── Gmail
  Claude  ──── 自定义 API 适配 ────── Gmail
  Gemini  ──── 又一套适配代码 ────── Gmail

  N 个 AI × M 个工具 = N × M 套适配代码  💀

MCP 出现之后（标准化时代）：

  ChatGPT ─┐                    ┌── Gmail MCP Server
  Claude  ─┼── MCP Protocol ───┼── Slack MCP Server
  Gemini  ─┘                    └── GitHub MCP Server

  N 个 AI × M 个工具 = N + M 套代码  ✅
```

**为什么叫 "AI 世界的 USB"？**

| 类比维度 | USB | MCP |
|---------|-----|-----|
| **解决的问题** | 每种外设一种接口（串口、并口、PS/2） | 每个 AI 平台一套工具协议 |
| **核心价值** | 统一物理接口 + 通信协议 | 统一 AI-工具通信协议 |
| **即插即用** | 插上 U 盘就能用 | 注册 MCP Server 就能用 |
| **生态效应** | 外设厂商只需适配 USB | 工具只需实现 MCP Server |

**MCP 的核心设计理念：**

1. **协议标准化** — 基于 JSON-RPC 2.0，所有消息格式统一
2. **关注点分离** — AI 模型不需要知道工具如何实现，只需知道工具"能做什么"
3. **Server 可复用** — 一个 GitHub MCP Server 可以被所有支持 MCP 的 AI Client 使用
4. **安全第一** — 内置权限模型，工具调用需要用户授权

**前端类比：** 还记得 jQuery 时代吗？每个浏览器的 DOM API 都不一样，你要写一堆 `if (IE) {} else if (Firefox) {}` 适配代码。jQuery 统一了 DOM 操作接口。MCP 对 AI 工具做的就是同样的事——提供一个统一的"AI 工具操作接口"。

> **面试话术：** "MCP 是 Anthropic 在 2024 年底开源的标准化协议，核心思路是把 AI 与工具的 M×N 集成问题简化为 M+N 问题，类似 USB 统一了外设接口。它基于 JSON-RPC 2.0，定义了 Client-Server 架构，让任何 AI 平台都能通过同一套协议调用同一个工具 Server，实现真正的即插即用。"

---

## 7.2 MCP 架构

### Q: MCP 的架构是怎样的？Client、Server、Transport 三层分别负责什么？

**MCP 采用 Client-Server 架构，通过 Transport 层实现通信。三层职责清晰分离。**

```
┌──────────────────────────────────────────────────────┐
│                    Host Application                   │
│              (Claude Desktop / IDE / App)              │
│                                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │ MCP Client  │  │ MCP Client  │  │ MCP Client  │  │
│  │     #1      │  │     #2      │  │     #3      │  │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  │
└─────────┼────────────────┼────────────────┼──────────┘
          │                │                │
     Transport        Transport        Transport
     (stdio)          (SSE)            (HTTP)
          │                │                │
  ┌───────┴──────┐ ┌──────┴───────┐ ┌──────┴───────┐
  │  MCP Server  │ │  MCP Server  │ │  MCP Server  │
  │  (本地文件)   │ │  (Slack API) │ │  (数据库)     │
  └──────────────┘ └──────────────┘ └──────────────┘
```

**三层详解：**

**1. MCP Client（客户端）**
- 嵌入在 Host 应用内部（如 Claude Desktop、Cursor、VS Code）
- 负责与 MCP Server 建立连接、发送请求、接收响应
- 一个 Host 可以同时连接多个 MCP Client，每个 Client 对应一个 Server
- 类比：浏览器中的 `fetch` API

**2. MCP Server（服务端）**
- 暴露具体能力：Tools（工具）、Resources（资源）、Prompts（提示词模板）
- 轻量级进程，可以是本地脚本也可以是远程服务
- 类比：Express.js 服务，定义路由和处理逻辑

**3. Transport（传输层）**
- 定义 Client 和 Server 之间的通信方式

| Transport 类型 | 协议 | 适用场景 | 前端类比 |
|---------------|------|---------|---------|
| **stdio** | 标准输入/输出 | 本地进程，CLI 工具 | `child_process.spawn()` |
| **SSE (HTTP+SSE)** | Server-Sent Events | 远程服务（已弃用，legacy） | `EventSource` API |
| **Streamable HTTP** | HTTP POST + 可选流 | 2025+ 推荐的远程方式，取代 SSE | `fetch` + `ReadableStream` |

**MCP Server 的四大能力（Capabilities）：**

```
MCP Server 能力
├── Tools（工具）       ← 模型可以调用的函数（如：搜索文件、发送邮件）
├── Resources（资源）   ← 模型可以读取的数据（如：文件内容、数据库记录）
├── Prompts（提示词）   ← 预定义的 Prompt 模板（如：代码审查模板）
└── Sampling（采样）    ← Server 反向请求 Client 让 LLM 生成内容
```

| 能力 | 方向 | 描述 | 前端类比 |
|------|------|------|---------|
| **Tools** | Client → Server | "帮我执行这个操作" | POST 请求 |
| **Resources** | Client → Server | "给我这份数据" | GET 请求 |
| **Prompts** | Client → Server | "给我这个模板" | 静态资源请求 |
| **Sampling** | Server → Client | "帮我调用 LLM 生成" | WebSocket 反向推送 |

**消息通信基于 JSON-RPC 2.0：**

```json
// Client → Server: 调用工具
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "search_files",
    "arguments": {
      "query": "MCP protocol",
      "path": "/projects"
    }
  }
}

// Server → Client: 返回结果
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Found 3 files matching 'MCP protocol'..."
      }
    ]
  }
}
```

> **面试话术：** "MCP 是三层架构——Client 嵌入在 Host 应用中负责发起请求，Server 暴露 Tools/Resources/Prompts/Sampling 四大能力，Transport 层支持 stdio、SSE 和 Streamable HTTP 三种传输方式。消息格式基于 JSON-RPC 2.0。整个设计思路和 Web 前后端分离很像：Transport 层 ≈ HTTP 协议，Server ≈ REST API，Client ≈ 浏览器的 fetch。"

---

## 7.3 MCP vs Function Calling vs Tools API

### Q: MCP、Function Calling、Tools API 三者有什么区别？各自适用什么场景？

**这三个概念经常混淆，但它们处于不同层级，解决不同问题。**

```
层级关系：

  ┌─────────────────────────────────────────────┐
  │          应用层（Application Layer）          │
  │                                              │
  │   MCP Protocol                               │
  │   (跨平台的工具连接标准)                       │
  └──────────────────┬──────────────────────────┘
                     │ 底层可以使用
  ┌──────────────────▼──────────────────────────┐
  │          模型层（Model Layer）                │
  │                                              │
  │   Function Calling / Tools API               │
  │   (模型厂商的原生工具调用能力)                  │
  └─────────────────────────────────────────────┘
```

**详细对比表：**

| 维度 | Function Calling | Tools API | MCP |
|------|-----------------|-----------|-----|
| **提出者** | OpenAI (2023.06) | 各模型厂商 | Anthropic (2024.11) |
| **本质** | 模型输出结构化的函数调用意图 | Function Calling 的升级/别名 | 跨平台的工具连接协议 |
| **标准化程度** | 厂商专属 | 厂商专属 | 开放标准 |
| **执行者** | 开发者自己执行函数 | 开发者自己执行函数 | MCP Server 执行 |
| **跨模型复用** | 不行，换模型要改代码 | 不行 | 可以，任何 MCP Client 通用 |
| **发现机制** | 无，工具列表硬编码 | 无 | 有，Server 动态暴露工具列表 |
| **传输协议** | HTTP API | HTTP API | JSON-RPC (stdio/SSE/HTTP) |
| **前端类比** | 手写 `XMLHttpRequest` | `fetch` API | `axios` + OpenAPI Spec |

**代码对比——以"查询天气"为例：**

**方式一：OpenAI Function Calling**

```python
# 1. 定义函数 schema（和 OpenAI API 绑定）
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取指定城市的天气",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名"}
            },
            "required": ["city"]
        }
    }
}]

# 2. 调用 OpenAI API
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "北京天气怎么样？"}],
    tools=tools
)

# 3. 解析模型返回的函数调用意图
tool_call = response.choices[0].message.tool_calls[0]
# tool_call.function.name == "get_weather"
# tool_call.function.arguments == '{"city": "北京"}'

# 4. 开发者自己执行函数并返回结果（MCP 不需要这步！）
weather_data = call_weather_api(json.loads(tool_call.function.arguments))

# 5. 把结果塞回对话，让模型生成最终回答
messages.append({"role": "tool", "content": json.dumps(weather_data)})
final_response = openai.chat.completions.create(
    model="gpt-4", messages=messages
)
```

**方式二：MCP 方式**

```python
# MCP Server 端（写一次，所有 AI 都能用）
from mcp.server import Server
from mcp.types import Tool, TextContent

server = Server("weather-server")

@server.tool()
async def get_weather(city: str) -> list[TextContent]:
    """获取指定城市的天气"""
    data = await call_weather_api(city)
    return [TextContent(type="text", text=f"{city}: {data['temp']}°C")]

# 启动 Server
# mcp run weather-server

# Client 端（Claude Desktop / Cursor / 任何 MCP Client）
# 只需在配置文件中注册这个 Server，AI 自动发现并使用
# 用户说 "北京天气怎么样？" → AI 自动调用 get_weather("北京")
```

**关键区别总结：**

```
Function Calling:
  开发者定义 schema → 发给模型 → 模型输出调用意图 → 开发者执行 → 结果返回模型
  （5 步，开发者要做很多胶水代码）

MCP:
  Server 定义工具 → Client 自动发现 → 模型决定调用 → Server 自动执行 → 结果返回
  （全自动，开发者只需写 Server 逻辑）
```

**它们并非互斥，而是协作关系：** MCP Client 在底层可能使用 Function Calling 来让模型决定调用哪个工具，但开发者不需要关心这层——MCP 把它封装了。

> **面试话术：** "Function Calling 是模型厂商提供的原生能力，让模型输出结构化的函数调用意图，但执行和编排都要开发者自己做，而且绑定特定厂商。MCP 是更高层级的跨平台协议，把工具定义、发现、执行全部标准化了。类比前端：Function Calling 像手写 XMLHttpRequest，MCP 像 axios + OpenAPI——有自动发现、统一接口、跨平台复用。两者不是替代关系，MCP 底层仍可能使用 Function Calling 驱动模型决策。"

---

## 7.4 MCP Server 开发实战

### Q: 如何从零开发一个 MCP Server？请用实际例子说明。

**以构建一个"文件搜索 MCP Server"为例，演示完整的开发流程。**

**Step 1: 环境准备**

```bash
# 安装 MCP Python SDK
pip install mcp

# 项目结构
file-search-server/
├── server.py          # MCP Server 主逻辑
└── pyproject.toml     # 项目配置
```

**Step 2: 实现 MCP Server**

```python
# server.py
import os
import glob
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool, TextContent, Resource, ResourceTemplate
)

# 创建 Server 实例
server = Server("file-search-server")

# ============================================
# 1. 定义 Tool（工具）—— 模型可以"调用"的操作
# ============================================

@server.tool()
async def search_files(
    directory: str,
    pattern: str,
    max_results: int = 10
) -> list[TextContent]:
    """
    在指定目录中搜索匹配 pattern 的文件。

    Args:
        directory: 要搜索的目录路径
        pattern: 文件名匹配模式（支持 glob，如 *.py）
        max_results: 最大返回数量
    """
    search_path = os.path.join(directory, "**", pattern)
    matches = glob.glob(search_path, recursive=True)[:max_results]

    if not matches:
        return [TextContent(type="text", text=f"未找到匹配 '{pattern}' 的文件")]

    result = f"找到 {len(matches)} 个文件:\n"
    for f in matches:
        size = os.path.getsize(f)
        result += f"  - {f} ({size} bytes)\n"

    return [TextContent(type="text", text=result)]


@server.tool()
async def read_file_content(file_path: str) -> list[TextContent]:
    """
    读取文件内容。

    Args:
        file_path: 文件的完整路径
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return [TextContent(
            type="text",
            text=f"=== {file_path} ===\n{content}"
        )]
    except Exception as e:
        return [TextContent(type="text", text=f"读取失败: {str(e)}")]


# ============================================
# 2. 定义 Resource（资源）—— 模型可以"读取"的数据
# ============================================

@server.resource("file://{file_path}")
async def get_file_resource(file_path: str) -> str:
    """将文件暴露为 MCP Resource，模型可以直接读取。"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# ============================================
# 3. 定义 Prompt（提示词模板）
# ============================================

@server.prompt()
async def code_review_prompt(file_path: str) -> str:
    """生成代码审查的 Prompt 模板。"""
    with open(file_path, "r", encoding="utf-8") as f:
        code = f.read()
    return f"""请审查以下代码，关注：
1. 潜在 bug 和边界情况
2. 性能问题
3. 可读性和命名规范
4. 安全隐患

代码文件: {file_path}
```
{code}
```

请用中文回答，给出具体的改进建议。"""


# ============================================
# 4. 启动 Server
# ============================================

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

**Step 3: 注册到 MCP Client（以 Claude Desktop 为例）**

```json
// ~/Library/Application Support/Claude/claude_desktop_config.json
{
  "mcpServers": {
    "file-search": {
      "command": "python",
      "args": ["/path/to/file-search-server/server.py"],
      "env": {
        "PYTHONPATH": "/path/to/file-search-server"
      }
    }
  }
}
```

**Step 4: 工作流程**

```
用户: "帮我找一下项目里所有的 Python 测试文件"

Claude Desktop (Host)
    │
    ├─ MCP Client 发现 file-search Server 有 search_files 工具
    │
    ├─ 模型决定调用: search_files(directory="/project", pattern="test_*.py")
    │
    ├─ MCP Client → Transport (stdio) → MCP Server 执行搜索
    │
    ├─ Server 返回: "找到 5 个文件: test_auth.py, test_api.py ..."
    │
    └─ 模型基于结果生成自然语言回答给用户
```

**Decorator 模式解析（核心设计模式）：**

```python
# @server.tool() 做了什么？
# 1. 注册函数到 Server 的工具列表
# 2. 自动从函数签名 + type hints 生成 JSON Schema
# 3. 自动从 docstring 提取工具描述
# 4. 处理 JSON-RPC 请求/响应的序列化

# 类似前端的路由注册：
# @app.route("/api/search")     ← Express.js
# @server.tool()                ← MCP Server
```

**实战建议：**
- 每个 Tool 做一件事，保持单一职责（类似 RESTful API 设计）
- Docstring 要写清楚，因为模型会根据描述决定何时调用
- 参数用 type hints 标注，SDK 会自动生成 JSON Schema
- 错误处理要友好，返回可读的错误信息而非 traceback

> **面试话术：** "开发 MCP Server 和写 Express.js 中间件的体验非常像——用 decorator 注册 Tools/Resources/Prompts，SDK 自动处理协议细节。关键是 Tool 的 docstring 要写好，因为 AI 模型依赖描述来决定何时调用哪个工具。部署方式灵活，本地用 stdio 启动，远程用 SSE 或 Streamable HTTP。"

---

## 7.5 MCP 安全与权限管理

### Q: MCP 的安全模型是怎样的？如何防止工具被滥用？

**MCP 的安全设计遵循"最小权限 + 用户授权"原则，从多个层面构建安全体系。**

```
MCP 安全防护层级：

  ┌─────────────────────────────────────┐
  │  Layer 5: Audit Logging（审计日志）  │  ← 全链路追踪
  ├─────────────────────────────────────┤
  │  Layer 4: Rate Limiting（速率限制）  │  ← 防止滥用
  ├─────────────────────────────────────┤
  │  Layer 3: Input Validation（输入校验）│ ← 防注入攻击
  ├─────────────────────────────────────┤
  │  Layer 2: OAuth 2.0（身份认证）      │  ← 远程 Server 认证
  ├─────────────────────────────────────┤
  │  Layer 1: Permission Model（权限模型）│ ← 基础权限控制
  └─────────────────────────────────────┘
```

**Layer 1: 权限模型（Permission Model）**

MCP 规范要求"Human-in-the-loop"——关键工具调用必须经用户确认。

```
权限级别：

  ┌──────────┬────────────────────────────────┐
  │ 级别      │ 说明                           │
  ├──────────┼────────────────────────────────┤
  │ Read     │ 读取资源（低风险，可自动放行）    │
  │ Write    │ 修改数据（中风险，提示用户确认）  │
  │ Execute  │ 执行操作（高风险，必须用户授权）  │
  │ Admin    │ 管理操作（最高风险，需要额外认证） │
  └──────────┴────────────────────────────────┘
```

```python
# 在 MCP Server 中标注权限要求
@server.tool(
    annotations={
        "readOnlyHint": False,       # 这不是只读操作
        "destructiveHint": True,     # 这是破坏性操作
        "idempotentHint": False,     # 不是幂等的
        "openWorldHint": True        # 会访问外部服务
    }
)
async def delete_file(file_path: str) -> list[TextContent]:
    """删除指定文件（危险操作）。"""
    os.remove(file_path)
    return [TextContent(type="text", text=f"已删除: {file_path}")]
```

**Layer 2: OAuth 2.0 集成（远程 MCP Server）**

```
Remote MCP Server 认证流程：

  用户                MCP Client            MCP Server         OAuth Provider
   │                    │                      │                    │
   │  使用工具           │                      │                    │
   ├──────────────────>│                      │                    │
   │                    │  发现需要认证          │                    │
   │                    │─────────────────────>│                    │
   │                    │  401 + OAuth metadata │                    │
   │                    │<─────────────────────│                    │
   │  授权弹窗           │                      │                    │
   │<──────────────────│                      │                    │
   │  用户同意           │                      │                    │
   ├──────────────────>│                      │                    │
   │                    │  Authorization Code   │                    │
   │                    │──────────────────────────────────────────>│
   │                    │  Access Token         │                    │
   │                    │<─────────────────────────────────────────│
   │                    │  带 Token 的请求       │                    │
   │                    │─────────────────────>│                    │
   │                    │  工具执行结果          │                    │
   │                    │<─────────────────────│                    │
```

**Layer 3: 输入校验（防 Prompt Injection）**

```python
# 不安全的实现（直接拼接用户输入）
@server.tool()
async def query_db(sql: str) -> list[TextContent]:
    result = db.execute(sql)  # SQL 注入风险！
    return [TextContent(type="text", text=str(result))]

# 安全的实现（参数化查询 + 白名单）
@server.tool()
async def query_db(
    table: str,
    conditions: dict
) -> list[TextContent]:
    """查询数据库记录。"""
    # 白名单校验
    allowed_tables = {"users", "orders", "products"}
    if table not in allowed_tables:
        return [TextContent(type="text", text=f"不允许查询表: {table}")]

    # 参数化查询
    where_clause = " AND ".join(f"{k} = ?" for k in conditions)
    values = list(conditions.values())
    result = db.execute(
        f"SELECT * FROM {table} WHERE {where_clause}",
        values
    )
    return [TextContent(type="text", text=str(result))]
```

**Layer 4 & 5: 速率限制与审计日志**

```python
import time
import logging
from functools import wraps

# 速率限制装饰器
def rate_limit(max_calls: int, window_seconds: int):
    calls = []
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            now = time.time()
            calls[:] = [t for t in calls if now - t < window_seconds]
            if len(calls) >= max_calls:
                return [TextContent(
                    type="text",
                    text=f"速率限制: 每 {window_seconds}s 最多 {max_calls} 次调用"
                )]
            calls.append(now)
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# 审计日志
logger = logging.getLogger("mcp.audit")

@server.tool()
@rate_limit(max_calls=10, window_seconds=60)
async def send_email(to: str, subject: str, body: str) -> list[TextContent]:
    """发送邮件。"""
    logger.info(f"TOOL_CALL: send_email | to={to} | subject={subject}")
    # ... 发送逻辑
    logger.info(f"TOOL_RESULT: send_email | status=success")
    return [TextContent(type="text", text=f"邮件已发送至 {to}")]
```

**前端类比：** MCP 的安全模型和 Web 安全非常相似——OAuth 2.0 = 前端 SSO 登录，输入校验 = XSS/SQL 注入防护，Rate Limiting = API 限流（nginx `limit_req`），Audit Logging = 前端埋点 + 后端 access log。

> **面试话术：** "MCP 的安全设计是多层防护：底层是权限模型，要求关键操作必须用户确认（Human-in-the-loop）；远程 Server 通过 OAuth 2.0 做身份认证；工具层面做输入校验防 Prompt Injection，用参数化查询和白名单替代直接拼接；运维层面有速率限制防滥用，审计日志做全链路追踪。核心原则是最小权限和纵深防御。"

---

## 7.6 2026 MCP 生态与趋势

### Q: MCP 目前的生态发展如何？2026 年有哪些值得关注的趋势？

**MCP 从 2024 年 11 月发布至今，生态增长迅猛，已成为 AI 工具连接的事实标准。**

**当前生态全景：**

```
MCP 生态版图（2026）

  官方 / 一线支持:
  ┌─────────────────────────────────────────────┐
  │  GitHub    Slack    Google Drive   Postgres  │
  │  GitLab    Discord  Notion        MySQL     │
  │  Jira      Teams    Confluence    MongoDB   │
  │  Linear    Email    Google Maps   Redis     │
  └─────────────────────────────────────────────┘

  AI Client 支持:
  ┌─────────────────────────────────────────────┐
  │  Claude Desktop  ✅  (原生支持)              │
  │  Cursor          ✅  (主力 IDE)              │
  │  VS Code (Copilot) ✅  (2025 加入)          │
  │  Windsurf        ✅                          │
  │  OpenAI ChatGPT  ✅  (2025.03 宣布支持)      │
  │  Google Gemini   ✅  (2025 加入)             │
  └─────────────────────────────────────────────┘

  社区贡献:
  ┌─────────────────────────────────────────────┐
  │  1000+ 开源 MCP Server（npm / PyPI）         │
  │  Smithery.ai（MCP Server 注册中心）           │
  │  mcp.run（在线 MCP Server 市场）              │
  └─────────────────────────────────────────────┘
```

**2026 关键趋势：**

**趋势一：Remote MCP Server 成为主流**

```
2024-2025（本地时代）：
  MCP Server 以本地进程运行（stdio）
  每个用户要自己安装、配置
  类比：前端的 localhost 开发

2025-2026（远程时代）：
  MCP Server 部署在云端（Streamable HTTP）
  用户像装浏览器扩展一样"一键启用"
  类比：从本地部署到 Vercel/Netlify
```

远程 MCP Server 带来的变化：
- **免安装** — 不需要 `pip install`，直接连接远程 URL
- **集中管理** — 企业 IT 统一部署和更新
- **OAuth 认证** — 通过标准 OAuth 2.0 授权访问

**趋势二：企业级 MCP 采用**

| 采用模式 | 描述 | 典型场景 |
|---------|------|---------|
| **内部工具 Server** | 企业为内部系统构建 MCP Server | HR 系统、CRM、工单系统 |
| **MCP Gateway** | 统一管理多个 MCP Server 的网关 | 权限控制、流量管理、监控 |
| **MCP Marketplace** | 企业内部的 MCP Server 市场 | 团队间共享工具能力 |

**趋势三：MCP 与 Agent 框架的深度整合**

```
2025-2026 典型的 AI Agent 架构：

  ┌─────────────────────────────────────┐
  │           Agent Framework           │
  │  (LangGraph / CrewAI / AutoGen)     │
  │                                     │
  │  ┌─────────┐  ┌──────────────────┐ │
  │  │ Planner │  │ Tool Router      │ │
  │  │ (规划器) │  │ (工具路由器)      │ │
  │  └────┬────┘  └────────┬─────────┘ │
  │       │                │            │
  │       │    ┌───────────▼─────────┐ │
  │       │    │   MCP Client Pool   │ │
  │       │    │  (MCP 客户端连接池)   │ │
  │       │    └───┬────┬────┬───────┘ │
  └───────┼────────┼────┼────┼─────────┘
          │        │    │    │
     ┌────▼──┐ ┌──▼─┐ ┌▼──┐ ┌▼───────┐
     │GitHub │ │ DB │ │K8s│ │Slack   │
     │Server │ │Srvr│ │Svr│ │Server  │
     └───────┘ └────┘ └───┘ └────────┘
```

Agent 框架将 MCP 作为默认的工具接入方式，开发者不再需要手写工具适配代码。

**趋势四：协议演进**

| 版本节点 | 变化 |
|---------|------|
| 2024.11 初版 | stdio + SSE，基础 Tools/Resources |
| 2025.03 | Streamable HTTP 取代 SSE 成为推荐远程传输 |
| 2025.06 | OAuth 2.0 认证标准化，Tool Annotations（权限标注） |
| 2025-2026 | 工具组合（Composition）、多 Server 编排、Elicitation（Server 向用户提问） |

**对前端开发者的机会：**

1. **TypeScript MCP SDK** — 用熟悉的语言开发 MCP Server
2. **MCP Server 可视化管理** — 用 React/Vue 构建 MCP Dashboard
3. **Web-based MCP Client** — 在浏览器中直接连接 MCP Server
4. **企业内部 MCP 市场** — 类似 npm registry 的内部 MCP Server 注册中心

```typescript
// TypeScript MCP Server 示例（前端开发者友好）
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const server = new McpServer({ name: "npm-search", version: "1.0.0" });

server.tool(
  "search_packages",
  "搜索 npm 包",
  { query: z.string(), limit: z.number().default(5) },
  async ({ query, limit }) => {
    const res = await fetch(
      `https://registry.npmjs.org/-/v1/search?text=${query}&size=${limit}`
    );
    const data = await res.json();
    const packages = data.objects
      .map((o: any) => `${o.package.name} - ${o.package.description}`)
      .join("\n");
    return { content: [{ type: "text", text: packages }] };
  }
);

const transport = new StdioServerTransport();
await server.connect(transport);
```

> **面试话术：** "MCP 生态在 2025-2026 年经历了爆发式增长——OpenAI、Google 都已宣布支持 MCP，社区已有上千个开源 Server。关键趋势包括：Remote MCP Server 成为主流，从 stdio 本地进程转向 Streamable HTTP 云端部署；企业级采用加速，出现了 MCP Gateway 和内部 Server 市场；Agent 框架深度整合 MCP 作为默认工具层。对前端开发者来说，TypeScript SDK 已经成熟，这是一个非常好的切入点。"

---

## 本章总结

```
MCP 核心知识 Checklist：

✅ MCP = Anthropic 开源的 AI-工具标准化协议（"AI 世界的 USB"）
✅ 架构三层：Client（嵌入 Host）、Server（暴露能力）、Transport（stdio/SSE/HTTP）
✅ Server 四大能力：Tools、Resources、Prompts、Sampling
✅ vs Function Calling：MCP 是跨平台标准，FC 是厂商专属；MCP 自动发现+执行，FC 需手动编排
✅ 开发模式：Decorator 注册 → 自动生成 Schema → stdio/HTTP 启动
✅ 安全五层：权限模型 + OAuth 2.0 + 输入校验 + 速率限制 + 审计日志
✅ 2026 趋势：Remote Server 主流化、企业采用加速、Agent 框架深度整合
```

---

## 导航

| 上一章 | 下一章 |
|--------|--------|
| [06 - AI Agent 与 Agentic 架构](06-agent.md) | [08 - AI 应用工程化](08-ai-engineering.md) |
