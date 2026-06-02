# 06 - AI Agent

> **难度：** ⭐⭐⭐⭐⭐ | **定位：** AI 应用开发的终极形态，2026 面试高频考点
>
> **前端类比：** Agent 之于 LLM，就像 Next.js 之于 React——从一个"渲染库"升级为"全栈框架"。LLM 只负责"生成文本"，Agent 在此基础上加入了路由（Planning）、API 调用（Tool Use）、状态管理（Memory）和中间件（Reflection），形成一个能自主完成复杂任务的完整系统。

## 本章知识树

```
AI Agent
├── 6.1 Agent 核心概念（定义、四大组件：LLM/Tools/Memory/Planning）
├── 6.2 ReAct 模式（Reasoning + Acting 循环）
├── 6.3 Function Calling / Tool Use
├── 6.4 Agent 规划策略
│   ├── Plan-and-Solve
│   ├── REWOO（Reasoning Without Observation）
│   └── 动态重规划（Dynamic Replanning）
├── 6.5 Agent 反思机制
│   ├── Reflexion
│   ├── LATS（Language Agent Tree Search）
│   └── Generator-Evaluator 架构
├── 6.6 Agent 记忆架构
│   ├── 短期记忆（对话历史）
│   ├── 长期记忆（向量存储）
│   ├── 情景记忆（Episodic Memory）
│   └── 记忆框架（Mem0 / Zep / Letta）
├── 6.7 Multi-Agent 系统
│   ├── AutoGen v3
│   ├── CrewAI
│   ├── A2A（Agent-to-Agent）协议
│   └── 协作模式（层级/扁平/竞争）
├── 6.8 Copilot vs Agent 范式
├── 6.9 Chain vs Loop 架构
├── 6.10 LangGraph 工作流
├── 6.11 Agent 可观测性
├── 6.12 Voyager：终身学习 Agent
├── 6.13 Agent 防死循环与容错
├── 6.14 工具调用失败处理
├── 6.15 Human-in-the-Loop 系统设计
├── 6.16 Agent 评测 Benchmark
└── 6.17 Extended Thinking / Thinking Budget
```

---

## 6.1 Agent 核心概念

### Q: 什么是 AI Agent？它和普通 LLM 调用有什么区别？

**AI Agent = LLM + Tools + Memory + Planning。它不只是"问一句答一句"的对话机器人，而是一个能自主规划、调用工具、记忆上下文、完成复杂任务的智能系统。**

```
普通 LLM 调用（无状态、被动）：
  用户输入 → LLM 生成文本 → 返回结果
  特点：一次性、无工具、无记忆、无规划

AI Agent（有状态、主动）：
  用户目标 → Agent 拆解任务 → 循环执行 → 调用工具 → 反思纠错 → 返回结果
  特点：多步骤、有工具、有记忆、有规划
```

**Agent vs LLM 对比：**

| 维度 | 普通 LLM 调用 | AI Agent |
|------|-------------|----------|
| **交互模式** | 单轮 Input→Output | 多轮 Goal→Plan→Act→Observe 循环 |
| **工具使用** | 无 | 可调用 API、数据库、搜索引擎等 |
| **记忆** | 仅 Context Window 内 | 短期 + 长期记忆持久化 |
| **规划能力** | 无（靠 Prompt 引导） | 自主拆解子任务、动态调整计划 |
| **自主性** | 完全被动 | 可自主决策下一步 |
| **错误处理** | 无 | 可反思、重试、换策略 |
| **前端类比** | React 组件（纯渲染） | Next.js App（路由+API+状态+渲染） |

**Agent 的四大核心组件：**

```
┌─────────────────────────────────────────────┐
│                  AI Agent                    │
│                                             │
│  ┌──────────┐  ┌──────────┐                 │
│  │  🧠 LLM   │  │ 🔧 Tools │                 │
│  │ (大脑)    │  │ (手脚)   │                 │
│  │ 推理决策  │  │ 执行动作  │                 │
│  └──────────┘  └──────────┘                 │
│                                             │
│  ┌──────────┐  ┌──────────┐                 │
│  │ 💾 Memory │  │ 📋 Plan  │                 │
│  │ (记忆)    │  │ (规划)   │                 │
│  │ 经验积累  │  │ 任务拆解  │                 │
│  └──────────┘  └──────────┘                 │
└─────────────────────────────────────────────┘
```

| 组件 | 职责 | 前端类比 |
|------|------|---------|
| **LLM** | 核心推理引擎，理解指令、生成决策 | React 渲染引擎 |
| **Tools** | 外部能力接口（搜索、代码执行、API 调用） | API Routes / fetch 调用 |
| **Memory** | 存储对话历史、任务经验、用户偏好 | Redux / Zustand 状态管理 |
| **Planning** | 拆解任务、排序、动态调整 | React Router 路由规划 |

**面试话术：**

> "Agent 本质上是给 LLM 加了一个 **执行引擎**。LLM 只是 Agent 的'大脑'，Agent 还有'手'（Tools）、'记忆'（Memory）和'计划本'（Planning）。就像前端从 jQuery 时代的 `$.ajax` 升级到 Next.js 全栈框架——不只是渲染，还管路由、数据获取、缓存和中间件。"

---

## 6.2 ReAct 模式

### Q: 什么是 ReAct？请解释 Thought → Action → Observation 循环。

**ReAct = Reasoning + Acting，是 Agent 最经典的执行模式。核心思想是让 LLM 交替进行"推理思考"和"采取行动"，而不是一步到位。**

```
ReAct 循环：

  ┌─────────┐
  │  Goal    │  用户目标
  └────┬────┘
       │
       ▼
  ┌─────────┐     ┌─────────┐     ┌──────────────┐
  │ Thought │────▶│ Action  │────▶│ Observation  │
  │ (推理)  │     │ (行动)  │     │ (观察结果)   │
  └─────────┘     └─────────┘     └──────┬───────┘
       ▲                                 │
       │         循环直到完成              │
       └─────────────────────────────────┘

  最终 → Answer（最终回答）
```

**每一步的含义：**

- **Thought（思考）**：LLM 分析当前状态，推理下一步应该做什么
- **Action（行动）**：调用某个 Tool（搜索、计算、API 调用等）
- **Observation（观察）**：获取 Tool 返回的结果，作为下一轮 Thought 的输入

**ReAct 的代码实现：**

```python
from openai import OpenAI

client = OpenAI()

# 定义可用工具
tools = {
    "search": lambda query: f"搜索结果：{query} 的相关信息...",
    "calculator": lambda expr: str(eval(expr)),
    "get_weather": lambda city: f"{city}天气：晴，25°C",
}

def react_agent(goal: str, max_steps: int = 5):
    """ReAct Agent 核心循环"""
    messages = [
        {"role": "system", "content": """你是一个 ReAct Agent。
每一步严格按照以下格式输出：
Thought: <你的推理>
Action: <工具名>(<参数>)

如果已经有足够信息回答，输出：
Thought: 我已经有了足够信息
Answer: <最终回答>"""},
        {"role": "user", "content": goal}
    ]

    for step in range(max_steps):
        # 1. LLM 推理 → 生成 Thought + Action
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
        )
        output = response.choices[0].message.content
        messages.append({"role": "assistant", "content": output})
        print(f"--- Step {step + 1} ---\n{output}\n")

        # 2. 检查是否已有最终答案
        if "Answer:" in output:
            return output.split("Answer:")[-1].strip()

        # 3. 解析 Action，执行工具调用
        if "Action:" in output:
            action_line = output.split("Action:")[-1].strip()
            tool_name = action_line.split("(")[0].strip()
            tool_arg = action_line.split("(")[1].rstrip(")")

            # 4. 执行工具 → 获取 Observation
            observation = tools[tool_name](tool_arg)
            obs_msg = f"Observation: {observation}"
            messages.append({"role": "user", "content": obs_msg})
            print(f"{obs_msg}\n")

    return "达到最大步数，未能完成任务"

# 使用示例
result = react_agent("北京今天天气怎样？如果温度超过 20°C，计算 25 * 1.8 + 32 转为华氏度")
```

**ReAct 执行轨迹示例：**

```
--- Step 1 ---
Thought: 用户想知道北京天气，并根据温度做计算。先查天气。
Action: get_weather(北京)

Observation: 北京天气：晴，25°C

--- Step 2 ---
Thought: 北京 25°C，超过 20°C，需要计算华氏度 25 * 1.8 + 32
Action: calculator(25 * 1.8 + 32)

Observation: 77.0

--- Step 3 ---
Thought: 我已经有了足够信息
Answer: 北京今天晴，25°C（77°F）
```

**前端类比：** ReAct 就像前端的 `useEffect` + 数据流循环：组件渲染（Thought）→ 触发副作用请求数据（Action）→ 数据返回触发重新渲染（Observation）→ 再决定是否需要更多数据。每一轮 "渲染-请求-更新" 就是一次 ReAct 循环。

**面试话术：**

> "ReAct 的核心价值是让 LLM **边想边做**，而不是一次性输出所有结果。每一步都有观察反馈，这让 Agent 可以根据真实数据调整策略。就像前端开发中，我们不会一次性 fetch 所有 API 再渲染，而是 **逐步加载、逐步渲染**——这就是 ReAct 的思想。"

---

## 6.3 Function Calling / Tool Use

### Q: 什么是 Function Calling？请用 OpenAI API 实现一个 Tool Use 示例。

**Function Calling 是让 LLM "知道有哪些工具可用，并能自动决定何时调用、传什么参数"的能力。LLM 本身不执行代码，而是输出结构化的函数调用指令，由应用层去执行。**

```
Function Calling 流程：

  用户: "帮我查一下 AAPL 的股价"
       │
       ▼
  ┌──────────┐    LLM 不是自己查，
  │   LLM    │    而是输出调用指令：
  │          │──▶ {"name": "get_stock_price", "arguments": {"symbol": "AAPL"}}
  └──────────┘
       │
       ▼  应用层执行函数
  ┌──────────┐
  │ 实际 API │──▶ 返回 {"price": 195.23}
  └──────────┘
       │
       ▼  把结果喂回 LLM
  ┌──────────┐
  │   LLM    │──▶ "AAPL 当前股价为 $195.23"
  └──────────┘
```

**前端类比：** Function Calling ≈ API Routes。LLM 就像前端页面，它自己不能直接访问数据库，但它可以"发请求"给后端 API Route，后端执行后返回数据，前端再渲染。Function Calling 的 Tool Schema 就是 API 的 TypeScript 接口定义。

**OpenAI Function Calling 完整代码示例：**

```python
from openai import OpenAI
import json

client = OpenAI()

# 1. 定义工具 Schema（≈ TypeScript 接口定义）
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的当前天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如 '北京'、'上海'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "在知识库中搜索相关文档",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                    "top_k": {"type": "integer", "description": "返回数量", "default": 3}
                },
                "required": ["query"]
            }
        }
    }
]

# 2. 实际的工具实现（≈ API Route handler）
def get_weather(city: str, unit: str = "celsius") -> str:
    # 模拟天气 API
    return json.dumps({"city": city, "temp": 25, "unit": unit, "condition": "晴"})

def search_docs(query: str, top_k: int = 3) -> str:
    # 模拟知识库搜索
    return json.dumps({"results": [f"文档片段: 关于{query}的内容..."]})

# 工具名 → 函数的映射
tool_map = {
    "get_weather": get_weather,
    "search_docs": search_docs,
}

# 3. Agent 主循环：处理 Function Calling
def agent_with_tools(user_message: str):
    messages = [
        {"role": "system", "content": "你是一个有用的助手，可以查天气和搜索文档。"},
        {"role": "user", "content": user_message}
    ]

    while True:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto",  # LLM 自动决定是否调用工具
        )

        msg = response.choices[0].message
        messages.append(msg)

        # 如果 LLM 没有调用工具，说明已经有最终回答
        if not msg.tool_calls:
            return msg.content

        # 如果 LLM 调用了工具，逐个执行
        for tool_call in msg.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)

            print(f"调用工具: {func_name}({func_args})")

            # 执行工具函数
            result = tool_map[func_name](**func_args)

            # 把工具结果喂回 LLM
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

# 使用
answer = agent_with_tools("北京今天天气怎么样？")
print(answer)
```

**Function Calling vs 手动解析对比：**

| 维度 | 手动解析（正则/关键词匹配） | Function Calling（原生支持） |
|------|--------------------------|---------------------------|
| **参数提取** | 自己写正则，易出错 | LLM 自动提取成 JSON |
| **多工具选择** | if/else 硬编码 | LLM 自动选择最合适的工具 |
| **并行调用** | 自己实现 | OpenAI 原生支持 `parallel_tool_calls` |
| **类型安全** | 无 | JSON Schema 约束 |
| **前端类比** | 手动拼 URL + query string | tRPC / GraphQL 自动类型推导 |

**面试话术：**

> "Function Calling 的本质是 **让 LLM 输出结构化的 API 调用指令**，而不是自由文本。就像前端从手写 fetch 升级到 tRPC——参数有类型约束，路由有自动匹配，开发体验和可靠性都大幅提升。注意 LLM 只是"决定调用什么"，实际执行仍在应用层，所以安全性由我们自己把控。"

---

## 6.4 Agent 规划策略

### Q: 比较 Plan-and-Solve、REWOO 和动态重规划三种 Agent 规划策略。

**Planning 是 Agent 区别于简单 LLM 调用的关键能力。好的规划策略决定了 Agent 能否高效、准确地完成复杂任务。**

```
规划策略的本质问题：
  给定一个复杂目标，Agent 应该：
  1. 一次性制定完整计划再执行？（Plan-and-Solve）
  2. 先规划，但执行时不需要中间观察？（REWOO）
  3. 边执行边调整计划？（Dynamic Replanning）
```

**三种策略对比：**

| 维度 | Plan-and-Solve | REWOO | Dynamic Replanning |
|------|---------------|-------|-------------------|
| **核心思想** | 先制定完整计划，再逐步执行 | 预生成所有步骤+参数，无需中间观察 | 每步执行后评估，动态调整计划 |
| **LLM 调用次数** | 1（规划）+ N（执行） | 1（全规划）+ 1（总结） | N×2（每步规划+执行） |
| **Token 开销** | 中等 | 最低 | 最高 |
| **灵活性** | 低（计划固定） | 低（无中间反馈） | 最高（随时调整） |
| **适用场景** | 结构化、可预测的任务 | 简单线性任务、降本 | 不确定性高、需要探索的任务 |
| **前端类比** | 静态路由（预生成所有页面） | SSG（构建时生成） | 动态路由（按需渲染） |

**1. Plan-and-Solve 模式：**

```
Plan-and-Solve 流程：

  Goal: "写一篇关于 AI Agent 的技术博客"
       │
       ▼
  ┌─────────────────────────────────┐
  │ Step 1: Planning（一次性规划）    │
  │                                 │
  │  Plan:                          │
  │  1. 搜索 AI Agent 最新论文       │
  │  2. 整理核心概念和框架           │
  │  3. 编写大纲                    │
  │  4. 逐节撰写内容                │
  │  5. 润色和格式化                │
  └────────┬────────────────────────┘
           │
           ▼
  ┌─────────────────────────────────┐
  │ Step 2: Solving（逐步执行）      │
  │                                 │
  │  Execute Step 1 → Observation   │
  │  Execute Step 2 → Observation   │
  │  Execute Step 3 → Observation   │
  │  ...（按计划执行，不修改计划）    │
  └─────────────────────────────────┘
```

**2. REWOO（Reasoning Without Observation）：**

```python
# REWOO 的核心：一次性生成完整执行蓝图
# 关键区别：步骤之间用变量引用，不需要等待中间结果

rewoo_plan = """
Plan: 查找北京和上海的天气，然后对比。

#E1 = get_weather("北京")     # 不等结果
#E2 = get_weather("上海")     # 不等结果（可并行）
#E3 = compare(#E1, #E2)       # 引用前面的变量
#E4 = summarize(#E3)          # 最终总结
"""

# REWOO 优势：减少 LLM 调用，所有工具调用可以批量/并行执行
# REWOO 劣势：无法根据中间结果调整策略
```

**3. Dynamic Replanning（动态重规划）：**

```python
def dynamic_replan_agent(goal: str):
    plan = initial_plan(goal)  # 生成初始计划

    for step in plan:
        result = execute(step)

        # 关键：每步执行后评估是否需要调整计划
        evaluation = llm_evaluate(
            original_plan=plan,
            current_step=step,
            result=result,
            remaining_steps=plan.remaining()
        )

        if evaluation.needs_replan:
            # 动态调整剩余计划
            plan = replan(
                goal=goal,
                completed=plan.completed_steps(),
                observations=plan.all_observations(),
                reason=evaluation.reason
            )
            print(f"重新规划！原因：{evaluation.reason}")

    return synthesize(plan.all_results())
```

**面试话术：**

> "选择规划策略要看任务特性。可预测的线性任务用 Plan-and-Solve 够了；追求低延迟低成本用 REWOO；面对不确定性高、可能失败需要探索的任务，必须用 Dynamic Replanning。就像前端路由选择——博客站用 SSG，电商用 SSR，实时应用用 CSR。核心原则是 **用最低成本获得足够的灵活性**。"

---

## 6.5 Agent 反思机制

### Q: 什么是 Reflexion？什么是 LATS？Agent 为什么需要反思机制？

**反思机制让 Agent 从失败中学习，而不是重复犯同样的错误。核心思想：Agent 执行后自我评估结果，生成反思总结，存入记忆供后续使用。**

```
没有反思的 Agent：
  尝试 → 失败 → 再尝试（同样的方式）→ 再失败 → ...

有反思的 Agent：
  尝试 → 失败 → 反思"为什么失败？" → 总结经验 → 用新策略再尝试 → 成功
```

**1. Reflexion 框架：**

Reflexion 由三个核心组件构成：Actor（执行者）、Evaluator（评估者）、Self-Reflection（自我反思）。

```
Reflexion 循环：

  ┌─────────┐
  │  Actor  │──── 执行任务 ────▶ 结果
  └─────────┘                    │
       ▲                         ▼
       │                   ┌──────────┐
       │                   │Evaluator │ 评估是否成功
       │                   └────┬─────┘
       │                        │
       │              ┌─────────▼──────────┐
       │              │  Self-Reflection   │
       │              │  "我为什么失败了？  │
       │              │   下次应该怎么做？" │
       │              └─────────┬──────────┘
       │                        │
       │                 ┌──────▼──────┐
       └─────────────────│   Memory    │  存储反思结果
                         │ (经验库)    │  供下次使用
                         └─────────────┘
```

```python
class ReflexionAgent:
    def __init__(self):
        self.memory = []  # 跨任务的经验记忆

    def run(self, task: str, max_trials: int = 3) -> str:
        for trial in range(max_trials):
            # 1. Actor 执行任务（带上历史反思）
            trajectory = self.act(task, self.memory)

            # 2. Evaluator 评估结果
            score, feedback = self.evaluate(task, trajectory)

            if score >= 0.8:  # 成功阈值
                return trajectory.final_answer

            # 3. Self-Reflection：生成反思
            reflection = self.reflect(task, trajectory, feedback)
            # 例如："我在第 3 步使用了错误的搜索关键词，
            #        导致检索到不相关的文档。下次应该
            #        先明确关键实体再搜索。"

            self.memory.append(reflection)
            print(f"Trial {trial + 1} 失败，反思：{reflection}")

        return "未能成功完成任务"

    def reflect(self, task, trajectory, feedback) -> str:
        prompt = f"""
任务：{task}
执行轨迹：{trajectory}
评估反馈：{feedback}
之前的反思：{self.memory}

请分析失败原因，给出具体可执行的改进建议：
"""
        return llm_call(prompt)
```

**Reflexion 的核心创新点：**

- **跨 Trial 记忆**：反思结果在多次尝试之间保留，避免重复犯错
- **跨任务迁移**：不仅是同一任务内重试，经验可以迁移到新任务
- **自然语言反思**：不需要梯度更新，用文本总结经验（≈ 人类写复盘日志）

**2. LATS（Language Agent Tree Search）：**

**LATS 将蒙特卡洛树搜索（MCTS）引入 Agent 决策，把每一步的可能行动建模为搜索树的分支，通过评估和回溯找到最优路径。**

```
LATS 搜索树示例：

  Goal: "解决一个复杂数学题"

              Root（初始状态）
             /        |        \
        Action1    Action2    Action3
        (尝试法1)  (尝试法2)  (尝试法3)
        Score:0.3  Score:0.7  Score:0.5
                   /    \
              Act2.1   Act2.2    ← 扩展最优节点
              Score:0.8 Score:0.4
              /
          Act2.1.1     ← 继续扩展
          Score:0.95   ✅ 找到最优路径！
```

```python
class LATSAgent:
    """Language Agent Tree Search"""

    def search(self, goal: str, n_iterations: int = 10):
        root = Node(state=goal, parent=None)

        for _ in range(n_iterations):
            # 1. Selection：选择最有希望的节点（UCB 策略）
            node = self.select(root)

            # 2. Expansion：生成多个可能的下一步行动
            children = self.expand(node, n_children=3)

            # 3. Simulation：模拟执行，评估每个子节点
            for child in children:
                reward = self.simulate(child)
                child.value = reward

            # 4. Backpropagation：反向更新所有祖先节点的值
            best_child = max(children, key=lambda c: c.value)
            self.backpropagate(best_child)

        # 返回最优路径
        return self.best_path(root)

    def expand(self, node, n_children=3):
        """让 LLM 生成 N 个不同的候选行动"""
        prompt = f"当前状态：{node.state}\n请给出 {n_children} 种不同的下一步方案："
        candidates = llm_call(prompt, n=n_children)
        return [Node(state=c, parent=node) for c in candidates]
```

**3. Generator-Evaluator 架构：**

```
Generator-Evaluator 是最简单的反思模式：

  Generator（生成器 LLM）──▶ 候选答案 ──▶ Evaluator（评估器 LLM）
       ▲                                        │
       │           评分 + 反馈                    │
       └────────────────────────────────────────┘

  核心：两个 LLM 角色分离——一个负责生成，一个负责评判。
  类似：前端的 Code Review 流程——开发者写代码，Reviewer 审核。
```

**三种反思机制对比：**

| 维度 | Reflexion | LATS | Generator-Evaluator |
|------|-----------|------|-------------------|
| **搜索方式** | 线性重试 | 树搜索（广度优先） | 生成-评估循环 |
| **记忆** | 跨 Trial 自然语言记忆 | 树节点值函数 | 无长期记忆 |
| **计算成本** | 中等（3-5 次重试） | 高（树的分支因子×深度） | 低（1-2 次循环） |
| **适用场景** | 编程、推理任务 | 复杂决策、博弈 | 文本生成、翻译 |

**面试话术：**

> "Agent 反思机制解决的核心问题是 **如何从失败中学习**。Reflexion 像人类写复盘日志，LATS 像下棋时的搜索算法，Generator-Evaluator 像 Code Review。选择哪种取决于任务的不确定性程度和计算预算。"

---

## 6.6 Agent 记忆架构

### Q: 详细对比 Agent 的各种记忆类型和存储架构。

**Agent Memory 解决的核心问题：LLM 的 Context Window 是有限的，Agent 需要一个外部记忆系统来存储和检索历史信息。**

**前端类比：Agent Memory ≈ 前端状态管理生态。短期记忆 = React useState（组件级状态），长期记忆 = Redux/Zustand（全局持久化状态），情景记忆 = React Query 的缓存（带上下文的缓存），图记忆 = GraphQL 的关联数据查询。**

**记忆类型体系：**

```
Agent Memory 架构：

  ┌─────────────────────────────────────────────────┐
  │                 Agent Memory                     │
  │                                                  │
  │  ┌──────────────┐  ┌───────────────┐            │
  │  │ 短期记忆      │  │  长期记忆      │            │
  │  │ Short-Term   │  │  Long-Term    │            │
  │  │              │  │               │            │
  │  │ · 对话历史    │  │ · 向量存储     │            │
  │  │ · 工作缓存    │  │ · 知识图谱     │            │
  │  │ · 当前任务    │  │ · 用户画像     │            │
  │  │   上下文     │  │ · 技能库       │            │
  │  └──────────────┘  └───────────────┘            │
  │                                                  │
  │  ┌──────────────┐  ┌───────────────┐            │
  │  │ 情景记忆      │  │  语义记忆      │            │
  │  │ Episodic     │  │  Semantic     │            │
  │  │              │  │               │            │
  │  │ · 完整经历    │  │ · 抽象知识     │            │
  │  │ · 时间戳     │  │ · 事实规则     │            │
  │  │ · 情境上下文  │  │ · 概念关系     │            │
  │  └──────────────┘  └───────────────┘            │
  └─────────────────────────────────────────────────┘
```

**四种记忆存储架构对比：**

| 架构 | 存储方式 | 检索方式 | 优势 | 劣势 | 前端类比 |
|------|---------|---------|------|------|---------|
| **Flat Vector** | 向量数据库平铺 | 语义相似度搜索 | 简单、通用 | 无关系、无时序 | localStorage |
| **Episodic** | 带时间戳的事件序列 | 时间+语义检索 | 有上下文、可回溯 | 存储量大 | React Query cache |
| **Graph** | 知识图谱（实体+关系） | 图遍历+语义搜索 | 关系丰富、可推理 | 构建复杂 | GraphQL schema |
| **Hybrid** | 向量+图+关系型 | 多路检索+融合 | 最全面 | 架构复杂 | 全状态管理方案 |

**Flat Vector vs Episodic vs Graph 示例：**

```python
# ===== 1. Flat Vector 记忆 =====
# 最简单：所有记忆扁平存入向量数据库
flat_memory = VectorStore()
flat_memory.add("用户喜欢 Python")
flat_memory.add("用户是前端开发者")
flat_memory.add("上次讨论了 RAG 架构")
# 检索：query = "用户的技术栈" → 返回最相似的 K 条

# ===== 2. Episodic 记忆 =====
# 保留完整的交互"片段"，包含上下文
episodic_memory = [
    {
        "episode_id": "ep_001",
        "timestamp": "2026-05-30T14:00:00",
        "context": "用户在调试 RAG Pipeline",
        "events": [
            {"role": "user", "content": "我的检索结果不准"},
            {"role": "agent", "action": "search_docs", "result": "..."},
            {"role": "agent", "content": "建议使用 HyDE 改善检索"},
            {"role": "user", "content": "有效！谢谢"},
        ],
        "outcome": "success",
        "lesson": "HyDE 对模糊查询有效",
    }
]
# 检索：先按语义匹配，再按时间排序，返回完整的情景

# ===== 3. Graph 记忆 =====
# 实体 + 关系，可以推理
graph_memory = {
    "entities": {
        "User": {"role": "前端开发者", "lang": "Python, TypeScript"},
        "RAG": {"type": "架构", "components": ["Embedding", "VectorDB", "LLM"]},
    },
    "relations": [
        ("User", "正在学习", "RAG"),
        ("User", "擅长", "TypeScript"),
        ("RAG", "包含", "Embedding"),
    ]
}
# 检索：query = "用户和 RAG 的关系" → 图遍历找到所有相关路径
```

**主流记忆框架对比：**

| 框架 | 记忆类型 | 核心特色 | 适用场景 |
|------|---------|---------|---------|
| **Mem0** | Flat Vector + 自动摘要 | 自动提取用户偏好，API 简洁 | 个性化对话助手 |
| **Zep** | Episodic + 知识图谱 | 自动构建用户知识图谱 | 复杂对话系统 |
| **Letta (MemGPT)** | 分层记忆（Core/Recall/Archive） | 自主管理记忆的 Agent OS | 长期运行的自主 Agent |

```python
# Mem0 使用示例
from mem0 import Memory

memory = Memory()

# 添加记忆（自动提取关键信息）
memory.add(
    "我是前端开发者，正在转型 AI 方向，熟悉 React 和 TypeScript",
    user_id="user_123"
)

# 检索相关记忆
results = memory.search("用户的技术背景", user_id="user_123")
# → [{"memory": "用户是前端开发者，擅长 React/TypeScript，正在转型 AI"}]

# Mem0 会自动去重、合并、更新记忆
memory.add("我最近在学 Python 和 LangChain", user_id="user_123")
# → 自动合并为：用户是前端转 AI，技术栈：React/TS/Python/LangChain
```

**面试话术：**

> "Agent 记忆选型和前端状态管理一样——简单场景用 Flat Vector（≈ localStorage），需要上下文关联用 Episodic（≈ React Query），需要复杂关系推理用 Graph（≈ GraphQL），生产系统通常是 Hybrid。框架选择上，轻量个性化用 Mem0，复杂对话用 Zep，长期自主 Agent 用 Letta。"

---

## 6.7 Multi-Agent 系统

### Q: 什么是 Multi-Agent 系统？比较 AutoGen 和 CrewAI，什么是 A2A 协议？

**Multi-Agent 系统 = 多个 Agent 协作完成一个复杂任务。每个 Agent 有自己的角色（Role）、工具（Tools）和目标（Goal），通过消息传递进行协作。**

**前端类比：** Multi-Agent ≈ 微服务架构。单 Agent = 单体应用，Multi-Agent = 微服务——每个服务（Agent）有明确职责，通过 API（消息传递）协作。就像前端微服务化后，有 User Service、Order Service、Payment Service 各司其职。

```
Multi-Agent 协作模式：

1. 层级模式（Hierarchical）：
   ┌──────────┐
   │ Manager  │  管理者 Agent
   │  Agent   │  分配任务、汇总结果
   └────┬─────┘
        │
   ┌────┴────────────────┐
   │         │           │
   ▼         ▼           ▼
┌──────┐ ┌──────┐ ┌──────┐
│Worker│ │Worker│ │Worker│
│Agent1│ │Agent2│ │Agent3│
└──────┘ └──────┘ └──────┘

2. 扁平模式（Flat / Peer-to-peer）：
   Agent1 ◄──► Agent2
     ▲            ▲
     │            │
     ▼            ▼
   Agent3 ◄──► Agent4

3. 竞争模式（Competitive / Debate）：
   Agent1 ──提出方案──▶ ┌──────────┐
   Agent2 ──提出方案──▶ │  Judge   │──▶ 最优方案
   Agent3 ──提出方案──▶ │  Agent   │
                        └──────────┘
```

**AutoGen v3 vs CrewAI 对比：**

| 维度 | AutoGen v3 (Microsoft) | CrewAI |
|------|----------------------|--------|
| **架构理念** | Event-driven, Actor 模型 | Role-based, 顺序/并行流程 |
| **核心抽象** | Agent + Team + Termination | Agent + Task + Crew + Process |
| **通信方式** | 异步消息传递（Event Bus） | 任务链传递（Task Output → 下一个 Agent） |
| **Agent 定义** | `AssistantAgent`, `UserProxyAgent` | `Agent(role, goal, backstory)` |
| **工作流控制** | `SelectorGroupChat`, `RoundRobinGroupChat` | `Process.sequential` / `Process.hierarchical` |
| **状态管理** | 内置 Agent Runtime，持久化支持 | 任务上下文传递 |
| **可扩展性** | 高（分布式 Runtime） | 中（单进程） |
| **学习曲线** | 陡峭（概念多） | 平缓（API 直观） |
| **适用场景** | 复杂企业级多 Agent 系统 | 快速原型、简单多 Agent 流程 |
| **前端类比** | Kubernetes + 微服务 | Docker Compose + 几个服务 |

**CrewAI 代码示例：**

```python
from crewai import Agent, Task, Crew, Process

# 定义 Agent（每个 Agent 有明确角色）
researcher = Agent(
    role="AI 研究员",
    goal="搜索和整理 AI Agent 最新进展",
    backstory="你是一位资深 AI 研究员，擅长快速理解和总结论文",
    tools=[search_tool, arxiv_tool],
    llm="gpt-4o",
)

writer = Agent(
    role="技术写手",
    goal="将研究结果写成通俗易懂的技术博客",
    backstory="你是一位前端转 AI 的技术作者，擅长用前端类比解释 AI 概念",
    tools=[],
    llm="gpt-4o",
)

reviewer = Agent(
    role="技术审稿人",
    goal="审核文章的技术准确性和可读性",
    backstory="你是一位严格的技术审稿人",
    tools=[],
    llm="gpt-4o",
)

# 定义 Task（任务之间有依赖关系）
research_task = Task(
    description="搜索 2026 年 AI Agent 最新论文和框架，整理 Top 5 进展",
    agent=researcher,
    expected_output="5 个关键进展的结构化总结",
)

writing_task = Task(
    description="基于研究结果，撰写一篇 2000 字的技术博客",
    agent=writer,
    expected_output="完整的 Markdown 格式博客文章",
    context=[research_task],  # 依赖研究任务的输出
)

review_task = Task(
    description="审核文章的技术准确性，提出修改建议",
    agent=reviewer,
    expected_output="审核意见和修改建议",
    context=[writing_task],
)

# 组装 Crew 并执行
crew = Crew(
    agents=[researcher, writer, reviewer],
    tasks=[research_task, writing_task, review_task],
    process=Process.sequential,  # 顺序执行
    verbose=True,
)

result = crew.kickoff()
```

**A2A（Agent-to-Agent）协议：**

A2A 是 Google 于 2025 年提出的开放协议，定义了不同 Agent 之间如何发现、通信和协作的标准化接口。

```
A2A 协议核心概念：

  ┌──────────────────────────────────────────┐
  │              A2A Protocol                 │
  │                                          │
  │  1. Agent Card（身份卡）                  │
  │     - 描述 Agent 能力、技能、接口         │
  │     - 类似 OpenAPI spec / package.json    │
  │                                          │
  │  2. Task（任务对象）                      │
  │     - Agent 间交换的工作单元              │
  │     - 状态机：submitted → working         │
  │              → completed / failed         │
  │                                          │
  │  3. Message & Part（消息体）              │
  │     - 文本/文件/结构化数据                │
  │     - 类似 HTTP Request/Response          │
  │                                          │
  │  4. Artifact（产出物）                    │
  │     - Agent 的输出结果                    │
  │     - 类似 API Response Body              │
  └──────────────────────────────────────────┘
```

```json
// Agent Card 示例（≈ package.json for Agent）
{
  "name": "travel-planner-agent",
  "description": "规划旅行行程的 Agent",
  "url": "https://agent.example.com",
  "version": "1.0.0",
  "capabilities": {
    "streaming": true,
    "pushNotifications": false
  },
  "skills": [
    {
      "id": "plan_trip",
      "name": "规划旅行",
      "description": "根据用户需求规划完整旅行行程",
      "inputModes": ["text"],
      "outputModes": ["text", "json"]
    }
  ]
}
```

**前端类比：** A2A ≈ REST API + OpenAPI Spec。Agent Card ≈ Swagger 文档（描述服务能力），Task ≈ HTTP Request（工作单元），Artifact ≈ Response Body。A2A 就是给 Agent 世界定义了"HTTP 协议"。

**面试话术：**

> "Multi-Agent 的本质是 **分工协作**。AutoGen 适合复杂企业级场景，它的 Actor 模型天然支持分布式；CrewAI 更适合快速原型，API 友好。A2A 协议是在更上层解决'不同厂商的 Agent 如何互联互通'的问题——就像 HTTP 让不同的 Web Server 能互相通信一样，A2A 让不同平台的 Agent 能发现彼此、交换任务。"

---

## 6.8 Copilot vs Agent 范式

### Q: 对比 Copilot（人在环路中）和 Agent（自主执行）两种范式。

**Copilot 和 Agent 是 AI 应用的两种设计范式，核心区别在于 "人" 在决策链中的位置。**

```
Copilot 模式（Human-in-the-Loop）：

  人类 ──请求──▶ AI ──建议──▶ 人类 ──确认/修改──▶ 执行

  特点：AI 只是"副驾驶"，人类做最终决策
  例子：GitHub Copilot、ChatGPT 对话、Cursor 编辑器

Agent 模式（Autonomous）：

  人类 ──设定目标──▶ Agent ──自主规划+执行──▶ 结果

  特点：AI 是"自动驾驶"，人类只设定目标
  例子：Devin（自主编程）、AutoGPT、Claude Code Agent
```

**详细对比表：**

| 维度 | Copilot（副驾驶） | Agent（自动驾驶） |
|------|------------------|------------------|
| **人的角色** | 决策者，AI 辅助 | 目标设定者，AI 自主 |
| **控制权** | 人类保留最终控制 | Agent 自主决策 |
| **交互频率** | 每步都需要人确认 | 设定目标后无需干预 |
| **安全性** | 高（人工审核） | 需要额外安全护栏 |
| **效率** | 受限于人的响应速度 | 可 24/7 自主执行 |
| **适用任务** | 创意性、高风险决策 | 重复性、明确目标的任务 |
| **错误处理** | 人工纠正 | 需要反思机制+回滚能力 |
| **信任要求** | 低（人工兜底） | 高（需要对 Agent 充分信任） |
| **前端类比** | VS Code + 智能补全 | CI/CD 自动化流水线 |
| **代表产品** | GitHub Copilot, Cursor | Devin, Claude Code, SWE-Agent |

**现实中的混合范式：**

```
纯 Copilot ◄──────────────────────────────────►  纯 Agent

  GitHub          Cursor           Claude Code       Devin
  Copilot      Tab 补全 +         Agent Mode       全自主
 (逐行建议)   Multi-file Edit   (自主 + 人确认)    编程

  ◄─── 人的参与度高 ─────────────────── 人的参与度低 ───►
```

**生产系统中的最佳实践：Guard Rails（护栏）模式**

```python
class GuardedAgent:
    """带护栏的 Agent：关键操作需要人工确认"""

    def execute(self, task: str):
        plan = self.plan(task)

        for step in plan:
            if step.risk_level == "high":
                # 高风险操作：回退到 Copilot 模式
                approval = self.ask_human(
                    f"即将执行高风险操作：{step.description}\n确认？(y/n)"
                )
                if not approval:
                    step = self.replan(step)

            # 低风险操作：自主执行
            result = self.act(step)
            self.log(step, result)  # 全程记录，可审计
```

**面试话术：**

> "Copilot 和 Agent 不是非此即彼，而是一个 **连续频谱**。最好的设计是 **默认 Agent，关键时刻 Copilot**——低风险操作自动完成，高风险操作暂停等待人工确认。就像 CI/CD Pipeline：代码格式化、单元测试自动跑，但部署到生产环境需要人工 approve。核心原则是 **在效率和安全之间找平衡**。"

---

## 6.9 Chain vs Loop 架构

### Q: 对比 Chain（线性链式）和 Loop（循环迭代）两种 Agent 架构模式。

**Chain 和 Loop 是两种最基本的 Agent 执行架构，决定了任务流的执行方式。**

```
Chain 架构（线性/DAG）：

  Step1 ──▶ Step2 ──▶ Step3 ──▶ Step4 ──▶ Output

  每个步骤固定，不回头。
  像流水线：输入 → 加工 → 输出

Loop 架构（循环迭代）：

  ┌──▶ Reason ──▶ Act ──▶ Observe ──┐
  │                                  │
  │         Done?  ─── No ───────────┘
  │                  │
  │                 Yes
  │                  │
  └──────────────── Output

  反复循环直到满足终止条件。
  像 while 循环：直到条件满足才停止。
```

**详细对比：**

| 维度 | Chain（线性） | Loop（循环） |
|------|-------------|-------------|
| **执行流程** | A → B → C → D（固定顺序） | 循环执行直到条件满足 |
| **步骤数** | 预定义、固定 | 动态、不确定 |
| **灵活性** | 低（路径固定） | 高（可根据结果调整） |
| **可预测性** | 高（结果确定性强） | 低（可能死循环） |
| **调试难度** | 简单（线性追踪） | 复杂（循环状态追踪） |
| **延迟** | 可预测 | 不可预测 |
| **适用场景** | 管道式处理（ETL、数据处理） | 探索式任务（搜索、推理） |
| **前端类比** | Express 中间件链 | React useEffect 副作用循环 |
| **典型实现** | LangChain LCEL | ReAct / LangGraph |

**Chain 架构示例（LangChain LCEL）：**

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Chain = 一条直线，步骤固定
# 前端类比：Express app.use(middleware1).use(middleware2).use(handler)

chain = (
    ChatPromptTemplate.from_template("将以下文本翻译为英文：{text}")
    | ChatOpenAI(model="gpt-4o")         # Step 1: 翻译
    | StrOutputParser()
    | ChatPromptTemplate.from_template("润色以下英文，使其更自然：{text}")
    | ChatOpenAI(model="gpt-4o")         # Step 2: 润色
    | StrOutputParser()
    | ChatPromptTemplate.from_template("为以下文本生成摘要（50字内）：{text}")
    | ChatOpenAI(model="gpt-4o")         # Step 3: 摘要
    | StrOutputParser()
)

result = chain.invoke({"text": "人工智能正在改变软件开发的方式..."})
```

**Loop 架构示例：**

```python
def loop_agent(goal: str, max_iterations: int = 10):
    """Loop 架构：循环执行直到完成"""
    state = {"goal": goal, "history": [], "done": False}

    for i in range(max_iterations):
        # 1. Reason: 分析当前状态，决定下一步
        thought = reason(state)

        # 2. Act: 执行动作
        action, result = act(thought)

        # 3. Observe: 更新状态
        state["history"].append({
            "thought": thought,
            "action": action,
            "result": result,
        })

        # 4. 检查终止条件
        if should_stop(state):
            state["done"] = True
            break

    return synthesize(state)
```

**何时选择哪种架构？**

```
选择 Chain：
  ✅ 步骤明确、流程固定
  ✅ 不需要根据中间结果调整
  ✅ 需要可预测的延迟和成本
  ✅ 示例：RAG Pipeline、数据 ETL、翻译润色

选择 Loop：
  ✅ 任务结果不确定，需要探索
  ✅ 需要根据反馈动态调整
  ✅ 需要错误重试和自我纠正
  ✅ 示例：代码调试、复杂搜索、对话系统

实际生产 = Chain + Loop 混合：
  Chain 处理确定性的部分，
  Loop 处理不确定性的部分。
  就像 Next.js：静态页面用 SSG（Chain），
  动态页面用 SSR（Loop）。
```

**面试话术：**

> "Chain 和 Loop 的选择核心在于 **任务的确定性**。确定性高用 Chain，不确定性高用 Loop。生产系统通常是混合架构——用 Chain 做主流程骨架，在需要探索或纠错的节点嵌入 Loop。就像前端应用：路由是 Chain（确定的页面流转），但每个页面内部可能有 Loop（轮询数据、重试请求）。"

---

## 6.10 LangGraph 工作流

### Q: 什么是 LangGraph？请用状态机的概念解释，并给出代码示例。

**LangGraph 是 LangChain 团队推出的 Agent 工作流框架，核心思想是用 **有向图（Graph）+ 状态机（State Machine）** 来定义 Agent 的执行流程。**

**前端类比：** LangGraph ≈ XState（前端状态机库）。每个节点（Node）是一个状态/处理函数，边（Edge）定义状态转移条件，State 在节点之间流转和更新。

```
LangGraph 核心概念：

  ┌────────────────────────────────────────────┐
  │              LangGraph                      │
  │                                            │
  │  State（状态）                              │
  │    └─ 全局共享的数据结构，节点可读写         │
  │                                            │
  │  Node（节点）                               │
  │    └─ 处理函数，接收 State，返回更新后的 State│
  │                                            │
  │  Edge（边）                                 │
  │    └─ 普通边：固定跳转                      │
  │    └─ 条件边：根据 State 动态选择下一节点    │
  │                                            │
  │  Graph（图）                                │
  │    └─ 有向图，定义完整工作流                 │
  └────────────────────────────────────────────┘
```

**LangGraph 代码示例：一个简单的 ReAct Agent**

```python
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage
import operator

# 1. 定义 State（≈ React 的 state/store）
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]  # 消息列表，append 模式
    iteration: int  # 循环计数

# 2. 定义工具
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """搜索互联网获取信息"""
    return f"搜索结果：关于 '{query}' 的最新信息..."

@tool
def calculator(expression: str) -> str:
    """计算数学表达式"""
    return str(eval(expression))

tools = [search, calculator]
tool_node = ToolNode(tools)

# 3. 定义 LLM
llm = ChatOpenAI(model="gpt-4o").bind_tools(tools)

# 4. 定义节点函数（≈ React 组件 / Express handler）
def agent_node(state: AgentState) -> dict:
    """Agent 思考节点：调用 LLM 决定下一步"""
    response = llm.invoke(state["messages"])
    return {
        "messages": [response],
        "iteration": state.get("iteration", 0) + 1,
    }

# 5. 定义条件边（≈ React Router 的条件重定向）
def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """决定是继续调用工具，还是结束"""
    last_message = state["messages"][-1]

    # 如果 LLM 决定调用工具 → 转到 tools 节点
    if last_message.tool_calls:
        return "tools"

    # 否则 → 结束
    return "end"

# 6. 构建 Graph（≈ 定义路由和页面流转）
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

# 设置入口
workflow.set_entry_point("agent")

# 添加条件边
workflow.add_conditional_edges(
    "agent",                      # 从 agent 节点出发
    should_continue,              # 用这个函数决定去哪
    {
        "tools": "tools",         # 如果返回 "tools" → 去 tools 节点
        "end": END,               # 如果返回 "end" → 结束
    }
)

# 工具执行完 → 回到 agent 思考
workflow.add_edge("tools", "agent")

# 7. 编译 Graph
app = workflow.compile()

# 8. 执行
result = app.invoke({
    "messages": [HumanMessage(content="北京天气怎样？温度乘以2等于多少？")],
    "iteration": 0,
})
```

**执行流程可视化：**

```
                    ┌─────────┐
                    │  START  │
                    └────┬────┘
                         │
                         ▼
                 ┌───────────────┐
          ┌──────│    agent      │──────┐
          │      │  (LLM 思考)   │      │
          │      └───────────────┘      │
          │                             │
     has tool_calls              no tool_calls
          │                             │
          ▼                             ▼
  ┌───────────────┐             ┌───────────┐
  │    tools      │             │    END    │
  │ (执行工具)    │             └───────────┘
  └───────┬───────┘
          │
          └──────── 回到 agent ─────────┘
```

**LangGraph 的进阶特性：**

| 特性 | 描述 | 前端类比 |
|------|------|---------|
| **Checkpointing** | 保存图的执行状态，可恢复 | Redux Persist / 页面状态恢复 |
| **Human-in-the-Loop** | 在特定节点暂停等待人工输入 | 表单提交确认 |
| **Subgraph** | 图中嵌套子图 | 嵌套路由 / Layout |
| **Streaming** | 流式输出中间状态 | SSE / WebSocket 实时更新 |
| **Time Travel** | 回溯到之前的状态重新执行 | Redux DevTools 时间旅行 |

**面试话术：**

> "LangGraph 的价值在于它用 **状态机 + 有向图** 把 Agent 的执行流程 **可视化、可控、可调试**。和 LangChain 的线性 Chain 不同，LangGraph 天然支持循环、分支、并行、人工干预。如果说 LangChain LCEL 是 Express 中间件，那 LangGraph 就是 XState 状态机——可以表达任意复杂的工作流。"

---

## 6.11 Agent 可观测性

### Q: 如何监控和调试 Agent？有哪些关键指标和工具？

**Agent 可观测性（Observability）是生产 Agent 的核心挑战。不同于传统 API 的请求-响应模型，Agent 有多步执行、工具调用、记忆读写、动态规划等复杂行为，需要专门的监控体系。**

**前端类比：** Agent 可观测性 ≈ 前端性能监控（Lighthouse + Sentry + DataDog）。LLM 调用延迟 = API 响应时间，Token 消耗 = 流量成本，工具调用成功率 = API 可用性，Agent 任务完成率 = 核心 Web Vitals。

**Agent 可观测性三大支柱：**

```
                Agent 可观测性
               /       |       \
         Traces      Metrics     Logs
        (追踪)       (指标)     (日志)

  · 完整执行轨迹   · 延迟/成本   · 每步详细记录
  · 步骤依赖关系   · 成功率      · 错误堆栈
  · 工具调用链     · Token 用量  · 上下文快照
```

**关键监控指标：**

| 指标类别 | 具体指标 | 描述 | 前端类比 |
|---------|---------|------|---------|
| **性能** | Total Latency | Agent 完成任务总耗时 | Page Load Time |
| | LLM Latency (P50/P95) | 单次 LLM 调用延迟 | TTFB |
| | Steps per Task | 完成任务的平均步骤数 | DOM 节点数量 |
| **成本** | Token Usage | Input/Output Token 消耗 | 带宽消耗 |
| | Cost per Task | 每个任务的 API 费用 | CDN 费用 |
| | Tool Call Count | 工具调用次数 | API 请求数 |
| **质量** | Task Completion Rate | 任务成功完成比例 | Core Web Vitals |
| | Hallucination Rate | 幻觉/错误回答比例 | Error Rate |
| | User Satisfaction | 用户满意度评分 | NPS |
| **可靠性** | Tool Call Success Rate | 工具调用成功率 | API 可用性 |
| | Retry Count | 重试次数 | Retry 策略指标 |
| | Loop Detection | 是否陷入无限循环 | 内存泄漏检测 |

**主流 Agent 可观测性工具：**

| 工具 | 核心功能 | 特点 |
|------|---------|------|
| **LangSmith** | Trace + Evaluation + Dataset | LangChain 官方，最完整 |
| **Langfuse** | Trace + Cost + Score | 开源，自托管友好 |
| **Arize Phoenix** | Trace + Embedding 分析 | LLM 专用可观测性 |
| **OpenTelemetry + Jaeger** | 通用分布式追踪 | 标准化，可接入现有基础设施 |
| **Braintrust** | Eval + Logging + Prompt 管理 | 偏评估侧 |

**Langfuse 集成示例：**

```python
from langfuse.decorators import observe, langfuse_context
from langfuse.openai import openai  # 自动追踪 OpenAI 调用

@observe()  # 自动追踪整个函数
def agent_pipeline(user_input: str):

    # 1. 规划阶段（自动追踪 LLM 调用）
    plan = plan_task(user_input)

    # 2. 执行阶段
    for step in plan:
        result = execute_step(step)

        # 记录自定义指标
        langfuse_context.update_current_observation(
            metadata={"step": step.name, "tool": step.tool},
            level="DEFAULT",
        )

    # 3. 记录最终评分
    langfuse_context.score_current_trace(
        name="task_completion",
        value=1.0,  # 0-1
        comment="任务成功完成",
    )

    return result

@observe()
def execute_step(step):
    """每个步骤也被独立追踪"""
    if step.requires_tool:
        # 工具调用 → 自动记录输入/输出/延迟
        return call_tool(step.tool, step.args)
    else:
        return llm_generate(step.prompt)
```

**Trace 可视化示例：**

```
Trace: agent_pipeline (总耗时: 8.2s, 总 Token: 3,847)
│
├── plan_task (1.2s, 512 tokens)
│   └── ChatOpenAI.gpt-4o (1.1s)
│
├── execute_step: "搜索文档" (2.1s, 890 tokens)
│   ├── ChatOpenAI.gpt-4o (0.8s)
│   └── tool: search_docs (1.2s) ✅
│
├── execute_step: "分析结果" (1.8s, 1,245 tokens)
│   └── ChatOpenAI.gpt-4o (1.7s)
│
├── execute_step: "生成回答" (2.9s, 1,200 tokens)
│   ├── ChatOpenAI.gpt-4o (2.0s)
│   └── tool: format_output (0.8s) ✅
│
└── Score: task_completion = 1.0 ✅
```

**生产告警规则示例：**

```python
alert_rules = {
    "loop_detection": {
        "condition": "steps_count > 15",
        "action": "force_stop + alert",
        "message": "Agent 可能陷入循环",
    },
    "cost_anomaly": {
        "condition": "task_cost > $0.50",
        "action": "alert",
        "message": "单任务成本异常",
    },
    "latency_spike": {
        "condition": "p95_latency > 30s",
        "action": "alert",
        "message": "Agent 响应时间过长",
    },
    "tool_failure_rate": {
        "condition": "tool_success_rate < 0.9",
        "action": "alert + fallback",
        "message": "工具调用失败率过高",
    },
}
```

**面试话术：**

> "Agent 可观测性的核心挑战是 **非确定性执行路径**——同一个输入可能走完全不同的步骤。所以我们需要完整的 Trace 追踪（看清每一步做了什么）、关键指标监控（延迟、成本、成功率）、以及智能告警（循环检测、成本异常）。工具选择上，LangSmith 最完整但 SaaS 锁定，Langfuse 开源可自托管，如果已有 OTEL 基础设施可以用 OpenTelemetry 扩展。"

---

## 6.12 Voyager：终身学习 Agent

### Q: 什么是 Voyager？它如何实现技能积累和终身学习？

**Voyager 是 NVIDIA 在 2023 年发布的里程碑式 Agent，在 Minecraft 中实现了"终身学习"——Agent 能自主探索、发现新技能、存储技能库，并在未来的任务中复用之前学到的技能。**

**核心创新：不只是完成单个任务，而是在持续交互中不断积累能力，像人类一样"越用越强"。**

```
Voyager 架构：

  ┌─────────────────────────────────────────────────┐
  │                 Voyager Agent                    │
  │                                                  │
  │  ┌──────────────────┐  ┌───────────────────┐    │
  │  │ Automatic         │  │ Skill Library      │    │
  │  │ Curriculum        │  │ (技能库)           │    │
  │  │ (自动课程)        │  │                    │    │
  │  │                   │  │ · mine_wood()      │    │
  │  │ 根据当前能力      │  │ · build_house()    │    │
  │  │ 自动生成下一个    │  │ · craft_sword()    │    │
  │  │ 合适难度的任务    │  │ · fight_zombie()   │    │
  │  └──────────┬───────┘  │ · ...              │    │
  │             │          └────────┬──────────┘    │
  │             ▼                   │               │
  │  ┌──────────────────┐          │               │
  │  │ Iterative         │◄─────────┘               │
  │  │ Prompting         │  检索已有技能             │
  │  │ Mechanism         │  组合/复用                │
  │  │ (迭代提示机制)    │                           │
  │  │                   │                           │
  │  │ 生成代码 →        │                           │
  │  │ 执行 →            │                           │
  │  │ 验证 →            │                           │
  │  │ 修复/存储         │                           │
  │  └──────────────────┘                           │
  └─────────────────────────────────────────────────┘
```

**Voyager 的三大核心组件：**

**1. Automatic Curriculum（自动课程）**

```python
# 根据 Agent 当前的状态和能力，自动生成下一个"学习目标"
# 前端类比：就像一个自适应学习平台，根据你的水平推荐下一道题

class AutomaticCurriculum:
    def suggest_next_task(self, agent_state: dict) -> str:
        """
        输入：Agent 当前拥有的物品、技能、探索进度
        输出：下一个合适难度的任务

        原则：
        - 不能太简单（已经会的不重复）
        - 不能太难（需要先掌握前置技能）
        - 鼓励探索新领域
        """
        prompt = f"""
当前状态：
- 已有物品：{agent_state['inventory']}
- 已掌握技能：{agent_state['skills']}
- 已探索区域：{agent_state['explored']}

请建议一个合适的下一个目标，要求：
1. 比已掌握的技能稍微难一点
2. 可以解锁新的能力或物品
3. 和之前的目标不重复
"""
        return llm_call(prompt)
```

**2. Skill Library（技能库）**

```python
# 技能库 = 代码片段 + 向量索引
# 每个技能是一个可执行的 JavaScript 函数（Minecraft Mineflayer API）
# 前端类比：npm 包 / 可复用的 React Hook 库

class SkillLibrary:
    def __init__(self):
        self.skills = {}  # name → code
        self.vector_db = VectorStore()  # 语义检索

    def add_skill(self, name: str, code: str, description: str):
        """验证通过的技能，存入技能库"""
        self.skills[name] = code
        self.vector_db.add(
            text=f"{name}: {description}",
            metadata={"code": code}
        )

    def retrieve_skills(self, task: str, top_k: int = 5) -> list:
        """根据任务描述，检索最相关的已有技能"""
        return self.vector_db.search(task, top_k=top_k)

# 技能示例
skill_example = {
    "name": "mine_diamond",
    "description": "挖掘钻石矿石，需要铁镐或以上工具",
    "code": """
async function mineDiamond(bot) {
    // 1. 检查是否有铁镐
    const pickaxe = bot.inventory.items().find(
        item => item.name === 'iron_pickaxe'
    );
    if (!pickaxe) {
        await craftIronPickaxe(bot);  // 复用已有技能！
    }

    // 2. 找到钻石矿
    const diamondBlock = bot.findBlock({
        matching: mcData.blocksByName.diamond_ore.id,
        maxDistance: 64
    });

    // 3. 导航并挖掘
    await bot.pathfinder.goto(diamondBlock.position);
    await bot.dig(diamondBlock);
}
"""
}
```

**3. Iterative Prompting Mechanism（迭代提示机制）**

```python
def iterative_code_generation(task: str, skill_library: SkillLibrary):
    """
    生成代码 → 执行 → 检查结果 → 修复错误 → 重复
    直到代码正确运行
    """
    # 检索相关技能
    related_skills = skill_library.retrieve_skills(task)

    code = None
    for attempt in range(5):
        if code is None:
            # 首次生成
            prompt = f"""
任务：{task}
可复用的已有技能：
{related_skills}

请生成完成任务的 JavaScript 代码：
"""
        else:
            # 修复错误
            prompt = f"""
任务：{task}
上次生成的代码：{code}
执行错误：{error}
环境反馈：{feedback}

请修复代码：
"""

        code = llm_call(prompt)

        # 在 Minecraft 中执行
        success, error, feedback = execute_in_minecraft(code)

        if success:
            # 成功！存入技能库
            skill_library.add_skill(
                name=extract_skill_name(task),
                code=code,
                description=task,
            )
            return code

    return None  # 5次未成功
```

**Voyager 的核心理念对 Agent 设计的启示：**

| 启示 | Voyager 做法 | 通用 Agent 应用 |
|------|-------------|----------------|
| **技能复用** | 代码存入技能库，按需检索 | Agent 的成功经验存入向量库，供未来检索 |
| **自适应难度** | 自动课程根据能力推荐任务 | Agent 根据历史表现调整任务复杂度 |
| **迭代修复** | 代码错误自动修复，最多重试5次 | Agent 执行失败自动重试+反思 |
| **能力积累** | 技能库随时间增长 | Agent 的工具库/经验库持续扩展 |
| **组合创新** | 复杂技能由简单技能组合而成 | 复杂任务拆解为已有能力的组合 |

**前端类比：** Voyager 的 Skill Library 就像一个持续增长的 **npm registry**。每个技能是一个可复用的"包"，新任务优先检索已有的"包"来组合使用，只在没有匹配的包时才从头写。Agent 通过不断积累技能"包"，变得越来越强大——这就是 **终身学习** 的本质。

**面试话术：**

> "Voyager 最重要的贡献是提出了 **Agent 终身学习** 的完整框架。它的三个组件——自动课程（决定学什么）、迭代提示（怎么学）、技能库（存储学到的东西）——构成了一个正向循环：学得越多 → 技能库越丰富 → 能解决的任务越复杂 → 学到更多新技能。这个框架可以迁移到任何领域的 Agent 设计中。比如 Coding Agent 可以积累常用代码模式，Customer Service Agent 可以积累常见问题解决方案。"

---

## 本章总结

```
AI Agent 核心知识回顾：

  1. Agent = LLM + Tools + Memory + Planning（不只是调 API）

  2. ReAct = Thought → Action → Observation 循环
     （最经典的 Agent 执行模式）

  3. Function Calling = 结构化的工具调用
     （LLM 输出 JSON 指令，应用层执行）

  4. 规划策略：
     Plan-and-Solve（一次规划）
     REWOO（无观察规划，省 Token）
     Dynamic Replanning（边做边调整）

  5. 反思机制：
     Reflexion（自然语言反思 + 跨任务记忆）
     LATS（树搜索找最优路径）
     Generator-Evaluator（生成-评估循环）

  6. 记忆架构：
     Flat Vector → Episodic → Graph → Hybrid
     框架：Mem0 / Zep / Letta

  7. Multi-Agent：
     AutoGen（企业级）vs CrewAI（快速原型）
     A2A 协议（Agent 互联互通标准）

  8. Copilot（人在环路）vs Agent（自主执行）
     → 最佳实践：默认 Agent，关键节点 Copilot

  9. Chain（线性确定性）vs Loop（循环探索）
     → 生产系统通常混合使用

  10. LangGraph = 状态机 + 有向图
      → Agent 工作流的可视化、可控实现

  11. 可观测性 = Traces + Metrics + Logs
      → 工具：LangSmith / Langfuse / OTEL

  12. Voyager = 终身学习 Agent
      → 自动课程 + 迭代提示 + 技能库

  13. 防死循环：max_iterations + 重复检测 + 超时
  14. 工具失败：重试 + 降级 + 参数修正
  15. HITL：4 级风险 → 自动/通知/审批/禁止
```

---

## 6.13 Agent 防死循环与容错

### Q: Agent 如何防止无限循环？生产环境有哪些保护措施？

**Agent 死循环的三种常见原因：**

```
1. 工具调用循环：Agent 反复调用同一工具，参数相同，期望不同结果
2. 思考循环：Agent 在 Thought 阶段反复思考同一个问题
3. 乒乓循环（Multi-Agent）：Agent A 让 Agent B 做，B 又让 A 做
```

**五层防护机制：**

```python
class SafeAgentExecutor:
    def __init__(self):
        self.max_iterations = 10          # 第1层：硬性迭代上限
        self.max_time_seconds = 120       # 第2层：超时保护
        self.action_history = []          # 第3层：重复检测

    def run(self, query):
        start_time = time.time()

        for i in range(self.max_iterations):
            # 第2层：超时检查
            if time.time() - start_time > self.max_time_seconds:
                return self.fallback("执行超时")

            thought, action = self.llm_decide(query)

            # 第3层：重复行为检测
            action_sig = f"{action.tool}:{action.params}"
            if self.action_history.count(action_sig) >= 3:
                return self.fallback("检测到重复操作")
            self.action_history.append(action_sig)

            # 第4层：Token 预算
            if self.total_tokens > self.token_budget:
                return self.fallback("Token 预算耗尽")

            result = self.execute_action(action)

            # 第5层：进度检测（连续 N 步无进展）
            if self.no_progress_count >= 3:
                return self.escalate_or_fallback()

            if action.type == "final_answer":
                return result

        return self.fallback("达到最大迭代次数")
```

| 防护层 | 机制 | 触发条件 | 处理方式 |
|--------|------|----------|----------|
| 1. 迭代上限 | `max_iterations` | 超过 N 步 | 强制结束 |
| 2. 超时保护 | `max_time` | 超过 T 秒 | 中断并回复 |
| 3. 重复检测 | 行为签名去重 | 同一操作 ≥3 次 | 换策略或结束 |
| 4. Token 预算 | 累计 token 计数 | 超出预算 | 降级模型或结束 |
| 5. 进度检测 | 评估任务进展 | 连续 3 步无进展 | 提升策略或人工介入 |

**面试话术：**
> "Agent 防死循环是生产级系统的必备能力。我实现五层防护：迭代上限兜底、超时熔断、重复行为检测（同一工具调用 3 次以上触发）、Token 预算控制、进度评估（连续无进展就终止）。最关键的是第 3 层重复检测——通过对 action+params 做签名，发现循环行为。"

---

## 6.14 工具调用失败处理

### Q: Agent 的工具调用失败时，如何设计重试和降级策略？

**工具失败的四种类型及处理：**

| 失败类型 | 原因 | 处理策略 |
|----------|------|----------|
| **参数错误** | LLM 生成了不合法的参数 | 将错误信息反馈给 LLM，让它修正参数 |
| **超时** | 外部 API 响应慢 | 指数退避重试（最多 3 次）→ 降级到备选工具 |
| **权限不足** | 用户无权访问资源 | 告知用户，不重试 |
| **服务不可用** | API 宕机 | 降级到备选方案或缓存 |

**核心策略：让 LLM 自我修正**

```python
async def execute_with_recovery(self, tool_name, params, max_retries=3):
    for attempt in range(max_retries):
        try:
            result = await self.tools[tool_name].execute(params)
            return result
        except ParamError as e:
            # 策略 1：参数修正 — 让 LLM 看到错误后修正
            correction_prompt = f"""
            工具 {tool_name} 调用失败。
            参数：{params}
            错误：{e}
            请修正参数后重试。
            """
            params = await self.llm.correct_params(correction_prompt)
        except TimeoutError:
            # 策略 2：指数退避重试
            await asyncio.sleep(2 ** attempt)
        except PermissionError:
            # 策略 3：不重试，直接告知
            return {"error": "权限不足", "suggestion": "请联系管理员"}

    # 策略 4：降级 — 换备选工具
    fallback_tool = self.fallback_map.get(tool_name)
    if fallback_tool:
        return await fallback_tool.execute(params)

    return {"error": f"{tool_name} 不可用，已尝试 {max_retries} 次"}
```

**降级链设计：**

```
主工具失败 → 备选工具 → 缓存结果 → LLM 内部知识 → 告知用户

示例：
  搜索工具失败 → Bing API → 缓存搜索结果 → LLM 知识回答 → "我暂时无法搜索"
```

**面试话术：**
> "工具失败处理的核心是分类处置——参数错误让 LLM 自己看错误信息修正，超时做指数退避重试，权限错误直接告知不重试，服务不可用走降级链。关键设计是把错误信息作为 Observation 反馈给 LLM，利用它的自我修正能力。"

---

## 6.15 Human-in-the-Loop 系统设计

### Q: 如何设计 Human-in-the-Loop（人在环路）机制？哪些操作需要人工审批？

**核心原则：根据操作的风险等级，决定自动化程度。**

**四级风险框架：**

| 风险等级 | 操作类型 | 处理方式 | 示例 |
|----------|----------|----------|------|
| L1 安全 | 只读查询 | 全自动，无需人工 | 搜索、读取文件、查询数据库 |
| L2 低风险 | 可逆写操作 | 自动执行，事后通知 | 创建草稿、发送内部消息 |
| L3 中风险 | 不可逆操作 | 暂停，等待人工审批 | 发送邮件、修改数据库、调用付费 API |
| L4 高风险 | 关键操作 | 禁止自动化 | 删除数据、资金转账、修改权限 |

**实现架构：**

```python
class HumanInTheLoop:
    RISK_LEVELS = {
        "search": "L1",      # 自动
        "read_file": "L1",   # 自动
        "send_slack": "L2",  # 自动+通知
        "send_email": "L3",  # 需审批
        "delete_data": "L4", # 禁止
    }

    async def execute_with_approval(self, action):
        risk = self.RISK_LEVELS.get(action.tool, "L3")  # 未知工具默认中风险

        if risk == "L1":
            return await self.execute(action)

        elif risk == "L2":
            result = await self.execute(action)
            await self.notify_user(action, result)  # 事后通知
            return result

        elif risk == "L3":
            # 暂停，发审批请求
            approval = await self.request_approval(
                action=action,
                reason=f"Agent 需要执行: {action.tool}({action.params})",
                timeout=300  # 5 分钟超时
            )
            if approval.approved:
                return await self.execute(action)
            else:
                return {"status": "rejected", "reason": approval.reason}

        elif risk == "L4":
            return {"status": "blocked", "reason": "该操作需要人工执行"}
```

**审批界面的关键信息：**

```
┌─ Agent 审批请求 ────────────────────────────┐
│  操作：send_email                           │
│  收件人：client@example.com                 │
│  主题：合同确认                              │
│  内容预览：[展示完整邮件内容]                 │
│  风险等级：L3（不可逆）                       │
│  Agent 推理过程：[展示 Thought 链]            │
│                                              │
│  [✅ 批准]  [❌ 拒绝]  [✏️ 修改后批准]       │
└──────────────────────────────────────────────┘
```

**前端类比：** 这就像前端的权限系统（RBAC）——不同角色能执行不同操作。Agent 的 HITL 就是给 AI 操作加上了权限控制和审批流。

**面试话术：**
> "Human-in-the-Loop 的核心是四级风险分层——只读操作全自动（L1），可逆写操作自动+通知（L2），不可逆操作暂停等审批（L3），关键操作禁止自动化（L4）。未知工具默认 L3。这套机制让 Agent 在'够自主'和'够安全'之间取得平衡。"

---

## 6.16 Agent 评测 Benchmark

### Q: 如何系统性评测 Agent 的能力？有哪些主流 Benchmark？

**Agent 评测和 LLM 评测的区别：LLM 测"答得对不对"，Agent 测"做得成不成"。**

| Benchmark | 评测内容 | 指标 | 代表场景 |
|-----------|---------|------|----------|
| **SWE-bench** | 代码修复能力 | 解决率 % | 给 GitHub issue，Agent 能否自动修 bug |
| **WebArena** | Web 操作能力 | 任务完成率 | 在真实网站上完成任务（订票、购物） |
| **GAIA** | 通用 Agent 能力 | 准确率 | 多步推理 + 工具使用 + 信息检索 |
| **AgentBench** | 综合 Agent 能力 | 多维度打分 | 8 个环境：OS/DB/Web/Game... |
| **ToolBench** | 工具调用能力 | 通过率 | 16000+ API 的工具选择和调用 |
| **BFCL** | Function Calling 质量 | AST 准确率 | 参数类型、格式、嵌套调用 |

**自建评测的核心维度：**

```
Agent 评测五维模型：

1. 任务完成率     → 最终结果是否正确
2. 步骤效率       → 用了多少步完成（越少越好）
3. 工具使用准确率  → 调用了正确的工具 + 正确的参数
4. 容错能力       → 遇到错误能否恢复
5. 成本效率       → 消耗了多少 Token / 多长时间
```

**面试话术：**
> "Agent 评测不能只看最终结果，还要看过程效率。我关注五个维度：任务完成率、步骤效率（同样的任务用更少步骤）、工具调用准确率、容错恢复能力、Token 成本。业界有 SWE-bench（代码修复）、GAIA（通用推理）、BFCL（Function Calling）等标准 Benchmark。"

---

## 6.17 Extended Thinking / Thinking Budget

### Q: 什么是 Extended Thinking？Thinking Budget 如何控制 AI 的 "思考量"？

**Extended Thinking = 让模型在回答前进行"深度思考"，输出思维链（thinking tokens），然后给出最终答案。**

```
普通模式：
  User: "这个 bug 怎么修？"
  Assistant: "你可以改第 42 行..."  （直接回答）

Extended Thinking 模式：
  User: "这个 bug 怎么修？"
  [Thinking]: "让我分析这段代码... 第 42 行有个边界条件...
               可能的原因有三个... 方案 A 可行但有副作用...
               方案 B 更安全..."                          （思考过程，用户可见或不可见）
  Assistant: "建议用方案 B，修改第 42 行为..."              （最终答案）
```

**Thinking Budget = 控制模型花多少 token 在"思考"上。**

| Budget | 行为 | 适用场景 |
|--------|------|----------|
| 0（关闭） | 不思考，直接回答 | 简单问答、闲聊 |
| 1024 tokens | 轻度思考 | 一般任务 |
| 10000 tokens | 深度思考 | 复杂推理、数学、编程 |
| 最大值 | 充分思考 | 极难问题（代价是慢和贵） |

```python
# Claude API 示例
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=8000,
    thinking={
        "type": "enabled",
        "budget_tokens": 5000  # 最多用 5000 tokens 思考
    },
    messages=[{"role": "user", "content": "分析这段代码的性能瓶颈"}]
)

# 思考过程在 response.content 中，type="thinking" 的 block
for block in response.content:
    if block.type == "thinking":
        print(f"思考过程：{block.thinking}")
    elif block.type == "text":
        print(f"最终答案：{block.text}")
```

**面试话术：**
> "Extended Thinking 是 2025-2026 年的重要能力突破——让模型先'想清楚'再回答。Thinking Budget 控制思考深度，简单问题给少一点（省成本），复杂推理给多一点（提质量）。DeepSeek-R1 的长思维链、Claude 的 Extended Thinking 都是这个方向。核心取舍是质量 vs 成本 vs 延迟。"

---

## 6.18 AutoGPT

### Q: AutoGPT 是什么？它和普通 Agent 有什么区别？为什么它"失败"了？

**AutoGPT（2023 年 3 月）= 第一个引发全球关注的自主 Agent 项目，让 GPT-4 自己给自己下任务、自己执行、自己评估。**

**AutoGPT vs 普通 Agent：**

| 维度 | 普通 Agent（ReAct） | AutoGPT |
|------|---------------------|---------|
| 目标 | 用户给具体任务 | 用户给高层目标，Agent 自己拆解 |
| 循环 | Thought→Action→Observation | Goal→Plan→Execute→Evaluate→Replan |
| 人工介入 | 每步可控 | 完全自主（无人值守） |
| 记忆 | 短期（上下文内） | 长期（向量存储跨会话） |
| 典型步数 | 3-10 步 | 几十到上百步 |

**AutoGPT 的核心循环：**

```
用户输入：目标 = "调研 AI 编程工具市场并写一份报告"

AutoGPT 循环：
  1. 自我规划：拆解为 5 个子任务
  2. 执行子任务 1：用搜索工具查资料
  3. 自我评估：信息够不够？不够，继续搜索
  4. 执行子任务 2：整理搜索结果
  5. 自我反思：还需要什么？价格对比数据
  6. ...循环 20+ 轮...
  7. 最终输出：写成报告
```

**为什么 AutoGPT "失败"了？三个致命问题：**

| 问题 | 说明 | 后果 |
|------|------|------|
| **目标漂移** | Agent 在多轮循环中逐渐偏离原始目标 | 做了 50 步但结果偏题 |
| **成本爆炸** | 每轮都调用 GPT-4，几十轮下来成本极高 | 一次任务花几十美元 |
| **质量不可控** | 无人监督，错误逐步累积，无法纠偏 | 输出不可信 |

**AutoGPT 的启示（面试重点）：**

```
AutoGPT 教会我们的三件事：

1. 完全自主 ≠ 好
   → 需要 Human-in-the-Loop 在关键节点介入

2. 无限循环需要约束
   → 需要 max_iterations、Token 预算、进度检测

3. Agent 的价值在于可控的自主性
   → 2026 年主流是"有护栏的 Agent"（Copilot 模式）
   → 不是 AutoGPT 那样的"放飞自我"
```

**面试话术：**
> "AutoGPT 是第一个让公众理解 Agent 概念的项目，但它证明了'完全自主'在工程上不可行——目标漂移、成本爆炸、质量不可控。2026 年的 Agent 设计汲取了这个教训：用 ReAct/LangGraph 做可控循环，用 HITL 在关键节点审批，用 Token 预算和迭代上限兜底。核心是'可控的自主性'，而不是 AutoGPT 式的无限自主。"

---

## 导航

| 上一章 | 当前章 | 下一章 |
|--------|--------|--------|
| [05 - RAG 系统](./05-rag.md) | **06 - AI Agent** | [07 - MCP 协议](./07-mcp.md) |
