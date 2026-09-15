import os
import json
import asyncio

from dataclasses import dataclass, field
from openai import AsyncOpenAI

from rag_agent import search_knowledge


# ============================================================
# 1. Agent State
# ============================================================

@dataclass
class AgentState:

    # --------------------------------------------------------
    # 保存整个对话历史
    # --------------------------------------------------------
    messages: list = field(default_factory=list)

    # --------------------------------------------------------
    # 当前 Agent 执行步骤
    #
    # 注意：
    # 这里先简单使用。
    # 实际项目中通常会针对每次任务单独计算 step。
    # --------------------------------------------------------
    step: int = 0

    # --------------------------------------------------------
    # 当前任务是否完成
    # --------------------------------------------------------
    finished: bool = False


# ============================================================
# 2. DeepSeek Client
# ============================================================

client = AsyncOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


# ============================================================
# 3. Tools
# ============================================================

def add(a: int, b: int):
    return a + b


def multiply(a: int, b: int):
    return a * b


def subtract(a: int, b: int):
    return a - b


# ============================================================
# 4. Tool Registry
# ============================================================

TOOL_REGISTRY = {
    "add": add,
    "multiply": multiply,
    "subtract": subtract,
    "search_knowledge": search_knowledge,
}


# ============================================================
# 5. Tool Schema
# ============================================================

TOOLS = [

    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "计算两个数字的和",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "计算两个数字的乘积",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "subtract",
            "description": "计算两个数字的差",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "搜索知识库",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string"
                    }
                },
                "required": ["query"],
            },
        },
    },
]


# ============================================================
# 6. Dispatcher
# ============================================================

def dispatch_tool(tool_name: str, arguments: dict):

    print(f"\n[Tool] {tool_name}")
    print(f"[Args] {arguments}")

    tool = TOOL_REGISTRY.get(tool_name)

    if tool is None:
        return f"Tool 不存在：{tool_name}"

    try:

        result = tool(**arguments)

        print(f"[Result] {result}")

        return str(result)

    except Exception as e:

        return f"Tool 执行失败：{e}"


# ============================================================
# 7. Agent
#
# 注意：
# Agent 不再创建 State。
#
# State 从外部传入。
# ============================================================

async def agent(state: AgentState):

    MAX_STEPS = 10

    # --------------------------------------------------------
    # 每次处理一个用户问题时，
    # 重新计算本次 Agent Loop 的步骤。
    # --------------------------------------------------------

    for step in range(MAX_STEPS):

        state.step = step + 1

        print(
            f"\n========== Agent Step "
            f"{state.step} =========="
        )

        # ----------------------------------------------------
        # 调用 LLM
        # ----------------------------------------------------

        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=state.messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        message = response.choices[0].message

        # ----------------------------------------------------
        # 没有 Tool Call
        #
        # 说明本轮任务完成
        # ----------------------------------------------------

        if not message.tool_calls:

            state.finished = True

            return message.content

        # ----------------------------------------------------
        # 保存 assistant Tool Call
        # ----------------------------------------------------

        state.messages.append(message)

        # ----------------------------------------------------
        # 执行 Tool
        # ----------------------------------------------------

        for tool_call in message.tool_calls:

            tool_name = tool_call.function.name

            arguments = json.loads(
                tool_call.function.arguments
            )

            result = dispatch_tool(
                tool_name,
                arguments
            )

            # ------------------------------------------------
            # 保存 Tool Result
            # ------------------------------------------------

            state.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )

    return "Agent 执行超过最大步骤限制。"


# ============================================================
# 8. Main
# ============================================================

async def main():

    # ========================================================
    # 创建一个长期存在的 State
    # ========================================================

    state = AgentState()

    # ========================================================
    # 初始化 System Prompt
    # ========================================================

    state.messages.append(
        {
            "role": "system",
            "content": """
你是一个智能 Agent。

你可以使用：

1. add
2. multiply
3. subtract
4. search_knowledge

根据用户问题自主决定是否调用工具。

如果任务需要多个步骤，
可以连续调用多个工具。

只有任务完成后才返回最终答案。
"""
        }
    )

    # ========================================================
    # 多轮对话
    # ========================================================

    while True:

        question = input("\n用户：")

        # ----------------------------------------------------
        # 退出
        # ----------------------------------------------------

        if question.lower() == "exit":
            print("Agent 已退出。")
            break

        # ----------------------------------------------------
        # 把用户消息加入 State
        # ----------------------------------------------------

        state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        # ----------------------------------------------------
        # 调用 Agent
        #
        # 注意：
        # 传入的是同一个 state。
        # ----------------------------------------------------

        answer = await agent(state)

        print(f"\nAgent：{answer}")


# ============================================================
# 9. 启动
# ============================================================

if __name__ == "__main__":
    asyncio.run(main())