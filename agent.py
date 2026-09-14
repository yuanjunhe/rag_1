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
    # 保存 LLM 对话历史
    #
    # 这是 Agent 最核心的状态。
    # --------------------------------------------------------
    messages: list = field(default_factory=list)

    # --------------------------------------------------------
    # 当前 Agent 执行到第几步
    # --------------------------------------------------------
    step: int = 0

    # --------------------------------------------------------
    # Agent 是否已经完成任务
    # --------------------------------------------------------
    finished: bool = False


# ============================================================
# 2. DeepSeek
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
# ============================================================

async def agent(user_question: str):

    # --------------------------------------------------------
    # 创建 Agent State
    # --------------------------------------------------------

    state = AgentState()

    # --------------------------------------------------------
    # 初始化消息
    # --------------------------------------------------------

    state.messages = [

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
        },

        {
            "role": "user",
            "content": user_question
        }
    ]

    # ========================================================
    # Agent Loop
    # ========================================================

    MAX_STEPS = 10

    while state.step < MAX_STEPS:

        # ----------------------------------------------------
        # Step +1
        # ----------------------------------------------------

        state.step += 1

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
        # Agent 认为任务已经完成。
        # ----------------------------------------------------

        if not message.tool_calls:

            state.finished = True

            return message.content

        # ----------------------------------------------------
        # 保存 assistant 消息
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

    # ========================================================
    # 超过最大步骤
    # ========================================================

    return "Agent 执行超过最大步骤限制。"


# ============================================================
# 8. Main
# ============================================================

async def main():

    question = input("请输入问题：")

    answer = await agent(question)

    print("\n========== 最终答案 ==========")
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())