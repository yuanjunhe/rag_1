import asyncio
import json
from openai import AsyncOpenAI
import os


# ============================================================
# 1. 创建 DeepSeek 异步客户端
# ============================================================

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not DEEPSEEK_API_KEY:
    raise ValueError(
        "没有找到 DEEPSEEK_API_KEY 环境变量，"
        "请先运行: set DEEPSEEK_API_KEY=你的_API_KEY"
    )

client = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

# ============================================================
# 2. 定义 Tool
# ============================================================

async def add(a, b):
    # 模拟一个耗时操作
    await asyncio.sleep(2)

    return a + b


async def multiply(a, b):
    # 模拟一个耗时操作
    await asyncio.sleep(2)

    return a * b


# ============================================================
# 3. Tool Registry
#
# Tool 名称 -> Python 函数
# ============================================================

tools_map = {
    "add": add,
    "multiply": multiply
}


# ============================================================
# 4. 告诉 LLM 我们有哪些 Tool
# ============================================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "计算两个数字的和",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["a", "b"]
            }
        }
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
                    "b": {"type": "number"}
                },
                "required": ["a", "b"]
            }
        }
    }
]


# ============================================================
# 5. 执行单个 Tool
#
# 注意：
# 这里专门负责：
#
# Tool Call
#     ↓
# 找到 Python 函数
#     ↓
# 执行
#     ↓
# 返回结果
# ============================================================

async def execute_tool(tool_call):

    tool_name = tool_call.function.name

    # DeepSeek 返回的是 JSON 字符串
    arguments = json.loads(
        tool_call.function.arguments
    )

    print(
        f"开始执行 Tool: {tool_name}, "
        f"参数: {arguments}"
    )

    # 根据 Tool 名称找到 Python 函数
    tool_function = tools_map[tool_name]

    # 执行 Tool
    result = await tool_function(**arguments)

    print(
        f"Tool 执行完成: {tool_name}, "
        f"结果: {result}"
    )

    # 返回 Tool Result
    return {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": str(result)
    }


# ============================================================
# 6. Agent Loop
# ============================================================

async def main():

    messages = [
        {
            "role": "user",
            "content": "同时计算 10 + 20，以及 8 × 9。"
        }
    ]

    while True:

        print("\n================ 调用 DeepSeek ================\n")

        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        message = response.choices[0].message

        # 保存 LLM 的消息
        messages.append(message)

        # ====================================================
        # 如果没有 Tool Call
        #
        # 说明 LLM 已经得到足够的信息，可以直接回答。
        # ====================================================

        if not message.tool_calls:

            print("\n================ 最终回答 ================\n")

            print(message.content)

            break


        # ====================================================
        # 关键部分：
        #
        # message.tool_calls
        #
        # 可能包含：
        #
        # [
        #     add(10, 20),
        #     multiply(8, 9)
        # ]
        #
        # 我们把多个 Tool 同时交给 asyncio 执行。
        # ====================================================

        tasks = [
            execute_tool(tool_call)
            for tool_call in message.tool_calls
        ]

        # ====================================================
        # asyncio.gather()
        #
        # 同时执行所有 Tool
        #
        # 例如：
        #
        # add       ────────→ 30
        # multiply  ────────→ 72
        #
        # 两个任务同时进行。
        # ====================================================

        results = await asyncio.gather(*tasks)


        # ====================================================
        # 把所有 Tool Result 加入 messages
        #
        # 注意：
        # 这里不是把结果直接告诉用户。
        #
        # 而是：
        #
        # Tool Result
        #      ↓
        # messages
        #      ↓
        # LLM
        #
        # 最终由 LLM 组织自然语言回答。
        # ====================================================

        messages.extend(results)


# ============================================================
# 7. 启动程序
# ============================================================

if __name__ == "__main__":
    asyncio.run(main())