import asyncio
import json
import os
from openai import AsyncOpenAI


# ============================================================
# 1. DeepSeek 客户端
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
    await asyncio.sleep(2)
    return a + b


async def multiply(a, b):
    await asyncio.sleep(2)
    return a * b


async def subtract(a, b):
    await asyncio.sleep(2)
    return a - b


# ============================================================
# 3. Tool Registry
#
# 作用：
# Tool 名称 -> Python 函数
#
# Agent 不需要知道具体函数在哪里。
# 只需要通过 Tool 名称找到对应函数。
# ============================================================

TOOL_REGISTRY = {
    "add": add,
    "multiply": multiply,
    "subtract": subtract
}


# ============================================================
# 4. Dispatcher
#
# Dispatcher 的职责：
#
# ① 根据 Tool 名称找到 Tool
# ② 解析参数
# ③ 执行 Tool
# ④ 返回 Tool Result
#
# 相当于 Agent 的"工具调度中心"
# ============================================================

async def dispatch_tool(tool_call):

    # --------------------------------------------------------
    # 获取 Tool 名称
    # --------------------------------------------------------

    tool_name = tool_call.function.name

    # --------------------------------------------------------
    # 获取 Tool 参数
    #
    # 例如：
    #
    # '{"a": 10, "b": 20}'
    #
    # 转成：
    #
    # {"a": 10, "b": 20}
    # --------------------------------------------------------

    arguments = json.loads(
        tool_call.function.arguments
    )

    print(
        f"Dispatcher 收到请求："
        f"{tool_name}({arguments})"
    )

    # --------------------------------------------------------
    # 从 Registry 查找 Tool
    # --------------------------------------------------------

    tool_function = TOOL_REGISTRY.get(tool_name)

    # --------------------------------------------------------
    # Tool 不存在
    # --------------------------------------------------------

    if tool_function is None:

        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": f"错误：Tool '{tool_name}' 不存在"
        }

    # --------------------------------------------------------
    # 执行 Tool
    # --------------------------------------------------------

    try:

        result = await tool_function(**arguments)

        print(
            f"Tool 执行完成："
            f"{tool_name} → {result}"
        )

        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": str(result)
        }

    # --------------------------------------------------------
    # Tool 执行异常
    #
    # 这里暂时简单捕获异常。
    # 下一阶段我们会专门学习：
    #
    # Tool 参数校验
    # Tool 异常处理
    # Tool 超时
    # --------------------------------------------------------

    except Exception as e:

        print(
            f"Tool 执行失败："
            f"{tool_name} → {e}"
        )

        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": f"Tool 执行失败：{str(e)}"
        }


# ============================================================
# 5. Tool Schema
#
# 注意：
#
# TOOL_REGISTRY 是给 Python 使用的。
#
# tools 是给 LLM 使用的。
#
# 两者不要混淆。
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
                    "b": {"type": "number"}
                },
                "required": ["a", "b"]
            }
        }
    }
]


# ============================================================
# 6. Agent Loop
# ============================================================

async def main():

    messages = [
        {
            "role": "user",
            "content": (
                "同时计算："
                "10 + 20，"
                "8 × 9，"
                "100 - 30。"
            )
        }
    ]

    while True:

        print(
            "\n================ "
            "调用 DeepSeek "
            "================\n"
        )

        # ----------------------------------------------------
        # 调用 LLM
        # ----------------------------------------------------

        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        message = response.choices[0].message

        # ----------------------------------------------------
        # 保存 LLM 消息
        # ----------------------------------------------------

        messages.append(message)

        # ----------------------------------------------------
        # 没有 Tool Call
        # → LLM 直接回答
        # ----------------------------------------------------

        if not message.tool_calls:

            print(
                "\n================ "
                "最终回答 "
                "================\n"
            )

            print(message.content)

            break

        # ----------------------------------------------------
        # 多 Tool Calling
        # ----------------------------------------------------

        print(
            f"LLM 请求调用 "
            f"{len(message.tool_calls)} 个 Tool"
        )

        # ----------------------------------------------------
        # 为每个 Tool 创建一个异步任务
        #
        # 注意：
        # 这里不再自己寻找 Tool。
        #
        # 全部交给 Dispatcher。
        # ----------------------------------------------------

        tasks = [
            dispatch_tool(tool_call)
            for tool_call in message.tool_calls
        ]

        # ----------------------------------------------------
        # 并行执行
        # ----------------------------------------------------

        results = await asyncio.gather(*tasks)

        # ----------------------------------------------------
        # 将所有 Tool Result 返回给 LLM
        # ----------------------------------------------------

        messages.extend(results)


# ============================================================
# 7. 启动 Agent
# ============================================================

if __name__ == "__main__":

    asyncio.run(main())