import asyncio
import json
import os
from openai import AsyncOpenAI


# ============================================================
# 1. DeepSeek Client
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
# 2. Tool
# ============================================================

async def add(a, b):
    await asyncio.sleep(1)

    return a + b


async def multiply(a, b):
    await asyncio.sleep(1)

    return a * b


# ============================================================
# 3. Tool Registry
# ============================================================

TOOL_REGISTRY = {
    "add": add,
    "multiply": multiply
}


# ============================================================
# 4. 参数校验器
#
# 这是本阶段最重要的新代码。
#
# 它负责检查：
#
# ① 参数是否存在
# ② 参数类型是否正确
#
# 注意：
# 这里只实现一个最简单的版本。
# ============================================================

def validate_arguments(tool_name, arguments):

    # --------------------------------------------------------
    # add
    # --------------------------------------------------------

    if tool_name == "add":

        # 检查 a
        if "a" not in arguments:
            raise ValueError("缺少参数：a")

        # 检查 b
        if "b" not in arguments:
            raise ValueError("缺少参数：b")

        # 检查类型
        if not isinstance(arguments["a"], (int, float)):
            raise TypeError("参数 a 必须是数字")

        if not isinstance(arguments["b"], (int, float)):
            raise TypeError("参数 b 必须是数字")


    # --------------------------------------------------------
    # multiply
    # --------------------------------------------------------

    elif tool_name == "multiply":

        if "a" not in arguments:
            raise ValueError("缺少参数：a")

        if "b" not in arguments:
            raise ValueError("缺少参数：b")

        if not isinstance(arguments["a"], (int, float)):
            raise TypeError("参数 a 必须是数字")

        if not isinstance(arguments["b"], (int, float)):
            raise TypeError("参数 b 必须是数字")


    else:

        raise ValueError(
            f"未知 Tool：{tool_name}"
        )


# ============================================================
# 5. Dispatcher
# ============================================================

async def dispatch_tool(tool_call):

    tool_name = tool_call.function.name

    # --------------------------------------------------------
    # 解析 JSON 参数
    # --------------------------------------------------------

    try:

        arguments = json.loads(
            tool_call.function.arguments
        )

    except json.JSONDecodeError:

        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": "Tool 参数不是合法 JSON"
        }


    print(
        f"\nDispatcher 收到请求："
        f"{tool_name}"
    )

    print(
        f"Tool 参数：{arguments}"
    )


    # ========================================================
    # 第一步：检查 Tool 是否存在
    # ========================================================

    tool_function = TOOL_REGISTRY.get(tool_name)

    if tool_function is None:

        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": f"Tool 不存在：{tool_name}"
        }


    # ========================================================
    # 第二步：参数校验
    # ========================================================

    try:

        validate_arguments(
            tool_name,
            arguments
        )

    except (ValueError, TypeError) as e:

        print(
            f"参数校验失败：{e}"
        )

        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": f"参数错误：{e}"
        }


    # ========================================================
    # 第三步：执行 Tool
    # ========================================================

    try:

        result = await tool_function(
            **arguments
        )

        print(
            f"Tool 执行成功："
            f"{tool_name} → {result}"
        )

        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": str(result)
        }


    # ========================================================
    # 第四步：Tool 执行异常
    # ========================================================

    except Exception as e:

        print(
            f"Tool 执行异常：{e}"
        )

        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": f"Tool 执行失败：{e}"
        }


# ============================================================
# 6. Tool Schema
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

                    "a": {
                        "type": "number",
                        "description": "第一个数字"
                    },

                    "b": {
                        "type": "number",
                        "description": "第二个数字"
                    }
                },

                "required": [
                    "a",
                    "b"
                ]
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

                    "a": {
                        "type": "number"
                    },

                    "b": {
                        "type": "number"
                    }
                },

                "required": [
                    "a",
                    "b"
                ]
            }
        }
    }
]


# ============================================================
# 7. Agent Loop
# ============================================================

async def main():

    messages = [
        {
            "role": "user",
            "content": (
                "同时计算 "
                "10 + 20，"
                "以及 8 × 9。"
            )
        }
    ]


    while True:

        print(
            "\n================ "
            "调用 DeepSeek "
            "================"
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
        # ----------------------------------------------------

        if not message.tool_calls:

            print(
                "\n================ "
                "最终回答 "
                "================"
            )

            print(message.content)

            break


        # ----------------------------------------------------
        # 并行执行多个 Tool
        # ----------------------------------------------------

        tasks = [

            dispatch_tool(tool_call)

            for tool_call in message.tool_calls
        ]


        results = await asyncio.gather(
            *tasks
        )


        # ----------------------------------------------------
        # Tool Result 返回给 LLM
        # ----------------------------------------------------

        messages.extend(results)


# ============================================================
# 8. 启动
# ============================================================

if __name__ == "__main__":

    asyncio.run(main())