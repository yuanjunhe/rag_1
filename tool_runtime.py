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
# 模拟一个很慢的 Tool
#
# 实际项目中可能对应：
#
# HTTP 请求
# 数据库查询
# 第三方 API
# ============================================================

async def slow_tool():

    print("slow_tool 开始执行")

    # 故意等待 10 秒
    await asyncio.sleep(10)

    return "slow tool success"


# ============================================================
# 3. Tool Registry
# ============================================================

TOOL_REGISTRY = {

    "add": add,

    "multiply": multiply,

    "slow_tool": slow_tool
}


# ============================================================
# 4. 参数校验
# ============================================================

def validate_arguments(tool_name, arguments):

    if tool_name in ["add", "multiply"]:

        if "a" not in arguments:
            raise ValueError("缺少参数：a")

        if "b" not in arguments:
            raise ValueError("缺少参数：b")

        if not isinstance(
            arguments["a"],
            (int, float)
        ):
            raise TypeError(
                "参数 a 必须是数字"
            )

        if not isinstance(
            arguments["b"],
            (int, float)
        ):
            raise TypeError(
                "参数 b 必须是数字"
            )


# ============================================================
# 5. 执行 Tool
#
# 这里增加：
#
# ① 超时
# ② 重试
# ③ 异常处理
# ============================================================

async def execute_tool(
    tool_function,
    arguments,
    timeout=3,
    max_retries=3
):

    # ========================================================
    # 最多执行 max_retries 次
    # ========================================================

    for attempt in range(1, max_retries + 1):

        try:

            print(
                f"Tool 第 {attempt} 次执行"
            )

            # ------------------------------------------------
            # wait_for：
            #
            # 最多等待 timeout 秒
            # ------------------------------------------------

            result = await asyncio.wait_for(

                tool_function(**arguments),

                timeout=timeout
            )

            # ------------------------------------------------
            # 成功
            # ------------------------------------------------

            return result


        # ====================================================
        # Tool 超时
        # ====================================================

        except asyncio.TimeoutError:

            print(
                f"Tool 执行超时："
                f"{timeout} 秒"
            )

            # ------------------------------------------------
            # 超时后继续重试
            # ------------------------------------------------

            if attempt < max_retries:

                print("准备重试...")

                await asyncio.sleep(1)

            else:

                raise RuntimeError(
                    f"Tool 执行超时，"
                    f"已经重试 {max_retries} 次"
                )


        # ====================================================
        # Tool 普通异常
        # ====================================================

        except Exception as e:

            print(
                f"Tool 执行异常：{e}"
            )

            if attempt < max_retries:

                print("准备重试...")

                await asyncio.sleep(1)

            else:

                raise RuntimeError(
                    f"Tool 执行失败：{e}"
                )


# ============================================================
# 6. Dispatcher
# ============================================================

async def dispatch_tool(tool_call):

    # --------------------------------------------------------
    # 获取 Tool 名称
    # --------------------------------------------------------

    tool_name = tool_call.function.name


    # --------------------------------------------------------
    # 解析参数
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
        f"\nDispatcher 收到：{tool_name}"
    )

    print(
        f"参数：{arguments}"
    )


    # ========================================================
    # 1. 查找 Tool
    # ========================================================

    tool_function = TOOL_REGISTRY.get(
        tool_name
    )

    if tool_function is None:

        return {

            "role": "tool",

            "tool_call_id": tool_call.id,

            "content": (
                f"Tool 不存在：{tool_name}"
            )
        }


    # ========================================================
    # 2. 参数校验
    # ========================================================

    try:

        validate_arguments(
            tool_name,
            arguments
        )

    except (ValueError, TypeError) as e:

        return {

            "role": "tool",

            "tool_call_id": tool_call.id,

            "content": f"参数错误：{e}"
        }


    # ========================================================
    # 3. 执行 Tool
    #
    # 增加：
    #
    # timeout = 3 秒
    # max_retries = 3 次
    # ========================================================

    try:

        result = await execute_tool(

            tool_function,

            arguments,

            timeout=3,

            max_retries=3
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
    # 4. 最终失败
    # ========================================================

    except Exception as e:

        print(
            f"Tool 最终执行失败：{e}"
        )

        return {

            "role": "tool",

            "tool_call_id": tool_call.id,

            "content": (
                f"Tool 执行失败：{e}"
            )
        }


# ============================================================
# 7. Tool Schema
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
    },


    {
        "type": "function",

        "function": {

            "name": "slow_tool",

            "description": "一个执行时间很长的测试工具",

            "parameters": {

                "type": "object",

                "properties": {}
            }
        }
    }
]


# ============================================================
# 8. Agent Loop
# ============================================================

async def main():

    messages = [

        {
            "role": "user",

            "content": (
                "同时计算 10 + 20，"
                "8 × 9，"
                "并测试一下 slow_tool。"
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
        # LLM
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
        # 多 Tool 并行执行
        # ----------------------------------------------------

        tasks = [

            dispatch_tool(tool_call)

            for tool_call in message.tool_calls

        ]


        results = await asyncio.gather(
            *tasks
        )


        # ----------------------------------------------------
        # Tool Result 返回 LLM
        # ----------------------------------------------------

        messages.extend(results)


# ============================================================
# 9. 启动
# ============================================================

if __name__ == "__main__":

    asyncio.run(main())