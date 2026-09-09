from openai import OpenAI
import json
import os


# ============================================================
# 1. 创建 DeepSeek 客户端
# ============================================================

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not DEEPSEEK_API_KEY:
    raise ValueError(
        "没有找到 DEEPSEEK_API_KEY 环境变量，"
        "请先运行: set DEEPSEEK_API_KEY=你的_API_KEY"
    )

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)


# ============================================================
# 2. 定义真正执行 Tool 的 Python 函数
#
# 注意：
# LLM 不会直接执行这些函数。
# LLM 只负责告诉我们：
#
# "请调用 add，并传入 a=10, b=20"
#
# 真正执行的是下面的 Python 程序。
# ============================================================

def add(a, b):
    return a + b


def multiply(a, b):
    return a * b


# ============================================================
# 3. Tool Registry
#
# 先简单理解成：
#
# Tool 名称
#     ↓
# Python 函数
#
# 以后我们会专门学习 Tool Registry / Dispatcher。
# ============================================================

tools_map = {
    "add": add,
    "multiply": multiply
}


# ============================================================
# 4. 给 LLM 描述有哪些 Tool
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
                    "a": {
                        "type": "number",
                        "description": "第一个数字"
                    },
                    "b": {
                        "type": "number",
                        "description": "第二个数字"
                    }
                },
                "required": ["a", "b"]
            }
        }
    }
]


# ============================================================
# 5. 保存对话上下文
# ============================================================

messages = [
    {
        "role": "user",
        "content": "同时计算 10 + 20，以及 8 × 9。"
    }
]


# ============================================================
# 6. Agent / Tool Calling Loop
# ============================================================

while True:

    print("\n================ 调用 DeepSeek ================\n")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    message = response.choices[0].message

    # --------------------------------------------------------
    # 非常重要：
    # 把 LLM 返回的 message 加入上下文
    #
    # 因为下一轮 LLM 需要知道：
    # "刚才我要求调用了哪些 Tool"
    # --------------------------------------------------------

    messages.append(message)

    # --------------------------------------------------------
    # 如果 LLM 没有调用 Tool
    #
    # 说明 LLM 已经可以直接回答用户了
    # --------------------------------------------------------

    if not message.tool_calls:

        print("================ 最终回答 ================\n")
        print(message.content)

        break


    # ========================================================
    # 7. 处理多个 Tool Call
    # ========================================================
    #
    # 注意这里：
    #
    # 不再假设只有一个 Tool Call。
    #
    # message.tool_calls 是一个列表。
    #
    # 例如：
    #
    # [
    #     add(10, 20),
    #     multiply(8, 9)
    # ]
    #
    # 所以需要遍历。
    # ========================================================

    for tool_call in message.tool_calls:

        # ----------------------------------------------------
        # 获取 Tool 名称
        # ----------------------------------------------------

        tool_name = tool_call.function.name

        # ----------------------------------------------------
        # 获取 Tool 参数
        #
        # DeepSeek 返回的是 JSON 字符串，
        # 所以需要 json.loads() 转成 Python dict。
        # ----------------------------------------------------

        arguments = json.loads(tool_call.function.arguments)

        print("LLM 请求调用 Tool:", tool_name)
        print("Tool 参数:", arguments)

        # ----------------------------------------------------
        # 根据 Tool 名称找到真正的 Python 函数
        # ----------------------------------------------------

        tool_function = tools_map[tool_name]

        # ----------------------------------------------------
        # 执行 Tool
        # ----------------------------------------------------

        result = tool_function(**arguments)

        print("Tool 执行结果:", result)
        print()

        # ----------------------------------------------------
        # 把 Tool 执行结果返回给 LLM
        #
        # tool_call_id 非常重要。
        #
        # 它告诉 LLM：
        #
        # "这个结果对应刚才哪一个 Tool Call"
        # ----------------------------------------------------

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            }
        )
        