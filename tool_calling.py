from openai import OpenAI
import os
import json


# ============================================================
# 1. 初始化 DeepSeek 客户端
# ============================================================

# 从环境变量中读取 DeepSeek API Key
#
# 例如 Windows 中可以提前设置：
# set DEEPSEEK_API_KEY=你的API_KEY
#
# 这样就不用把 API Key 直接写在代码里。
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


# ============================================================
# 2. 定义真正执行的 Python Tool
# ============================================================

# 注意：
#
# 下面两个函数才是真正执行计算的地方。
#
# DeepSeek 本身不会执行这些 Python 函数。
#
# 整个过程是：
#
#     DeepSeek
#        ↓
#     决定调用哪个 Tool
#        ↓
#     返回 Tool 名称 + 参数
#        ↓
#     我们的 Python 程序
#        ↓
#     真正执行下面的函数
#
# 所以：
#
#     LLM = 决策者
#     Python = 执行者


def add(a, b):
    """
    计算两个数字的和。
    """
    return a + b


def multiply(a, b):
    """
    计算两个数字的积。
    """
    return a * b


# ============================================================
# 3. 定义 Tool Schema
# ============================================================

# 这里非常重要。
#
# 我们不是把 Python 函数直接交给 DeepSeek，
# 而是通过 JSON Schema 告诉 DeepSeek：
#
#     我有哪些 Tool？
#     每个 Tool 是干什么的？
#     Tool 需要什么参数？
#     参数是什么类型？
#
# DeepSeek 会根据这些描述决定是否调用 Tool。
#
# 注意：
#
#     Tool Schema ≠ Python 函数本身
#
# Tool Schema 是给 LLM "看"的。
#
# Python 函数是给我们的程序真正执行的。


tools = [

    # --------------------------------------------------------
    # Tool 1：add
    # --------------------------------------------------------
    {
        "type": "function",
        "function": {
            "name": "add",

            # description 非常重要。
            # LLM 会根据这个描述判断什么时候应该使用这个 Tool。
            "description": "计算两个数字的和",

            # 定义 Tool 的参数
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

                # required 表示这两个参数都是必须的
                "required": ["a", "b"]
            }
        }
    },

    # --------------------------------------------------------
    # Tool 2：multiply
    # --------------------------------------------------------
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "计算两个数字的积",

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
# 4. 建立消息历史
# ============================================================

# messages 保存整个对话过程。
#
# 例如：
#
#     user
#       ↓
#     assistant：我要调用 add
#       ↓
#     tool：add 的结果是 30
#       ↓
#     assistant：我要调用 multiply
#       ↓
#     tool：multiply 的结果是 150
#       ↓
#     assistant：最终答案
#
# 这些消息会不断追加到 messages 中，
# 然后每次请求 LLM 时一起发送。


messages = [
    {
        "role": "user",

        # 为了测试 Tool Calling，
        # 我们故意让任务稍微复杂一点。
        #
        # 期望 LLM：
        #
        #     第一步：10 + 20
        #     第二步：结果 × 5
        #
        # 最终：
        #
        #     150
        "content": "先计算 10 + 20，然后把计算结果乘以 5，告诉我最终结果。"
    }
]


# ============================================================
# 5. Tool 名称 → Python 函数的映射
# ============================================================

# 这是 Tool Calling 中非常重要的一步。
#
# LLM 返回的只是字符串：
#
#     "add"
#
# Python 程序必须知道：
#
#     "add" → add()
#
# 同理：
#
#     "multiply" → multiply()
#
# 所以我们建立一个映射表。


tool_functions = {
    "add": add,
    "multiply": multiply
}


# ============================================================
# 6. Tool Calling Loop
# ============================================================

# 这就是今天最重要的代码。
#
# 不再是：
#
#     LLM → Tool → LLM
#
# 而是：
#
#     LLM
#      ↓
#     Tool
#      ↓
#     LLM
#      ↓
#     Tool
#      ↓
#     LLM
#      ↓
#     ...
#
# 直到 LLM 不再要求调用 Tool。
#
# 这就是 Tool Calling Loop。


while True:

    print("\n================ 调用 DeepSeek ================\n")

    # --------------------------------------------------------
    # 6.1 调用 LLM
    # --------------------------------------------------------

    response = client.chat.completions.create(
        model="deepseek-chat",

        # 把完整消息历史交给 LLM
        messages=messages,

        # 把我们提供的 Tool 告诉 LLM
        tools=tools
    )

    # 获取 LLM 返回的 assistant 消息
    message = response.choices[0].message

    # --------------------------------------------------------
    # 6.2 把 assistant 消息加入消息历史
    # --------------------------------------------------------

    # 非常重要：
    #
    # 无论这次 LLM 是：
    #
    #     直接回答
    #
    # 还是：
    #
    #     要调用 Tool
    #
    # 都需要把这条 assistant 消息保存下来。
    #
    # 因为下一次调用 LLM 时，它需要知道之前发生了什么。
    messages.append(message)

    # --------------------------------------------------------
    # 6.3 判断 LLM 是否要求调用 Tool
    # --------------------------------------------------------

    tool_calls = message.tool_calls

    # 如果没有 Tool Call，
    # 说明 LLM 已经可以直接回答用户了。
    #
    # 例如：
    #
    #     tool_calls = None
    #
    # 此时跳出循环。
    if not tool_calls:

        print("\n================ 最终回答 ================\n")

        print(message.content)

        break

    # --------------------------------------------------------
    # 6.4 如果存在 Tool Call，逐个执行
    # --------------------------------------------------------

    # 注意：
    #
    # 一次 LLM 调用可能返回多个 Tool Call。
    #
    # 所以不能再写：
    #
    #     message.tool_calls[0]
    #
    # 而应该遍历：
    #
    #     for tool_call in tool_calls:


    for tool_call in tool_calls:

        # ----------------------------------------------------
        # 6.4.1 获取 Tool 名称
        # ----------------------------------------------------

        tool_name = tool_call.function.name

        print("LLM 请求调用 Tool:", tool_name)

        # ----------------------------------------------------
        # 6.4.2 获取 Tool 参数
        # ----------------------------------------------------

        # DeepSeek 返回的 arguments 是 JSON 字符串。
        #
        # 例如：
        #
        #     '{"a": 10, "b": 20}'
        #
        # json.loads() 会把它转换成 Python 字典：
        #
        #     {
        #         "a": 10,
        #         "b": 20
        #     }

        try:
            arguments = json.loads(
                tool_call.function.arguments
            )

        except json.JSONDecodeError as e:

            # 如果 LLM 返回的 JSON 参数格式不正确，
            # 不应该让整个程序直接崩溃。
            print("Tool 参数 JSON 解析失败:", e)

            # 把错误结果告诉 LLM，
            # 让 LLM 有机会重新处理。
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": f"Tool 参数解析失败：{e}"
                }
            )

            continue

        print("Tool 参数:", arguments)

        # ----------------------------------------------------
        # 6.4.3 根据 Tool 名称找到真正的 Python 函数
        # ----------------------------------------------------

        tool_function = tool_functions.get(tool_name)

        # 如果 LLM 返回了一个我们没有注册的 Tool，
        # 说明出现了异常情况。
        if tool_function is None:

            print("不存在的 Tool:", tool_name)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": f"不存在名为 {tool_name} 的 Tool"
                }
            )

            continue

        # ----------------------------------------------------
        # 6.4.4 真正执行 Python Tool
        # ----------------------------------------------------

        try:

            result = tool_function(**arguments)

            print("Tool 执行结果:", result)

        except Exception as e:

            # Tool 自身执行失败时，
            # 也不要直接让程序崩溃。
            #
            # 把错误信息作为 Tool Result 返回给 LLM。

            result = f"Tool 执行失败：{e}"

            print(result)

        # ----------------------------------------------------
        # 6.4.5 把 Tool 执行结果返回给 LLM
        # ----------------------------------------------------

        # 这是整个 Tool Calling 最关键的一步之一。
        #
        # role = tool
        #
        # 表示：
        #
        #     这是一条 Tool 执行结果。
        #
        # tool_call_id：
        #
        #     告诉 LLM：
        #
        #     "这个结果对应你刚才发出的哪一次 Tool Call"
        #
        # content：
        #
        #     真正的 Tool 执行结果。

        messages.append(
            {
                "role": "tool",

                # 和刚才 LLM 返回的 tool_call.id 对应
                "tool_call_id": tool_call.id,

                # Tool 真正执行出来的结果
                "content": str(result)
            }
        )

# ============================================================
# 程序结束
# ============================================================
