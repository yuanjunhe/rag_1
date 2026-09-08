from openai import OpenAI
import os


client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


def multiply(a, b):
    return a * b


tools = [
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


messages = [
    {
        "role": "user",
        "content": "帮我计算 12 * 30"
    }
]

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    tools=tools
)

message = response.choices[0].message
print(message)


import json

tool_call = message.tool_calls[0]

tool_name = tool_call.function.name
arguments = json.loads(tool_call.function.arguments)

print("工具名称:", tool_name)
print("参数:", arguments)

result = multiply(
    arguments["a"],
    arguments["b"]
)

print("工具执行结果:", result)


# ============================================================
# 3. 把计算结果返回给 DeepSeek
# ============================================================

# --------------------------------------------------------
# 第一步：把 DeepSeek 返回的 assistant 消息
# （里面带 tool_calls）追加进消息历史
# --------------------------------------------------------

messages.append(message)

# --------------------------------------------------------
# 第二步：追加 tool 角色的消息，
# 用 tool_call_id 告诉 DeepSeek 它对应刚才哪个工具调用，
# content 里放工具真实执行的结果
# --------------------------------------------------------

messages.append(
    {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": str(result)
    }
)

# --------------------------------------------------------
# 第三步：带着工具结果再次调用 DeepSeek，
# 让它基于结果生成最终回答
# --------------------------------------------------------

second_response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    tools=tools
)

final_message = second_response.choices[0].message

print("\n========== DeepSeek 回答 ==========")
print(final_message.content)

