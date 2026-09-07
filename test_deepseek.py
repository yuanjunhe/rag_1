import os

from openai import OpenAI


# 从环境变量读取 API Key
api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise RuntimeError(
        "没有找到 DEEPSEEK_API_KEY 环境变量"
    )


# DeepSeek 提供 OpenAI 兼容接口
client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)


# 调用 DeepSeek V4 Flash
response = client.chat.completions.create(
    model="deepseek-v4-flash",

    messages=[
        {
            "role": "system",
            "content": "你是一个简洁的知识问答助手。"
        },
        {
            "role": "user",
            "content": "什么是 RAG？"
        }
    ],

    # 第一次测试先关闭思考模式，
    # 让调用链路尽可能简单。
    # OpenAI SDK 未内置 thinking 参数，需通过 extra_body 透传给 DeepSeek
    extra_body={
        "thinking": {
            "type": "disabled"
        }
    },

    stream=False
)


# 获取模型最终回答
answer = response.choices[0].message.content

print("========== DeepSeek ==========")
print(answer)

print("\n========== Token 使用情况 ==========")
print(response.usage)