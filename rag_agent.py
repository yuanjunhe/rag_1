import os
import json
import asyncio

import chromadb
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. DeepSeek 客户端
# ============================================================

client = AsyncOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


# ============================================================
# 2. Embedding 模型
#
# 注意：
# 查询时必须使用和数据入库时相同的 Embedding 模型。
# ============================================================

embedding_model = SentenceTransformer(
    "BAAI/bge-small-zh-v1.5"
)


# ============================================================
# 3. 连接 ChromaDB
#
# 这里的路径要和你之前 RAG 项目的 ChromaDB 持久化路径一致。
# ============================================================

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)


# ============================================================
# 4. 获取之前创建的知识库 Collection
#
# collection 名称也必须和你之前 ingest.py 使用的一致。
# ============================================================

collection = chroma_client.get_collection(
    name="java_knowledge"
)


# ============================================================
# 5. RAG 核心能力
#
# 现在把原来的 RAG 查询逻辑封装成一个 Tool。
# ============================================================

def search_knowledge(query: str) -> str:
    """
    根据用户问题，从知识库中检索相关内容。

    参数：
        query: 用户想查询的问题

    返回：
        检索到的知识内容
    """

    print(f"\n[RAG] 查询知识库：{query}")

    # --------------------------------------------------------
    # 第一步：把用户问题转换成向量
    # --------------------------------------------------------

    query_embedding = embedding_model.encode(
        query
    ).tolist()

    # --------------------------------------------------------
    # 第二步：从 ChromaDB 检索最相关的 3 个文档
    # --------------------------------------------------------

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    # --------------------------------------------------------
    # 第三步：取出 documents
    # --------------------------------------------------------

    documents = results.get("documents", [[]])[0]

    if not documents:
        return "知识库中没有找到相关内容。"

    # --------------------------------------------------------
    # 第四步：把多个文档整理成字符串
    #
    # 因为最终要把 Tool Result 交给 LLM。
    # --------------------------------------------------------

    knowledge = "\n\n".join(
        f"[知识片段 {i + 1}]\n{doc}"
        for i, doc in enumerate(documents)
    )

    print("[RAG] 检索完成")

    return knowledge


# ============================================================
# 6. Tool Registry
#
# 和你之前学习的 Tool Registry 完全一致。
# ============================================================

TOOL_REGISTRY = {
    "search_knowledge": search_knowledge
}


# ============================================================
# 7. Tool Schema
#
# 告诉 LLM：
#
# 你可以调用 search_knowledge
# 参数叫 query
# query 是字符串
# ============================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": (
                "搜索内部知识库。当用户的问题需要查询知识库中的"
                "技术文档、项目资料或内部知识时使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "需要查询知识库的问题"
                    }
                },
                "required": ["query"]
            }
        }
    }
]


# ============================================================
# 8. Dispatcher
#
# LLM 给我们：
#
# tool_name
# tool_arguments
#
# Dispatcher 负责真正执行 Python 函数。
# ============================================================

def dispatch_tool(tool_name: str, arguments: dict):
    """
    根据 Tool 名称找到对应 Python 函数并执行。
    """

    print(f"\n[Dispatcher] Tool: {tool_name}")
    print(f"[Dispatcher] 参数: {arguments}")

    # --------------------------------------------------------
    # 查找 Tool
    # --------------------------------------------------------

    tool = TOOL_REGISTRY.get(tool_name)

    if tool is None:
        return f"错误：不存在 Tool {tool_name}"

    try:
        # ----------------------------------------------------
        # 执行 Tool
        # ----------------------------------------------------

        result = tool(**arguments)

        print("[Dispatcher] Tool 执行完成")

        return result

    except Exception as e:
        return f"Tool 执行失败：{str(e)}"


# ============================================================
# 9. Agent
# ============================================================

async def agent(user_question: str):

    # --------------------------------------------------------
    # System Prompt
    #
    # 告诉 LLM：
    #
    # 1. 你是一个知识库助手
    # 2. 如果需要知识库，就调用 search_knowledge
    # 3. 获取结果后再回答用户
    # --------------------------------------------------------

    messages = [
        {
            "role": "system",
            "content": """
你是一个知识库问答助手。

当用户的问题需要查询知识库时，
请调用 search_knowledge。

获得知识库结果后，
请严格根据知识库内容回答。

如果知识库中没有相关信息，
请明确告诉用户没有找到相关知识，
不要编造答案。
"""
        },
        {
            "role": "user",
            "content": user_question
        }
    ]

    # --------------------------------------------------------
    # Agent Loop
    #
    # 这里就是你之前已经学习过的 Tool Calling Loop。
    # --------------------------------------------------------

    while True:

        print("\n================ 调用 LLM ================")

        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )

        message = response.choices[0].message

        # ----------------------------------------------------
        # 情况一：
        # LLM 不需要 Tool
        #
        # 直接返回最终答案
        # ----------------------------------------------------

        if not message.tool_calls:

            return message.content

        # ----------------------------------------------------
        # 情况二：
        # LLM 决定调用 Tool
        # ----------------------------------------------------

        # 必须把 assistant 的 tool_calls 加入 messages
        messages.append(message)

        # ----------------------------------------------------
        # 处理 LLM 请求的 Tool
        # ----------------------------------------------------

        for tool_call in message.tool_calls:

            tool_name = tool_call.function.name

            arguments = json.loads(
                tool_call.function.arguments
            )

            # ------------------------------------------------
            # Dispatcher 执行 Tool
            # ------------------------------------------------

            result = dispatch_tool(
                tool_name,
                arguments
            )

            # ------------------------------------------------
            # 把 Tool Result 返回给 LLM
            # ------------------------------------------------

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                }
            )


# ============================================================
# 10. 主程序
# ============================================================

async def main():

    question = input("请输入问题：")

    answer = await agent(question)

    print("\n================ 最终答案 ================")
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())