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
    # 保存整个对话历史
    # --------------------------------------------------------
    messages: list = field(default_factory=list)

    # --------------------------------------------------------
    # 当前 Agent 执行步骤
    #
    # 注意：
    # 这里先简单使用。
    # 实际项目中通常会针对每次任务单独计算 step。
    # --------------------------------------------------------
    step: int = 0

    # --------------------------------------------------------
    # 当前任务是否完成
    # --------------------------------------------------------
    finished: bool = False

    # --------------------------------------------------------
    # 历史对话摘要总结
    # --------------------------------------------------------
    summary: str = ""


# ============================================================
# 2. 配置
# ============================================================

MODEL = "deepseek-chat"

# 单次任务最多允许的 Agent Loop 步数
MAX_STEPS = 5

# 消息数量超过该阈值时触发摘要
SUMMARY_TRIGGER = 4

# 摘要时至少保留的最近消息数量（实际会以 user 消息为边界对齐）
KEEP_RECENT = 2


# ============================================================
# 3. DeepSeek Client
# ============================================================

client = AsyncOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


# ============================================================
# 4. 消息读取辅助
#
# state.messages 里混合了两种形态：
#
#   1. dict                        —— 我们自己构造的 user / tool 消息
#   2. ChatCompletionMessage 对象  —— SDK 返回的 assistant 消息
#
# 这里统一封装，后面的代码不需要再到处判断类型。
# ============================================================

def message_role(message) -> str:
    if isinstance(message, dict):
        return message.get("role") or ""
    return getattr(message, "role", "") or ""


def message_content(message) -> str:
    if isinstance(message, dict):
        return message.get("content") or ""
    return getattr(message, "content", None) or ""


# ============================================================
# 5. Summary Memory
# ============================================================

# ------------------------------------------------------------
# 让 LLM 根据「已有摘要 + 新的旧消息」生成一份新的摘要
# ------------------------------------------------------------
async def summarize_messages(messages: list, old_summary: str = "") -> str:

    # --------------------------------------------------------
    # 把消息转换成文本
    #
    # Tool 消息不参与总结；
    # 带 tool_calls 的 assistant 消息 content 为 None，也要跳过，
    # 否则会生成一堆只有角色名的空行。
    # --------------------------------------------------------
    conversation = [
        f"{message_role(message)}: {message_content(message)}"
        for message in messages
        if message_role(message) in ("user", "assistant")
        and message_content(message).strip()
    ]

    # 没有可总结的内容，保持原摘要不变
    if not conversation:
        return old_summary

    conversation_text = "\n".join(conversation)

    prompt = f"""
你是一个对话记忆总结器。

请总结下面的历史对话，只保留未来继续对话时有价值的信息。

重点保留：
1. 用户的个人信息
2. 用户的技术背景
3. 用户的学习目标
4. 用户已经完成的事情
5. 用户的偏好
6. 尚未完成的任务
7. 重要上下文

不要编造信息。

已有摘要：
{old_summary}

新的历史对话：
{conversation_text}

请输出一份简洁的摘要。
"""

    response = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "你负责维护 Agent 的长期对话摘要。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    new_summary = response.choices[0].message.content

    # 模型偶尔会返回空内容，这种情况下保持原摘要
    return new_summary.strip() if new_summary else old_summary


# ------------------------------------------------------------
# 计算安全的切分点
#
# 返回的 index 表示：
#   messages[1:index]  交给 LLM 总结
#   messages[index:]   原样保留
#
# 必须以 user 消息作为边界，原因：
#
#   assistant 的 tool_calls 消息，必须紧跟着对应的 tool 消息，
#   否则下一次请求会因为「tool 消息找不到对应的 tool_calls」而报错。
#
#   如果从中间切开，就会出现：
#     - 保留的窗口以 tool 消息开头
#     - 或者 tool_calls 被总结掉、只剩 tool 结果
#   两种情况都会让对话直接失败。
#
# 向前回溯到最近的 user 消息，就能保证切分点落在完整轮次的边界上。
# ------------------------------------------------------------
def find_summary_split(messages: list) -> int:

    split = max(1, len(messages) - KEEP_RECENT)

    while split > 1 and message_role(messages[split]) != "user":
        split -= 1

    return split


async def update_summary(state: AgentState):

    # --------------------------------------------------------
    # 消息数量没有超过阈值，不需要总结
    # --------------------------------------------------------
    if len(state.messages) <= SUMMARY_TRIGGER:
        return

    # --------------------------------------------------------
    # 第一条是 system message
    # --------------------------------------------------------
    system_message = state.messages[0]

    # --------------------------------------------------------
    # 计算安全切分点
    # --------------------------------------------------------
    split = find_summary_split(state.messages)

    # 没有可以总结的完整轮次，直接放弃本次总结
    if split <= 1:
        return

    # --------------------------------------------------------
    # 需要总结的旧消息 / 需要保留的最近消息
    # --------------------------------------------------------
    old_messages = state.messages[1:split]
    recent_messages = state.messages[split:]

    # --------------------------------------------------------
    # 让 LLM 总结旧消息
    # --------------------------------------------------------
    new_summary = await summarize_messages(
        old_messages,
        state.summary
    )

    # 摘要是空的说明这次总结没有产出，宁可让消息继续堆积，
    # 也不能在没有摘要的情况下丢掉旧消息
    if not new_summary:
        return

    # --------------------------------------------------------
    # 更新 Summary，并删除旧消息
    # 只保留 system + 最近消息
    # --------------------------------------------------------
    state.summary = new_summary

    state.messages = [
        system_message,
        *recent_messages
    ]

    print("\n========== Summary Memory ==========")
    print(state.summary)
    print("====================================\n")


# ------------------------------------------------------------
# 组装真正发给 LLM 的上下文
#
# system prompt + 摘要 + 最近消息
# ------------------------------------------------------------
def build_context(state: AgentState) -> list:

    context = []

    # System Prompt
    if state.messages:
        context.append(state.messages[0])

    # Summary
    if state.summary:
        context.append({
            "role": "system",
            "content": f"以下是之前对话的重要记忆：\n\n{state.summary}"
        })

    # 最近消息
    context.extend(state.messages[1:])

    return context


# ============================================================
# 6. Tools
# ============================================================

def add(a: int, b: int):
    return a + b


def multiply(a: int, b: int):
    return a * b


def subtract(a: int, b: int):
    return a - b


# ============================================================
# 7. Tool Registry
# ============================================================

TOOL_REGISTRY = {
    "add": add,
    "multiply": multiply,
    "subtract": subtract,
    "search_knowledge": search_knowledge,
}


# ============================================================
# 8. Tool Schema
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
# 9. Dispatcher
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
# 10. Agent
#
# 注意：
# Agent 不再创建 State。
#
# State 从外部传入。
# ============================================================

async def agent(state: AgentState):

    # --------------------------------------------------------
    # 每次处理一个用户问题时，
    # 重置本轮状态，重新计算本次 Agent Loop 的步骤。
    # --------------------------------------------------------
    state.step = 0
    state.finished = False

    while state.step < MAX_STEPS:

        state.step += 1

        print(
            f"\n========== Agent Step "
            f"{state.step} =========="
        )

        # ----------------------------------------------------
        # 调用 LLM
        # ----------------------------------------------------

        context = build_context(state)
        response = await client.chat.completions.create(
            model=MODEL,
            messages=context,
            tools=TOOLS,
            tool_choice="auto",
        )

        message = response.choices[0].message

        # ----------------------------------------------------
        # 保存 assistant 消息
        #
        # 无论是否带 tool_calls 都要保存，
        # 否则对话历史里只剩下用户说的话，
        # 模型看不到自己之前的回答，摘要也就无从谈起。
        # ----------------------------------------------------

        state.messages.append(message)

        # ----------------------------------------------------
        # 没有 Tool Call
        #
        # 说明本轮任务完成
        # ----------------------------------------------------

        if not message.tool_calls:

            state.finished = True

            return message.content

        # ----------------------------------------------------
        # 执行 Tool
        # ----------------------------------------------------

        for tool_call in message.tool_calls:

            tool_name = tool_call.function.name

            # 解析失败时不要把整个 Agent 弄崩，
            # 把错误当成 Tool 结果交回给模型，让它自己纠正
            try:

                arguments = json.loads(
                    tool_call.function.arguments
                )

            except json.JSONDecodeError as e:

                result = f"Tool 参数解析失败：{e}"

            else:

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

    return "Agent 执行超过最大步骤限制。"


# ============================================================
# 11. Main
# ============================================================

async def main():

    # ========================================================
    # 创建一个长期存在的 State
    # ========================================================

    state = AgentState()

    # ========================================================
    # 初始化 System Prompt
    # ========================================================

    state.messages.append(
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
        }
    )

    # ========================================================
    # 多轮对话
    # ========================================================

    while True:

        question = input("\n用户：").strip()

        # ----------------------------------------------------
        # 退出
        # ----------------------------------------------------

        if question.lower() == "exit":
            print("Agent 已退出。")
            break

        # ----------------------------------------------------
        # 空输入直接跳过，避免把空消息发给模型
        # ----------------------------------------------------

        if not question:
            continue

        # ----------------------------------------------------
        # 把用户消息加入 State
        # ----------------------------------------------------

        state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        # ----------------------------------------------------
        # 调用 Agent
        #
        # 注意：
        # 传入的是同一个 state。
        # ----------------------------------------------------

        answer = await agent(state)

        print(f"\nAgent：{answer}")

        # 一轮任务完成后检查是否需要总结
        await update_summary(state)


# ============================================================
# 12. 启动
# ============================================================

if __name__ == "__main__":
    asyncio.run(main())
