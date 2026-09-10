# rag_1 —— 从 RAG 到 Agent 的学习记录

这是一个**学习性质**的仓库，代码按"一步一个坑"的方式推进，每个 `.py` 文件就是一个学习阶段。
主线是：

```
调用大模型  →  RAG（检索增强生成）  →  Tool Calling（工具调用）  →  Agent Runtime（工具运行时）
```

每个文件都是**独立可运行**的，后一个文件通常在前一个文件的基础上解决一个新问题。
文件里保留了大量中文注释，注释本身就是学习笔记。

---

## 环境准备

```bash
# 依赖（项目使用 uv 管理，Python 3.12）
uv sync

# 必须设置 DeepSeek API Key（所有需要调用 LLM 的脚本都依赖它）
# PowerShell:
$env:DEEPSEEK_API_KEY="你的_API_KEY"
# cmd:
set DEEPSEEK_API_KEY=你的_API_KEY
```

> **国内网络注意**：`ingest.py` / `query.py` 首次运行需要从 HuggingFace 下载
> `BAAI/bge-small-zh-v1.5` 模型，建议先设置镜像，并避免 Windows 下的编码报错：
>
> ```powershell
> $env:HF_ENDPOINT="https://hf-mirror.com"
> $env:PYTHONUTF8="1"
> ```

运行方式：

```bash
uv run python <文件名>.py
```

---

## 学习路径总览

| # | 文件 | 阶段 | 一句话说明 |
|---|------|------|-----------|
| 1 | `test_deepseek.py` | LLM 入门 | 用 OpenAI SDK 调通 DeepSeek，拿到第一个回答 |
| 2 | `ingest.py` | RAG 前半程 | 文档 → 切分 → Embedding → 存进 ChromaDB |
| 3 | `query.py` | RAG 后半程 | 问题 → 向量检索 → 拼 Prompt → LLM 生成答案 |
| 4 | `tool_calling_0.py` | Tool Calling 起步 | 最小闭环：一次工具调用 + 一次结果回传 |
| 5 | `tool_calling.py` | Tool Calling 核心 | `while True` Agent Loop，支持多轮、多工具、异常兜底 |
| 6 | `multi_tool_calling.py` | Tool Calling 进阶 | 一次响应返回多个工具调用，遍历执行 |
| 7 | `async_tool_calling.py` | 异步 | `asyncio.gather` 并行执行多个工具 |
| 8 | `tool_registry.py` | Agent 架构 | 抽出 Tool Registry + Dispatcher，LLM 只认名字 |
| 9 | `tool_validation.py` | Agent 健壮性 | 参数校验：缺参数、类型不对都要拦住 |
| 10 | `tool_runtime.py` | Agent 运行时 | 超时 + 重试 + 异常处理，工具执行不再拖垮整个 Agent |

依赖关系大致是：

```
test_deepseek.py
      │
      ├──────────────► ingest.py ──► query.py        （RAG 线：检索增强生成）
      │
      └──────────────► tool_calling_0.py
                            │
                            ├──► tool_calling.py ──► multi_tool_calling.py
                            │                              │
                            │                              ▼
                            │                     async_tool_calling.py
                            │                              │
                            └──────────────────────────────┴──► tool_registry.py
                                                                      │
                                                                      ▼
                                                            tool_validation.py
                                                                      │
                                                                      ▼
                                                              tool_runtime.py
```

---

## 文件详解

### 1. `test_deepseek.py` —— 第一步：把大模型调通

**学习目标**：HTTP 调用大模型到底长什么样。

- 用 `OpenAI` SDK + `base_url="https://api.deepseek.com"` 访问 DeepSeek（OpenAI 兼容接口）
- API Key 从环境变量 `DEEPSEEK_API_KEY` 读取，不硬编码在代码里
- 模型：`deepseek-v4-flash`
- `extra_body` 透传 DeepSeek 私有参数，关闭 thinking 思考模式，让调用链路最简单
- 最后打印 `response.usage`，建立"Token 是要花钱的"这个意识

> 关键认知：**LLM 调用就是一个 HTTP 请求 + 一个 JSON 响应**，没有任何魔法。

---

### 2. `ingest.py` —— RAG 的"入库"半边

**学习目标**：把私有文档变成可以"按语义搜索"的向量库。

完整流水线（也是 `main()` 的执行顺序）：

```
data/*.txt
   ↓ load_documents()      读取 txt/md，附带来源文件名
   ↓ build_chunks()        切分成 Chunk
   ↓ load_embedding_model() 加载 BAAI/bge-small-zh-v1.5（512 维）
   ↓ create_embeddings()   每个 Chunk → 一个向量
   ↓ save_to_chroma()      向量 + 原文 + metadata 一起存库
   ↓ verify_collection()   抽查 5 条，确认写进去了
```

关键配置：

| 配置 | 值 | 说明 |
|------|-----|------|
| `CHUNK_SIZE` | 200 | 每个片段 200 字符（学习阶段用字符数，生产上一般按 token） |
| `CHUNK_OVERLAP` | 50 | 相邻片段重叠 50 字符，避免知识点被切断 |
| `COLLECTION_NAME` | `java_knowledge` | 必须和 `query.py` 保持一致 |
| `CHROMA_DIR` | `chroma_db/` | `PersistentClient` 持久化，关掉程序数据还在 |

> 关键认知：**RAG 的检索不是关键词匹配，而是向量距离**。
> 切分策略（chunk size / overlap）直接决定检索质量。

数据源：`data/jvm.txt`、`data/mysql.txt`、`data/thread.txt`（Java 知识库）。
注意 `data/` 和 `chroma_db/` 都在 `.gitignore` 里，换机器要重新跑一次 `ingest.py`。

---

### 3. `query.py` —— RAG 的"问答"半边

**学习目标**：检索出来的片段，怎么变成大模型的答案。

流程：

```
用户问题
   ↓ model.encode()          问题也变成向量（必须和入库用同一个模型！）
   ↓ collection.query()      Top-K（默认 3）相似片段
   ↓ build_prompt()          片段 + 来源拼成 Prompt
   ↓ generate_answer()       交给 DeepSeek 生成自然语言回答
```

- `search()` —— 向量检索，返回带 `distances` 的原始结果
- `print_results()` —— 打印 ID / 距离 / metadata / 原文，方便调试检索质量
- `build_prompt()` —— 把片段编号并标注来源（`[知识片段 1]（来源：jvm.txt）`），
  Prompt 里明确要求"知识库中没有就说没有"，抑制幻觉
- `generate_answer()` —— system prompt 约束"必须优先根据知识库内容回答"

主程序是交互式循环，输入 `q` 退出。

> 关键认知：**入库和检索必须用同一个 Embedding 模型**，否则向量空间对不上，检索结果全是噪声。

---

### 4. `tool_calling_0.py` —— 最小的工具调用闭环

**学习目标**：搞懂 Tool Calling 里到底谁在干活。

这是最朴素的版本，一次调用、一个工具、没有循环：

1. 把 `tools`（JSON Schema）和用户问题发给 DeepSeek
2. DeepSeek 返回 `message.tool_calls[0]`，里面只有**工具名 + JSON 字符串参数**，它自己不会执行任何 Python
3. 程序 `json.loads()` 解析参数，手动调用 `multiply(arguments["a"], arguments["b"])`
4. 把结果包成 `{"role": "tool", "tool_call_id": ..., "content": ...}` 追加进 `messages`
5. **再调用一次** DeepSeek，它根据工具结果生成最终回答

> 关键认知：`LLM = 决策者，Python = 执行者`。
> `tool_call_id` 是"这条结果对应哪次调用"的凭证，不带它 LLM 就串不上。
> 注意这里用的是 `tool_calls[0]`，这个假设在后面会被打破。

---

### 5. `tool_calling.py` —— Agent Loop（本仓库的核心文件）

**学习目标**：从"一次问答"升级成"会自己干活的循环"。

和 `_0` 版本的三个升级：

1. **`while True` 循环** —— 不再是一次往返，而是 `LLM → Tool → LLM → Tool → ...`
   直到 LLM 不再要求调用工具（`message.tool_calls` 为空）才 `break`
2. **遍历 `tool_calls`** —— 一次响应可能返回多个工具调用，不能再用 `[0]`
3. **异常兜底** —— 三种失败都转成一条 tool 消息**喂回给 LLM**，而不是让程序崩掉：
   - 参数不是合法 JSON（`json.JSONDecodeError`）
   - 工具名不在 `tool_functions` 映射表里
   - 工具函数执行时抛异常

测试用例故意设计成两步任务：*"先计算 10 + 20，然后把计算结果乘以 5"*，
必须连续调用 `add` → `multiply` 才能得到 150，用来证明循环真的在跑。

> 关键认知：**把错误当成工具结果返回给 LLM，LLM 就能自己纠错**，
> 这是 Agent 比"一次性脚本"更抗造的原因。

---

### 6. `multi_tool_calling.py` —— 一次响应，多个工具

**学习目标**：理解 `message.tool_calls` 是**列表**这件事。

- 用户问题："同时计算 10 + 20，以及 8 × 9。" → 一次响应里出现两个 tool call
- `for tool_call in message.tool_calls:` 逐个执行，`tools_map[tool_name]` 取函数
- 明确写出 `tool_choice="auto"`，让模型自己决定要不要用工具
- 这一版是**串行**执行（for 循环），而且去掉了异常处理，代码更短、更聚焦

> 对比 `tool_calling.py`：那个版本讲"循环和容错"，这个版本讲"一次多个"。
> 但串行执行有个明显问题——两个各要 2 秒的工具，就要等 4 秒，于是有了下一个文件。

---

### 7. `async_tool_calling.py` —— 并行执行工具

**学习目标**：用 `async/await` 把工具执行时间从"求和"变成"取最大值"。

- 客户端换成 `AsyncOpenAI`，调用要 `await client.chat.completions.create(...)`
- 工具函数本身是 `async def`，内部 `await asyncio.sleep(2)` 模拟耗时操作（HTTP 请求、查数据库）
- 抽出 `execute_tool(tool_call)`：**一个 tool_call → 一条 tool 消息**，职责单一
- 关键三行：

  ```python
  tasks = [execute_tool(tc) for tc in message.tool_calls]
  results = await asyncio.gather(*tasks)   # 并发执行，不是循环等待
  messages.extend(results)                 # 结果统一回灌给 LLM
  ```

> 关键认知：`asyncio.gather` 让多个工具**同时**跑。
> 工具是 IO 密集型（网络/数据库）时收益巨大；纯 CPU 计算则不受益（那要用多进程）。

---

### 8. `tool_registry.py` —— 把调度逻辑抽出来

**学习目标**：从"能跑"到"有架构"。

把散落在主循环里的工具处理逻辑收拢成两个概念：

| 概念 | 是什么 | 给谁看 |
|------|--------|--------|
| `TOOL_REGISTRY = {"add": add, ...}` | 工具名 → Python 函数的字典 | **Python** 用 |
| `tools = [{...JSON Schema...}]` | 工具名、描述、参数类型 | **LLM** 用 |

`dispatch_tool(tool_call)` 是"工具调度中心"，固定四步：

```
① 取工具名 → ② json.loads 解析参数 → ③ 从 TOOL_REGISTRY 查函数 → ④ await 执行 → 返回 tool 消息
```

主循环因此变得极简，只剩下"调 LLM / 收 tool_calls / gather / 回灌"：

```python
results = await asyncio.gather(*[dispatch_tool(tc) for tc in message.tool_calls])
```

文件里也明确标注了下一步要补的：**参数校验、异常处理、超时**——就是后面两个文件。

> 关键认知：**Tool Schema ≠ Python 函数**。前者是给 LLM 的说明书，后者是给程序的执行体，
> 两者靠工具名这个"契约"连接。Registry 就是这份契约的落地。

---

### 9. `tool_validation.py` —— 参数校验

**学习目标**：不信任 LLM 传来的参数。

新增 `validate_arguments(tool_name, arguments)`，手写两种检查：

- **存在性**：`if "a" not in arguments: raise ValueError("缺少参数：a")`
- **类型**：`if not isinstance(arguments["a"], (int, float)): raise TypeError(...)`

`dispatch_tool` 变成清晰的四步流水线，每一步失败都返回一条 tool 消息而不是抛出去：

```
JSON 解析 → 查注册表 → 参数校验 → 执行工具
   ↓失败       ↓失败       ↓失败      ↓失败
             全部转成 {"role": "tool", "content": "..."} 交还给 LLM
```

> 关键认知：LLM 生成的参数**经常会错**（漏字段、把数字写成字符串、幻觉出不存在的工具）。
> 校验的意义不是"防坏人"，而是**在错误进入业务代码之前拦住它**，并把原因告诉 LLM 让它重试。
> 这里用的是手工 if/elif，真实项目一般换成 Pydantic 按 JSON Schema 自动校验。

---

### 10. `tool_runtime.py` —— 工具运行时（当前进度）

**学习目标**：工具会慢、会挂，Agent 不能跟着一起挂。

新增 `execute_tool(tool_function, arguments, timeout=3, max_retries=3)`，
把"执行"这件事包装成可以容错的运行时：

- **超时**：`await asyncio.wait_for(tool_function(**arguments), timeout=timeout)`
- **重试**：`for attempt in range(1, max_retries + 1)`，失败后 `await asyncio.sleep(1)` 再试
- **异常分类**：`asyncio.TimeoutError` 单独捕获，其它异常走通用分支
- **最终失败**：重试用尽后 `raise RuntimeError(...)`，由 `dispatch_tool` 转成 tool 消息回给 LLM

为了能观察到超时，专门加了 `slow_tool()` —— 它要睡 10 秒，而超时是 3 秒，
所以会看到"第 1 次 / 第 2 次 / 第 3 次执行 → 超时"的完整日志。

> 关键认知：**重试要有边界**。这里 `timeout(3s) × max_retries(3) ≈ 9 秒+`，
> 如果 Agent 里有 5 个工具并行超时，用户要等多久？超时预算得和整体响应时间一起设计。
> 另外注意 `slow_tool` 没有参数，但 `validate_arguments` 里用
> `if tool_name in ["add", "multiply"]` 做了白名单式的跳过。

---

## 核心概念速查

| 概念 | 出现在 | 一句话 |
|------|--------|--------|
| `tool_call_id` | `tool_calling_0.py` 起 | 把工具结果和对应的那次调用绑定 |
| `role="tool"` 消息 | `tool_calling_0.py` 起 | 工具结果回灌给 LLM 的唯一格式 |
| Agent Loop | `tool_calling.py` | `while True` 直到 LLM 不再要求调用工具 |
| Tool Schema vs Registry | `tool_registry.py` | 给 LLM 看的说明书 vs 给 Python 用的查找表 |
| Dispatcher | `tool_registry.py` | 解析 → 查表 → 校验 → 执行 → 包装结果 |
| 参数校验 | `tool_validation.py` | 在错误进入业务代码前拦住它 |
| 超时 + 重试 | `tool_runtime.py` | 工具会挂，Agent 不能跟着挂 |
| `asyncio.gather` | `async_tool_calling.py` | 多个工具并发执行，耗时取最大值 |
| Chunk / Overlap | `ingest.py` | 检索的最小单位，重叠是为了不切断语义 |
| Top-K | `query.py` | 每次检索喂给 LLM 的知识片段数量 |

---

## 常用命令

```bash
# 1. 建库（改过 data/ 里的文档后要重新跑）
uv run python ingest.py

# 2. 交互式问答（依赖上一步生成的 chroma_db/）
uv run python query.py

# 3. 按学习顺序跑工具调用示例
uv run python tool_calling_0.py
uv run python tool_calling.py
uv run python multi_tool_calling.py
uv run python async_tool_calling.py
uv run python tool_registry.py
uv run python tool_validation.py
uv run python tool_runtime.py
```

---

## 目录结构

```
rag_1/
├── data/               # 原始知识库文档（jvm / mysql / thread），已 gitignore
├── chroma_db/          # ChromaDB 持久化数据，由 ingest.py 生成，已 gitignore
├── src/rag_1/          # uv 项目模板自带，暂无实际内容
├── ingest.py           # ① RAG：建库
├── query.py            # ② RAG：检索 + 生成
├── test_deepseek.py    # ⓪ LLM 调用入门
├── tool_calling_0.py   # ③ 最小工具调用
├── tool_calling.py     # ④ Agent Loop
├── multi_tool_calling.py    # ⑤ 多工具调用
├── async_tool_calling.py    # ⑥ 异步并行
├── tool_registry.py    # ⑦ Registry + Dispatcher
├── tool_validation.py  # ⑧ 参数校验
├── tool_runtime.py     # ⑨ 超时 / 重试（当前进度）
└── pyproject.toml      # 依赖：openai / chromadb / sentence-transformers
```
