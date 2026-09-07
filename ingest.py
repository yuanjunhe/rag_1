"""
RAG 知识库构建程序

完整流程：

文档
  ↓
读取文本
  ↓
文本切分 Chunk
  ↓
Embedding
  ↓
向量
  ↓
存入 ChromaDB
"""

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. 基础配置
# ============================================================

# 当前项目目录
BASE_DIR = Path(__file__).parent

# 原始文档目录
DATA_DIR = BASE_DIR / "data"

# ChromaDB 持久化目录
CHROMA_DIR = BASE_DIR / "chroma_db"

# ChromaDB Collection 名称
COLLECTION_NAME = "java_knowledge"

# Chunk 大小
# 这里为了方便学习，暂时使用字符数作为单位
CHUNK_SIZE = 200

# 相邻 Chunk 重叠的字符数
# 例如：
#
# Chunk 1：0   ~ 200
# Chunk 2：150 ~ 350
#
# 中间 50 个字符是重叠的
CHUNK_OVERLAP = 50

# 每次查询时使用的 Embedding 模型
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"


# ============================================================
# 2. 读取文档
# ============================================================

def load_documents():
    """
    从 data 目录读取所有 txt 文件。

    返回：

    [
        {
            "text": "...",
            "source": "java.txt"
        },
        ...
    ]
    """

    documents = []

    # 查找 data 目录下所有 .txt / .md 文件
    files = sorted(
        list(DATA_DIR.glob("*.txt"))
        + list(DATA_DIR.glob("*.md"))
    )

    if not files:
        raise RuntimeError(
            f"没有找到 txt / md 文档，请检查目录：{DATA_DIR}"
        )

    for file_path in files:

        print(f"正在读取：{file_path.name}")

        text = file_path.read_text(encoding="utf-8")

        # 去掉首尾空白
        text = text.strip()

        if not text:
            print(f"跳过空文件：{file_path.name}")
            continue

        documents.append({
            "text": text,
            "source": file_path.name
        })

    print(f"\n共读取 {len(documents)} 个文档")

    return documents


# ============================================================
# 3. 文本切分
# ============================================================

def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    将一篇长文本切分成多个 Chunk。

    例如：

    原始文本：
    ABCDEFGHIJKLMNOP...

    chunk_size = 10
    overlap = 3

    Chunk 1：
    ABCDEFGHIJ

    Chunk 2：
    HIJKLMNOPQ

    相邻 Chunk 会有部分重叠。

    这样可以避免一个完整的知识点刚好
    被切割到两个 Chunk 中。
    """

    if overlap >= chunk_size:
        raise ValueError(
            "CHUNK_OVERLAP 必须小于 CHUNK_SIZE"
        )

    chunks = []

    start = 0

    while start < len(text):

        # 当前 Chunk 的结束位置
        end = start + chunk_size

        # 截取文本
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        # 下一个 Chunk 的开始位置
        #
        # 例如：
        # chunk_size = 200
        # overlap = 50
        #
        # 第一个：0 ~ 200
        # 第二个：150 ~ 350
        #
        start += chunk_size - overlap

    return chunks


# ============================================================
# 4. 加载 Embedding 模型
# ============================================================

def load_embedding_model():
    """
    加载 Embedding 模型。

    这里使用：

    BAAI/bge-small-zh-v1.5

    该模型生成 512 维向量。
    """

    print("\n正在加载 Embedding 模型：")
    print(EMBEDDING_MODEL_NAME)

    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    dimension = model.get_embedding_dimension()

    print(f"Embedding 维度：{dimension}")

    return model


# ============================================================
# 5. 创建 ChromaDB
# ============================================================

def create_collection():
    """
    创建一个持久化的 ChromaDB Collection。

    数据会保存到：

    chroma_db/

    程序结束后数据不会丢失。
    """

    print("\n正在初始化 ChromaDB...")

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    print(f"Collection：{COLLECTION_NAME}")
    print(f"当前数据量：{collection.count()}")

    return collection


# ============================================================
# 6. 构建 Chunk
# ============================================================

def build_chunks(documents):
    """
    将所有文档切分成 Chunk。

    最终得到：

    [
        {
            "text": "...",
            "source": "java.txt",
            "chunk": 0
        },
        {
            "text": "...",
            "source": "java.txt",
            "chunk": 1
        }
    ]
    """

    all_chunks = []

    for document in documents:

        text = document["text"]
        source = document["source"]

        chunks = split_text(text)

        print(
            f"{source} -> {len(chunks)} 个 Chunk"
        )

        for index, chunk in enumerate(chunks):

            all_chunks.append({
                "text": chunk,
                "source": source,
                "chunk": index
            })

    print(f"\n总 Chunk 数量：{len(all_chunks)}")

    return all_chunks


# ============================================================
# 7. Embedding
# ============================================================

def create_embeddings(model, chunks):
    """
    将所有 Chunk 转换成向量。

    例如：

    Chunk
      ↓
    Embedding Model
      ↓
    [0.12, -0.03, ..., 0.56]

    每个 Chunk 对应一个向量。
    """

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print("\n正在生成 Embedding...")

    embeddings = model.encode(
        texts,
        show_progress_bar=True
    )

    print(f"Embedding 数量：{len(embeddings)}")
    print(f"Embedding Shape：{embeddings.shape}")

    return embeddings


# ============================================================
# 8. 保存到 ChromaDB
# ============================================================

def save_to_chroma(collection, chunks, embeddings):
    """
    将：

    Chunk
    Embedding
    Metadata

    一起保存到 ChromaDB。
    """

    ids = []
    documents = []
    metadatas = []

    for index, chunk in enumerate(chunks):

        # 每一条数据必须有唯一 ID
        ids.append(
            f"{chunk['source']}_{chunk['chunk']}_{index}"
        )

        # 原始文本
        documents.append(
            chunk["text"]
        )

        # Metadata
        metadatas.append({
            "source": chunk["source"],
            "chunk": chunk["chunk"]
        })

    print("\n正在保存到 ChromaDB...")

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    print(
        f"保存完成，当前数据库数据量："
        f"{collection.count()}"
    )


# ============================================================
# 9. 查看数据库内容
# ============================================================

def verify_collection(collection):
    """
    简单验证 ChromaDB 中的数据。
    """

    print("\n========== 数据库验证 ==========")

    result = collection.get(
        limit=5
    )

    for i in range(len(result["ids"])):

        print(f"\n--- 数据 {i + 1} ---")

        print("ID：")
        print(result["ids"][i])

        print("文本：")
        print(result["documents"][i])

        print("Metadata：")
        print(result["metadatas"][i])


# ============================================================
# 10. 主程序
# ============================================================

def main():

    print("================================")
    print("       RAG 知识库构建程序")
    print("================================")

    # --------------------------------------------------------
    # 第一步：读取原始文档
    # --------------------------------------------------------

    documents = load_documents()

    # --------------------------------------------------------
    # 第二步：文本切分
    # --------------------------------------------------------

    chunks = build_chunks(documents)

    # --------------------------------------------------------
    # 第三步：加载 Embedding 模型
    # --------------------------------------------------------

    model = load_embedding_model()

    # --------------------------------------------------------
    # 第四步：生成 Embedding
    # --------------------------------------------------------

    embeddings = create_embeddings(
        model,
        chunks
    )

    # --------------------------------------------------------
    # 第五步：创建 ChromaDB
    # --------------------------------------------------------

    collection = create_collection()

    # --------------------------------------------------------
    # 第六步：保存到 ChromaDB
    # --------------------------------------------------------

    save_to_chroma(
        collection,
        chunks,
        embeddings
    )

    # --------------------------------------------------------
    # 第七步：验证
    # --------------------------------------------------------

    verify_collection(collection)

    print("\n================================")
    print("       RAG 知识库构建完成")
    print("================================")


# Python 程序入口
if __name__ == "__main__":
    main()
