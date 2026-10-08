import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ========== 1. 加载已有的向量库 ==========
persist_dir = r"C:\Users\37872\Desktop\chroma_db"

print("正在加载向量库...")
embeddings = HuggingFaceEmbeddings(
    model_name="shibing624/text2vec-base-chinese",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

vectorstore = Chroma(
    persist_directory=persist_dir,
    embedding_function=embeddings
)
print("向量库加载完成\n")

# ========== 2. 输入问题 ==========
question = "MCP9843 的温度精度是多少？"
print(f"问题：{question}\n")

# ========== 3. 语义检索 Top3 ==========
docs = vectorstore.similarity_search(question, k=10)

# ========== 4. 打印结果 ==========
print(f"返回 Top{len(docs)} 相关片段：\n")
for i, doc in enumerate(docs):
    source = doc.metadata.get("source", "未知")
    page = doc.metadata.get("page", "?")
    print(f"========== 第 {i+1} 名 ==========")
    print(f"来源：{source}")
    print(f"页码：第 {page} 页")
    print(f"内容：{doc.page_content[:300]}")
    print()