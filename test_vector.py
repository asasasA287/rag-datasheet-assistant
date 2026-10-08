import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ========== 1. 加载 PDF ==========
folder = r"C:\Users\37872\Desktop\样本"
all_docs = []
for filename in os.listdir(folder):
    if filename.lower().endswith(".pdf"):
        filepath = os.path.join(folder, filename)
        loader = PyPDFLoader(filepath)
        docs = loader.load()
        all_docs.extend(docs)

print(f"原始页数：{len(all_docs)}")

# ========== 2. 递归分块 ==========
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""]
)
chunks = splitter.split_documents(all_docs)
print(f"切块总数：{len(chunks)}")

# ========== 3. 加载 Embedding 模型 ==========
print("正在加载嵌入模型，第一次会比较慢...")
embeddings = HuggingFaceEmbeddings(
    model_name="shibing624/text2vec-base-chinese",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
print("嵌入模型加载完成")

# ========== 4. 批量存入 ChromaDB ==========
persist_dir = r"C:\Users\37872\Desktop\chroma_db"

print("正在向量化并存入 ChromaDB，请稍候...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_dir
)
print(f"已存入 {len(chunks)} 个向量块")
print(f"数据库位置：{persist_dir}")