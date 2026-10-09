import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma

# 加载 PDF
folder = r"C:\Users\37872\Desktop\样本"
all_docs = []
for filename in os.listdir(folder):
    if filename.lower().endswith(".pdf"):
        filepath = os.path.join(folder, filename)
        loader = PyPDFLoader(filepath)
        docs = loader.load()
        all_docs.extend(docs)

print(f"原始页数：{len(all_docs)}")

# 切块
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""]
)
chunks = splitter.split_documents(all_docs)
print(f"切块总数：{len(chunks)}")

# 用阿里云 Embedding API
print("正在调用 Embedding API...")
embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",
    dashscope_api_key="sk-ws-H.PEDPDYY.f2wj.MEUCIQC1oVpS34RfKn6_CAQtDMVP-NUu_COZDmuTMbizhWK9egIgAs_B__cyL9YPDaoan0rTbZwegC0W8RAeZfONfpHCDms"
)

# 存入 ChromaDB
persist_dir = r"C:\Users\37872\Desktop\新建文件夹\chroma_db"
print("正在向量化并存入 ChromaDB...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_dir
)
print(f"已存入 {len(chunks)} 个向量块")
print(f"数据库位置：{persist_dir}")