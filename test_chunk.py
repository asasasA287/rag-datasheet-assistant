import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

folder = r"C:\Users\37872\Desktop\样本"

# 1. 先加载所有 PDF
all_docs = []
for filename in os.listdir(folder):
    if filename.lower().endswith(".pdf"):
        filepath = os.path.join(folder, filename)
        loader = PyPDFLoader(filepath)
        docs = loader.load()
        all_docs.extend(docs)

print(f"原始文档页数：{len(all_docs)}")

# 2. 创建切块器
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # 每块最多 500 个字符
    chunk_overlap=50,    # 相邻块重叠 50 字符，避免上下文断裂
    separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""]
)

# 3. 切块
chunks = splitter.split_documents(all_docs)

print(f"切块后总数：{len(chunks)}")
print(f"\n=== 第 1 块 ===\n{chunks[0].page_content}")
print(f"\n=== 第 2 块 ===\n{chunks[1].page_content}")