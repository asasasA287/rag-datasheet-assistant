import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from openai import OpenAI

# ===== 把下面引号里替换成你真实的 Key =====
client = OpenAI(
    api_key="sk-ws-H.PEDPDYY.f2wj.MEUCIQC1oVpS34RfKn6_CAQtDMVP-NUu_COZDmuTMbizhWK9egIgAs_B__cyL9YPDaoan0rTbZwegC0W8RAeZfONfpHCDms",
    base_url="https://ws-r4k6ui3r4eerjm2z.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
)

# ... 之前的代码 ...

# 加载向量库
persist_dir = r"C:\Users\37872\Desktop\chroma_db"

print("正在加载向量库...")
# ↓↓↓ 修改这里 ↓↓↓
# 将 model_name 替换为你本地存放模型的文件夹路径
# 注意路径前的 r 不要漏掉
local_model_path = r"D:\models\text2vec-base-chinese"

embeddings = HuggingFaceEmbeddings(
    model_name=local_model_path,  # 使用本地路径
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# ... 之后的代码 ...

vectorstore = Chroma(
    persist_directory=persist_dir,
    embedding_function=embeddings
)
print("向量库加载完成")

question = "MCP9843 的温度精度是多少？"

docs = vectorstore.similarity_search(question, k=4)

print(f"\n检索到 {len(docs)} 个相关块：")
for i, doc in enumerate(docs):
    print(f"\n--- 第 {i+1} 块（来源：{doc.metadata.get('source')} 第 {doc.metadata.get('page')} 页）---")
    print(doc.page_content[:200])

context = "\n\n".join([doc.page_content for doc in docs])
prompt = f"""请根据下面的参考资料回答问题。如果资料里没有答案，就说“资料中没有相关信息”。

参考资料：
{context}

问题：{question}
"""

print("\n正在调用大模型...")
response = client.chat.completions.create(
    model="qwen-turbo",
    messages=[
        {"role": "system", "content": "你是一个电子元器件规格书助手。"},
        {"role": "user", "content": prompt}
    ],
    temperature=0.3
)

print("\n========== 最终回答 ==========")
print(response.choices[0].message.content)