
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"        # ← 加这行
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # ← 加这行
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from openai import OpenAI

# ========== 1. 初始化大模型客户端 ==========
client = OpenAI(
    api_key="sk-ws-H.PEDPDYY.f2wj.MEUCIQC1oVpS34RfKn6_CAQtDMVP-NUu_COZDmuTMbizhWK9egIgAs_B__cyL9YPDaoan0rTbZwegC0W8RAeZfONfpHCDms",   # 换成你自己的
    base_url="https://ws-r4k6ui3r4eerjm2z.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
)

# ========== 2. 加载向量库 ==========
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

# ========== 3. 用户问题 ==========
question = "MCP9843 的温度精度是多少？"

# ========== 4. 检索 Top3 ==========
docs = vectorstore.similarity_search(question, k=3)

# ========== 5. 拼接参考资料 ==========
context_parts = []
for i, doc in enumerate(docs):
    source = os.path.basename(doc.metadata.get("source", "未知"))
    page = doc.metadata.get("page", "?")
    context_parts.append(
        f"[片段 {i+1}] 来源：{source}（第 {page} 页）\n{doc.page_content}"
    )

context = "\n\n".join(context_parts)

# ========== 6. 拼接 Prompt 模板 ==========
prompt = f"""你是一个电子元器件规格书助手。请根据下面提供的参考资料回答用户的问题。

【回答规则】
1. 只使用参考资料中的信息，不要自己编造。
2. 如果参考资料中没有答案，请直接回答“资料中没有相关信息”。
3. 尽量引用来源文件名和页码。

【参考资料】
{context}

【用户问题】
{question}

【回答】"""

# ========== 7. 打印最终 Prompt（可选，调试用） ==========
print("========== 最终 Prompt（预览） ==========")
print(prompt[:600] + "...\n")

# ========== 8. 调用大模型 ==========
print("正在调用大模型...\n")
response = client.chat.completions.create(
    model="qwen-turbo",
    messages=[
        {"role": "system", "content": "你是一个严谨的电子元器件规格书助手。"},
        {"role": "user", "content": prompt}
    ],
    temperature=0.2
)

# ========== 9. 输出答案 ==========
print("========== 最终回答 ==========")
print(response.choices[0].message.content)