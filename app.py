import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from openai import OpenAI

# ========== 1. 初始化客户端和向量库（只执行一次） ==========
@st.cache_resource
def load_resources():
@st.cache_resource
def load_resources():
    # 从 Streamlit Secrets 读取 API Key
    api_key = st.secrets["DASHSCOPE_API_KEY"]
    base_url = st.secrets["DASHSCOPE_BASE_URL"]

 client = OpenAI(
    api_key=st.secrets["DASHSCOPE_API_KEY"],
    base_url=st.secrets["DASHSCOPE_BASE_URL"]
)
    # ... 后面的 embeddings 和 vectorstore 不变 ...

    embeddings = HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

   vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)
    )

    return client, vectorstore

client, vectorstore = load_resources()

# ========== 2. 页面标题 ==========
st.set_page_config(page_title="电子元器件规格书助手", page_icon="📘")
st.title("📘 电子元器件规格书助手")
st.caption("上传了 12 份规格书，支持中英文提问")

# ========== 3. 输入框 ==========
question = st.text_input("请输入你的问题：", placeholder="例如：MCP9843 的温度精度是多少？")

# ========== 4. 点击按钮后执行 ==========
if st.button("提问") and question:

    with st.spinner("正在检索相关片段..."):
        docs = vectorstore.similarity_search(question, k=3)

    with st.spinner("正在生成答案..."):
        context_parts = []
        for i, doc in enumerate(docs):
            source = os.path.basename(doc.metadata.get("source", "未知"))
            page = doc.metadata.get("page", "?")
            context_parts.append(f"[片段 {i+1}] 来源：{source}（第 {page} 页）\n{doc.page_content}")

        context = "\n\n".join(context_parts)

        prompt = f"""你是一个电子元器件规格书助手。请根据下面提供的参考资料回答问题。

【回答规则】
1. 只使用参考资料中的信息，不要自己编造。
2. 如果参考资料中没有答案，请直接回答“资料中没有相关信息”。
3. 尽量引用来源文件名和页码。

【参考资料】
{context}

【用户问题】
{question}

【回答】"""

        response = client.chat.completions.create(
            model="qwen-turbo",
            messages=[
                {"role": "system", "content": "你是一个严谨的电子元器件规格书助手。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

    # ========== 5. 显示答案 ==========
    st.subheader("💡 回答")
    st.write(response.choices[0].message.content)

    # ========== 6. 显示来源 ==========
    with st.expander("📎 查看参考片段"):
        for i, doc in enumerate(docs):
            source = os.path.basename(doc.metadata.get("source", "未知"))
            page = doc.metadata.get("page", "?")
            st.markdown(f"**[片段 {i+1}]** {source} 第 {page} 页")
            st.text(doc.page_content[:400])
            st.divider()