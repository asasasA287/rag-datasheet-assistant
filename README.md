# 电子元器件规格书智能问答系统

基于 RAG（检索增强生成）架构的电子元器件规格书问答系统。

## 🎯 项目背景
工程师查阅 PDF 规格书时，往往需要翻几十页才能找到某个参数，效率低下。
本项目通过 RAG 技术，让用户用自然语言提问，直接从规格书中返回精准答案。

## 🚀 在线演示
[https://rag-datasheet-assistant-mgmt68igr5a8h3tvrmc5le.streamlit.app/](...)

## 🛠️ 技术栈
- **前端**：Streamlit
- **后端**：LangChain
- **向量库**：ChromaDB
- **嵌入模型**：DashScope text-embedding-v3
- **大模型**：通义千问 qwen-turbo

## 📐 系统架构
PDF → PyPDFLoader → RecursiveCharacterTextSplitter → Embedding → ChromaDB
                                                              ↓
用户提问 → Embedding → 检索 Top3 → Prompt → qwen-turbo → 回答

## 📊 项目数据
- 12 份规格书，234 页
- 1123 个向量块

## 💡 技术亮点
- RAG 架构，回答有依据，避免大模型幻觉
- 支持中英文混合检索
- 流式 Prompt 模板设计，引用来源可溯源

## 🔧 本地运行
1. 安装依赖：`pip install -r requirements.txt`
2. 配置 Secrets：见 `.streamlit/secrets.toml`
3. 运行：`streamlit run app.py`
