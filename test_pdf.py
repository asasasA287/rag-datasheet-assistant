import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from langchain_community.document_loaders import PyPDFLoader

# PDF 文件夹
folder = r"C:\Users\37872\Desktop\样本"

# 纯文本输出文件夹
output_folder = r"C:\Users\37872\Desktop\样本文本"
os.makedirs(output_folder, exist_ok=True)

# 遍历所有 PDF
for filename in os.listdir(folder):
    if filename.lower().endswith(".pdf"):
        filepath = os.path.join(folder, filename)
        loader = PyPDFLoader(filepath)
        docs = loader.load()

        # 把所有页的纯文本拼在一起
        full_text = ""
        for doc in docs:
            full_text += doc.page_content + "\n\n"

        # 保存成同名 txt
        txt_name = filename.replace(".pdf", ".txt").replace(".PDF", ".txt")
        txt_path = os.path.join(output_folder, txt_name)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        # 打印预览
        print(f"=== {filename} ===")
        print(f"页数: {len(docs)}，字符数: {len(full_text)}")
        print(full_text[:200])   # 只打印前200字，避免刷屏
        print()

print(f"\n所有纯文本已保存到：{output_folder}")