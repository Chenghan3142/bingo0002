import os
from langchain_chroma import Chroma
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except Exception:
        HuggingFaceEmbeddings = None

if HuggingFaceEmbeddings is None:
    print("[view_rag] WARNING: HuggingFaceEmbeddings 未安装，若需启用向量化功能请运行: pip install -U langchain-huggingface")
else:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db_path = os.path.join(base_dir, "data", "vector_db", "chroma_db")

vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)

collections = vectorstore.get()
docs = collections.get('documents', [])
print(f"RAG库中共找到 {len(docs)} 条记录：\n")
for i, doc in enumerate(docs, 1):
    print(f"[{i}] {doc}\n")
