# RAG Subsystem
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import datetime
import os

class SimpleRAG:
    def __init__(self, data_sources=None):
        self.kb = data_sources or []
        self.vectorstore = None
        self._build_index()

    def _build_index(self):
        print("[RAG Subsystem] 正在初始化 HuggingFace Embeddings 与 ChromaDB...")
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        documents = []
        for item in self.kb:
            if isinstance(item, dict):
                documents.append(Document(page_content=item['page_content'], metadata=item.get('metadata', {})))
            else:
                documents.append(Document(page_content=str(item), metadata={"date_int": 20991231}))
                
        # 初始化本地向量数据库
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.vectorstore = Chroma.from_documents(
            documents=documents, 
            embedding=embeddings,
            persist_directory=os.path.join(base_dir, "data", "vector_db", "chroma_db")
        )
        print(f"[RAG Subsystem] 向量知识库构建成功并且持久化到本地, 文档数: {len(self.kb)}")

    def retrieve(self, query: str, target_date: str = None, top_k: int = 3):
        print(f"[RAG Subsystem] 基于检索词 \"{query}\" 进行高维向量检索 (防未来函数 & 30天保鲜期 - 截止日期: {target_date})...")
        if not self.vectorstore:
            return []
            
        filter_dict = None
        if target_date:
            try:
                # 解析目标日期并限定 30天的信息保鲜期（避免由于历史真空把一年前的数据硬扯过来）
                dt_obj = datetime.datetime.strptime(str(target_date)[:10], "%Y-%m-%d")
                end_int = int(dt_obj.strftime("%Y%m%d"))
                
                start_dt = dt_obj - datetime.timedelta(days=30)
                start_int = int(start_dt.strftime("%Y%m%d"))
                
                filter_dict = {
                    "$and": [
                        {"date_int": {"$lte": end_int}},
                        {"date_int": {"$gte": start_int}}
                    ]
                }
            except Exception as e:
                pass
                
        # 使用相似度搜索找寻最相关的研报/新闻段落，并通过 filter 对源数据日期实施切断
        results = self.vectorstore.similarity_search(query, k=top_k, filter=filter_dict)
        return [doc.page_content for doc in results]

