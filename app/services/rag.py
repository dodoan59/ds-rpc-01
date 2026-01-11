import os
import pandas as pd
from typing import List, Dict
from pathlib import Path
from dotenv import load_dotenv

# Core LangChain
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Reranker & Processing
from sentence_transformers import CrossEncoder

load_dotenv()

# --- CONFIGURATION ---
CURRENT_FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE_PATH.parents[2]
VECTOR_STORE_PATH = PROJECT_ROOT / "resources" / "vector_store"
CSV_DATA_PATH = PROJECT_ROOT / "resources" / "data" / "hr" / "hr_data.csv"

RBAC_CONFIG = {
    "finance": {"allowed_docs": ["finance", "general"], "data_access": "restricted"},
    "hr": {"allowed_docs": ["hr", "general"], "data_access": "all"},
    "engineer": {"allowed_docs": ["engineering", "general"], "data_access": "restricted"},
    "marketing": {"allowed_docs": ["marketing", "general"], "data_access": "restricted"},
    "c-level": {"allowed_docs": ["*"], "data_access": "all"},
    "employee": {"allowed_docs": ["general"], "data_access": "restricted"},
    "default": {"allowed_docs": ["general"], "data_access": "restricted"}
}

class RAGService:
    def __init__(self):
        # 1. Models
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        
        # Embedding model (Bi-Encoder)
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="Qwen/Qwen3-Embedding-0.6B",
            model_kwargs={'device': 'cpu'}
        )
        
        # Reranker model (Cross-Encoder)
        self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cpu")

        # 2. Resources
        self.vector_dbs = self._load_all_vector_stores()
        self.raw_df = pd.read_csv(CSV_DATA_PATH) if CSV_DATA_PATH.exists() else None
        self.router_chain = self._create_router_chain()

    def _load_all_vector_stores(self) -> Dict[str, Chroma]:
        stores = {}
        if VECTOR_STORE_PATH.exists():
            for d in VECTOR_STORE_PATH.iterdir():
                if d.is_dir():
                    stores[d.name] = Chroma(
                        persist_directory=str(d),
                        embedding_function=self.embedding_model
                    )
        return stores

    def _create_router_chain(self):
        template = """Classify the user's question into one of two groups:
        1. "DATA": If the question contains terms related to **structured data analysis** (e.g., "average", "sum", "total", "count", "how many", "filter", "greater than", "less than", "top 5", "group by", "details of employee" etc.)
        2. "DOCUMENT": If the question is more about general understanding, summarization, definitions, or cannot be answered from structured tabular data, classify or If question is about summary of a document, process etc
        
        Only return one word: DATA or DOCUMENT.
        Question: {question}"""
        return ChatPromptTemplate.from_template(template) | self.llm | StrOutputParser()

    async def _rerank_docs(self, query: str, docs: List, top_n: int = 5):
        """Use Cross-Encoder to re-rank documents from Vector Search"""
        if not docs: return []
        
        # Prepare (Query, Document) pairs for Cross-Encoder
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.reranker.predict(pairs)
        
        # Assign scores and sort
        for i, doc in enumerate(docs):
            doc.metadata["rerank_score"] = float(scores[i])
        
        docs.sort(key=lambda x: x.metadata["rerank_score"], reverse=True)
        return docs[:top_n]

    async def _expand_parent_scope(self, db: Chroma, docs: List):
        """Expand context based on Metadata Header"""
        expanded = []
        seen_contents = set()
        
        for doc in docs:
            h1 = doc.metadata.get("H1")
            if h1:
                # Find related segments with the same H1 header
                related = db.similarity_search(f"header: {h1}", k=3)
                for r in related:
                    if r.page_content not in seen_contents:
                        expanded.append(r)
                        seen_contents.add(r.page_content)
            else:
                if doc.page_content not in seen_contents:
                    expanded.append(doc)
                    seen_contents.add(doc.page_content)
        return expanded

    async def ainvoke(self, query: str, user_role: str):
        user_role = user_role.lower()
        config = RBAC_CONFIG.get(user_role, RBAC_CONFIG["default"])
        
        # --- STEP 1: ROUTING ---
        try:
            route = (await self.router_chain.ainvoke({"question": query})).strip().upper()
        except: route = "DOCUMENT"

        # --- STEP 2: DATA (CSV) ---
        if "DATA" in route:
            if config["data_access"] != "all":
                return "⛔ Bạn không có quyền truy cập dữ liệu nhân sự nhạy cảm."
            
            agent = create_pandas_dataframe_agent(
                self.llm, self.raw_df, verbose=False, allow_dangerous_code=True
            )
            try:
                result = await agent.ainvoke(query)
                return result["output"]
            except Exception as e:
                return f"Lỗi xử lý dữ liệu: {str(e)}"

        # --- STEP 3: DOCUMENT ---
        else:
            allowed_cats = config['allowed_docs']
            if "*" in allowed_cats:
                target_dbs = list(self.vector_dbs.values())
            else:
                target_dbs = [self.vector_dbs[c] for c in allowed_cats if c in self.vector_dbs]
            
            if not target_dbs: return "Không tìm thấy tài liệu phù hợp."

            # 3.1. Retrieval with MMR (take 15 candidates)
            all_candidates = []
            for db in target_dbs:
                retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 15, "fetch_k": 30})
                docs = await retriever.ainvoke(query)
                all_candidates.extend(docs)

            # 3.2. Cross-Encoder Reranking (take 5 best candidates)
            relevant_docs = await self._rerank_docs(query, all_candidates, top_n=5)

            # 3.3. Parent Scope Expansion (for long technical/process documents)
            if user_role in ["engineer", "c-level"]:
                final_context_docs = await self._expand_parent_scope(target_dbs[0], relevant_docs)
            else:
                final_context_docs = relevant_docs

            # 3.4. Final Generation with Gemini
            context_text = ""
            for i, d in enumerate(final_context_docs):
                source = d.metadata.get("source", "Internal Document")
                context_text += f"\n[Segment {i+1} - Source: {source}]\n{d.page_content}\n"
            
            prompt = ChatPromptTemplate.from_template("""
            You are an internal chatbot with role-based access control.
            Use the provided documents to answer the user's query.
            Ensure that data access is restricted based on the user's role.
            Generate a clear and insight response.
            Only return the result in table format if the document content is a table. 
            If the document content is not a table, avoid using a table format. 
            If the user's query is not related to their role's documents, 
            respond with: 'I'm sorry, but I can't assist with that.'

            CONTEXT:
            {context}

            QUESTION: {question}
            """)
            
            chain = prompt | self.llm | StrOutputParser()
            return await chain.ainvoke({"context": context_text, "question": query})