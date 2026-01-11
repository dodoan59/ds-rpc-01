from typing import Dict, Optional, List
from fastapi import FastAPI, HTTPException, Depends, status, Request, Body
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from .services.rag import RAGService
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path

# Load environment variables
load_dotenv()
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
CATEGORIES = ["engineer", "hr", "finance", "general", "marketing"]
VECTOR_STORE_DIR = Path("resources/vector_store")


embeddings_model = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)


app = FastAPI()
security = HTTPBasic()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dummy user database
users_db: Dict[str, Dict[str, str]] = {
    "Tony": {"password": "password123", "role": "engineer"},
    "Bruce": {"password": "password123", "role": "marketing"},
    "Sam": {"password": "financepass", "role": "finance"},
    "Peter": {"password": "pete123", "role": "engineer"},
    "Sid": {"password": "sidpass123", "role": "marketing"},
    "Natasha": {"password": "hrpass123", "role": "hr"},
    "Alan": {"password": "ceo123", "role": "c-level"},
    "John": {"password": "johnpass123", "role": "employee"}
}

# Authentication dependency
def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    user = users_db.get(credentials.username)
    if not user or user["password"] != credentials.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return {"username": credentials.username, "role": user["role"]}

# --- 2. INIT RAG SERVICE ---
print("🚀 Init RAG Service...")
rag_service = RAGService()
print("✅ RAG Service ready!")

# --- 3. MODELS ---
class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    user_role: str
    debug_info: Optional[str] = None

# --- 4. ENDPOINT ---
@app.get("/")
def health_check():
    return {"status": "active", "service": "RAG RBAC System"}

@app.get("/login")
def login(user=Depends(authenticate)):
    """API to check user info after login"""
    full_info = users_db.get(user['username'])
    return {
        "message": f"Welcome {user['username']}!",
        "role": user["role"]
    }

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(
    request: QueryRequest,
    user: dict = Depends(authenticate)
):
    """
    Main endpoint to ask questions.
    RBAC logic will be handled inside RAGService based on 'user_role'.
    """
    try:
        user_role = user["role"]
        query_text = request.query
        
        print(f"📡 API Query: '{query_text}' | User: {user['username']} ({user_role})")

        # Call new ainvoke function in rag.py
        # This function will automatically run through Router -> Check Permissions -> Data Agent or Vector Store
        response_text = await rag_service.ainvoke(query=query_text, user_role=user_role)
        
        return QueryResponse(
            answer=str(response_text),
            user_role=user_role,
            debug_info="Processed via Hybrid RAG (Pandas/Vector)"
        )
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"System error: {str(e)}"
        )

