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
CATEGORIES = ["engineering", "hr", "finance", "general", "marketing"]
VECTOR_STORE_DIR = Path("resources/vector_store")


embeddings_model = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)


app = FastAPI()
security = HTTPBasic()

# Initialize RAG service
rag_service = RAGService(categories=CATEGORIES, embedding_model=embeddings_model, persist_base_dir=str(VECTOR_STORE_DIR.absolute()))

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class QueryRequest(BaseModel):
    query: str
    category: Optional[str] = None

class QueryResponse(BaseModel):
    response: str
    sources: List[Dict] = []

# Dummy user database
users_db: Dict[str, Dict[str, str]] = {
    "Tony": {"password": "password123", "role": "engineering"},
    "Bruce": {"password": " ", "role": "marketing"},
    "Sam": {"password": "financepass", "role": "finance"},
    "Peter": {"password": "pete123", "role": "engineering"},
    "Sid": {"password": "sidpass123", "role": "marketing"},
    "Natasha": {"password": "hrpass123", "role": "hr"},
    "Alan": {"password": "ceo123", "role": "c-level"},
    "John": {"password": "johnpass123", "role": "employee"}
}

# Authentication dependency
def authenticate(credentials: HTTPBasicCredentials = Depends(security)) -> Dict[str, str]:
    username = credentials.username
    password = credentials.password
    
    if username not in users_db or users_db[username].get("password") != password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return {"username": username, "role": users_db[username]["role"]}

# Login endpoint
@app.get("/login")
def login(user=Depends(authenticate)):
    return {"message": f"Welcome {user['username']}!", "role": user["role"]}

# Query endpoint
@app.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    user: dict = Depends(authenticate)
):
    try:
        # Get user role
        role = user["role"]
        
        # Get response from RAG service with role-based access
        response = await rag_service.get_rag_response(
            query=request.query,
            role=role,
            category=request.category  # Optional category filter
        )
        
        return QueryResponse(
            response=response,
            sources=[]  # Add sources if available
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

# Get user info endpoint
@app.get("/user/info")
async def get_user_info(user: dict = Depends(authenticate)):
    return {
        "username": user["username"],
        "role": user["role"]
    }