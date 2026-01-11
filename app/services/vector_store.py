import os
import shutil
import re
from typing import Dict, List
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import pandas as pd
from pathlib import Path
from document_loader import load_data_for_role


SAFE_EMBEDDING_LIMIT = 2000 
CURRENT_FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE_PATH.parents[2]
VECTOR_DB_PATH = PROJECT_ROOT / "resources" / "vector_store"

def create_smart_chunks(documents: List[Document]) -> List[Document]:
    """
    1. Slice by Header (Markdown) to preserve context.
    2. Check table: If the table and size are safe -> Keep as is.
    3. If too long -> Slice into smaller sections using recursion.
    """
    
    # 1. Splitter by Header 
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    # 2. RecursiveCharacter Text Splitter
    refinement_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    final_chunks = []

    for doc in documents:
        # 1. Split by Header Markdown
        # Note: MarkdownHeaderTextSplitter returns documents that already have header metadata
        header_splits = markdown_splitter.split_text(doc.page_content)
        
        for split in header_splits:
            # Merge original metadata (source file) with header metadata
            combined_metadata = {**doc.metadata, **split.metadata}
            content_len = len(split.page_content)
            
            # Check if this chunk contains a table
            is_table = '|' in split.page_content and '---' in split.page_content
            
            # --- DECISION LOGIC (CUSTOM + SAFETY GUARD) ---
            
            # Case A: Chunk is small (plain text or small table) -> Keep as is
            if content_len <= refinement_splitter._chunk_size:
                final_chunks.append(Document(page_content=split.page_content, metadata=combined_metadata))
            
            # Case B: Is a table AND length is within the safe limit of the model -> Keep as is (Don't split tables)
            elif is_table and content_len < SAFE_EMBEDDING_LIMIT:
                final_chunks.append(Document(page_content=split.page_content, metadata=combined_metadata))
                
            # Case C: Text is too long OR Table is too large -> Must split
            else:
                if is_table:
                    print(f"Warning: Table in '{combined_metadata.get('source')}' is too large ({content_len} chars). Splitting it.")
                
                recursive_splits = refinement_splitter.create_documents(
                    [split.page_content], metadatas=[combined_metadata]
                )
                final_chunks.extend(recursive_splits)
                
    return final_chunks

def initialize_vector_stores(
    categories: List[str],
    embeddings: HuggingFaceEmbeddings,
    force_reload: bool = False
) -> Dict[str, Chroma]:
    
    vector_stores = {}

    for category in categories:
        persist_path = os.path.join(VECTOR_DB_PATH, category)
        
        # --- CHECK IF EXISTS (Speed up startup) ---
        if os.path.exists(persist_path) and not force_reload:
            print(f"[{category}] Loading existing Vector Store...")
            try:
                vector_stores[category] = Chroma(
                    persist_directory=persist_path, 
                    embedding_function=embeddings
                )
                continue 
            except Exception as e:
                print(f"Error loading {category}, recreating... ({e})")

        # --- CREATE NEW (If not exists or force_reload=True) ---
        print(f"[{category}] Creating NEW Vector Store...")
        
        # 1. Load
        raw_docs = load_data_for_role(category)
        if not raw_docs:
            print(f"  -> No Markdown files found for {category}. Skipping.")
            continue

        # 2. Split (Smart Chunking)
        chunks = create_smart_chunks(raw_docs)
        print(f"  -> Processed {len(raw_docs)} files into {len(chunks)} chunks.")

        # 3. Embed & Save
        if chunks:
            if os.path.exists(persist_path):
                shutil.rmtree(persist_path)
                
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=persist_path
            )
            vector_stores[category] = vector_store
            print(f"  -> Saved to {persist_path}")

    return vector_stores

if __name__ == "__main__":
    # Configure Embedding Model (This model supports English well, Vietnamese is decent)
    # If specializing in Vietnamese, change to 'bkai-foundation-models/vietnamese-bi-encoder'
    embeddings_model = HuggingFaceEmbeddings(
        model_name="Qwen/Qwen3-Embedding-0.6B", 
        model_kwargs={'device': 'cpu'}, # Change to 'cuda' if you have GPU
        encode_kwargs={'normalize_embeddings': True}
    )

    # List of directories in resources/data/
    data_categories = ["marketing", "general", "finance", "engineering"]

    # force_reload=True: Delete old and recreate from scratch
    # force_reload=False: Use existing (Faster)
    stores = initialize_vector_stores(
        categories=data_categories, 
        embeddings=embeddings_model, 
        force_reload=True 
    )
    
    print("\n-----------------------------------")
    print(f"SYSTEM READY! Loaded {len(stores)} categories.")