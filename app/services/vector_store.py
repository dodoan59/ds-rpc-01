import os
import re
from typing import Dict, List
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


from document_loader import load_data_for_role

def _get_valid_collection_name(category: str) -> str:
    """Convert category name to a valid ChromaDB collection name"""
    valid_name = f"collection_{category}"
    valid_name = re.sub(r'[^a-zA-Z0-9_-]', '_', valid_name)
    return valid_name[:63]

def initialize_vector_stores(
    categories: List[str],
    refinement_splitter: RecursiveCharacterTextSplitter,
    embeddings: HuggingFaceEmbeddings,
) -> Dict[str, Chroma]:
    """
    Initialize and store vector stores for the given categories using a 2-step chunking process.
    1. Split by Markdown headers.
    2. Refine large chunks using the recursive splitter.
    """
    vector_stores: Dict[str, Chroma] = {}
    os.makedirs("resources/vector_store", exist_ok=True)

    headers_to_split_on = [
        ("#", "H1"),
        ("##", "H2"),
        ("###", "H3"),
        ("####", "H4"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False    
    )

    for category in categories:
        try:
            documents_with_type = load_data_for_role(category)
            if not documents_with_type:
                print(f"No documents found for category: {category}")
                continue
            
            all_split_docs = []
            for doc_info in documents_with_type:
                doc_content = doc_info['content']
                doc_type = doc_info['type']

                if doc_type == 'md':
                    initial_chunks = markdown_splitter.split_text(doc_content)
                    for chunk in initial_chunks:
                        # Add category to metadata
                        chunk.metadata['category'] = category
                        
                        is_table = '|' in chunk.page_content and '---' in chunk.page_content

                        if is_table or len(chunk.page_content) <= refinement_splitter._chunk_size:
                            all_split_docs.append(chunk)
                        else:
                            smaller_chunks = refinement_splitter.create_documents(
                                [chunk.page_content], metadatas=[chunk.metadata]
                            )
                            all_split_docs.extend(smaller_chunks)
                
                elif doc_type == 'csv':
                    # Add category to metadata
                    csv_metadata = {'category': category}
                    all_split_docs.append(Document(page_content=doc_content, metadata=csv_metadata))
            
            if not all_split_docs:
                print(f"No valid chunks created for category: {category}")
                continue

            collection_name = _get_valid_collection_name(category)
            persist_path = os.path.join("resources", "vector_store", category)

            try:
                vector_store = Chroma.from_documents(
                    documents=all_split_docs,
                    embedding=embeddings,
                    collection_name=collection_name,
                    persist_directory=persist_path
                )
                print(f"Successfully created vector store for {category} with {len(all_split_docs)} documents")
            except Exception as e:
                print(f"Error creating vector store for {category}: {str(e)}")
                continue
            vector_stores[category] = vector_store
            print(f"Created new vector store for category: {category} with {len(all_split_docs)} chunks")

        except Exception as e:
            print(f"Error initializing vector store for category {category}: {str(e)}")
            
    return vector_stores

if __name__ == "__main__":
    # Initialize the embedding model with explicit parameters
    embeddings_model = HuggingFaceEmbeddings(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        model_kwargs={'device': 'cpu'},  # or 'cuda' if you have GPU
        encode_kwargs={'normalize_embeddings': True}  # Important for e5 models
    )

    # Refinement splitter
    refinement_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    data_categories = ["marketing", "engineering", "hr", "finance", "general"]

    initialized_stores = initialize_vector_stores(
        categories=data_categories,
        refinement_splitter=refinement_splitter, # <--- THAY ĐỔI: Tên biến để rõ nghĩa hơn
        embeddings=embeddings_model,
    )

    if initialized_stores:
        print(f"Successfully initialized and saved {len(initialized_stores)} vector stores.")
        print(f"Data saved at: 'resources/vector_store'")
    else:
        print("Initialization completed, but no vector stores were created. Please check your data source.")