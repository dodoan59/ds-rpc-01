import os
import re
from typing import Dict, List
from dotenv import load_dotenv
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import asyncio

# Dowload environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

class RAGService:
    def __init__(self, categories: List[str], embedding_model: HuggingFaceEmbeddings = None, persist_base_dir: str = "resources/vector_store"):
        """
        Initialize RAGService by loading vector stores from disk.

        Args:
            categories: List of category names to load.
            embedding_model: Embedding model used to create vector stores.
            persist_base_dir: Base directory where vector stores of categories are stored.
        """
        
        db_path = os.path.abspath("./resources/data/hr/hr.db")
        database_uri = f"sqlite:///{db_path}"
        self.db = SQLDatabase.from_uri(database_uri)

        self.llm1 = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            max_tokens=1000,
        )

        # Create SQL toolkit
        toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm1)
        
        # Initialize SQL agent
        self.sql_agent = create_sql_agent(
            llm=self.llm1,
            toolkit=toolkit,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        )

        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.2,
            max_tokens=10000,
        )
        
        # Load vector stores from disk
        # Initialize embedding model if not provided
        if embedding_model is None:
            embedding_model = HuggingFaceEmbeddings(
                model_name="Qwen/Qwen3-Embedding-0.6B",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
        self.vector_stores = self._load_vector_stores(categories, embedding_model, persist_base_dir)
        
        # Define system prompt for chatbot
        self.system_prompt = (
            "You are an internal chatbot with role-based access control."
            "Use the provided documents to answer the user's query."
            "Ensure that data access is restricted based on the user's role."
            "Generate a clear and insight response."
            "Only return the result in table format if the document content is a table. "
            "If the document content is not a table, avoid using a table format. "
            "If the user's query is not related to their role's documents,"
            "respond with: 'I'm sorry, but I can't assist with that.'"
            "\n\n"
            "{context}"
        )
        
        # Initialize prompt template
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "{input}"),
            ]
        )

    def _load_vector_stores(
        self, categories: List[str], embeddings: HuggingFaceEmbeddings, persist_base_dir: str
    ) -> Dict[str, Chroma]:
        """Tải các kho lưu trữ vector Chroma đã được lưu trữ từ đĩa."""
        loaded_stores = {}
        print("Loading vector stores...")
        for category in categories:
            persist_path = os.path.join(persist_base_dir, category)
            if os.path.exists(persist_path):
                try:
                    print(f"Loading vector store for category: '{category}' from '{persist_path}'")
                    # Create collection name
                    collection_name = f"collection_{category}"
                    collection_name = re.sub(r'[^a-zA-Z0-9_-]', '_', collection_name)[:63]
                    
                    vector_store = Chroma(
                        persist_directory=persist_path,
                        embedding_function=embeddings,
                        collection_name=collection_name
                    )
                    
                    # Verify collection has been loaded correctly
                    try:
                        collection = vector_store._collection.get()
                        doc_count = len(collection.get('ids', [])) if collection else 0
                        print(f"  - Loaded {doc_count} documents for category '{category}'")
                    except Exception as e:
                        print(f"  - Warning: Could not verify document count for '{category}': {e}")
                    
                    loaded_stores[category] = vector_store
                except Exception as e:
                    print(f"Error loading vector store for category {category}: {e}")
            else:
                print(f"Warning: Persist directory not found for category '{category}' at '{persist_path}'. Skipping.")
        
        if not loaded_stores:
             print("Warning: No vector stores were loaded. RAG service may not function correctly.")
        else:
             print(f"Successfully loaded {len(loaded_stores)} vector stores.")
        return loaded_stores
    
    def _get_accessible_categories(self, role: str) -> List[str]:
        """Determine list of categories that a user with a specific role can access."""
        role = role.lower()
        
        # Define mapping from role to accessible categories
        role_access = {
            "finance": ["finance", "general"],
            "marketing": ["marketing", "general"],
            "engineering": ["engineering", "general"],
            "hr": ["hr", "general"],
            "c-level": ["finance", "marketing", "engineering", "hr", "general"],
            "employee": ["general"]
        }
        
        # Default to 'employee' access if role not found
        return role_access.get(role, ["general"])
    
    def _get_combined_retriever(self, categories: List[str]):
        """Create a retriever that can search across multiple specified categories."""
        if not categories:
            return None
            
        # Get vector stores for categories that user has access to
        retrievers = [
            self.vector_stores[cat].as_retriever(
                search_type="mmr",
                search_kwargs={"k": 15, "fetch_k": 30}
            )
            for cat in categories
            if cat in self.vector_stores
        ]

        if not retrievers:
            raise ValueError("No valid vector stores found for the specified categories.")

        if len(retrievers) > 1:
            # Assign equal weights to each retriever automatically
            weights = [1.0 / len(retrievers)] * len(retrievers)
            ensemble_retriever = EnsembleRetriever(retrievers=retrievers, weights=weights)
        else:
            ensemble_retriever = retrievers[0]
        cross_encoder_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-large")
        
        # Create a reranker compressor
        compressor = CrossEncoderReranker(model=cross_encoder_model, top_n=10)
        
        # Create ContextualCompressionRetriever
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=ensemble_retriever
        )
        
        return compression_retriever
    
    async def get_rag_response(self, query: str, role: str, category: str = None) -> str:
        """
        Create RAG response for a query based on user role and optional category filter.
        """
        # Determine categories that user can access based on their role
        accessible_categories = self._get_accessible_categories(role.lower())
        
        # If a specific category is requested, validate access
        if category and category.lower() != 'all':
            if category.lower() not in [c.lower() for c in accessible_categories]:
                return f"Access to category '{category}' is not permitted for your role."
            categories_to_search = [category.lower()]
        else:
            categories_to_search = accessible_categories
        try:
            retriever = self._get_combined_retriever(categories_to_search)
        except ValueError as e:
            return str(e)

        print(f"User role: {role}")
        print(f"Accessible categories: {accessible_categories}")
        print(f"Categories to search: {categories_to_search}")
    
        if not retriever:
            return "No relevant information was found based on your access level."
    
        # Step 1: Retrieve and rerank documents
        initial_docs = retriever.invoke(query)
        
        if not initial_docs:
            return "Could not find any relevant documents."

        # Step 2: Apply parent scope logic if necessary
        final_docs = initial_docs
        apply_parent_scope = False

        # Check if the first document has metadata and belongs to a structured category
        if initial_docs[0].metadata and 'category' in initial_docs[0].metadata:
            top_doc_category = initial_docs[0].metadata['category']
            if top_doc_category in ['engineering', 'general']:
                apply_parent_scope = True
                print(f"\n--- DEBUG: Top document is from '{top_doc_category}'. Applying parent scope logic. ---")

        if apply_parent_scope:
            # Step 2: Expand context flexibly
            anchor_meta = initial_docs[0].metadata
            top_doc_category = initial_docs[0].metadata['category'] # Lấy lại category để lọc
            
            parent_scope = {}
            # Determine parent scope based on metadata
            for h_level in ["H1", "H2", "H3"]: 
                if h_level in anchor_meta and anchor_meta[h_level]:
                    parent_scope[h_level] = anchor_meta[h_level]

            print(f"Determined Parent Scope: {parent_scope}")

            # Filter docs based on parent scope and category
            scoped_docs = []
            if parent_scope:
                for doc in initial_docs:
                    # Only process documents from the same category with structure
                    if doc.metadata.get('category') == top_doc_category:
                        is_in_scope = True
                        for h_level, h_value in parent_scope.items():
                            if doc.metadata.get(h_level) != h_value:
                                is_in_scope = False
                                break
                        if is_in_scope:
                            scoped_docs.append(doc)
                final_docs = scoped_docs
            else:
                # If no scope determined, use only the first doc to ensure relevance
                final_docs = [initial_docs[0]]

            # Sort again to ensure logical order
            def sort_key(doc):
                return tuple(doc.metadata.get(h, '') for h in ["H1", "H2", "H3", "H4"])
            final_docs.sort(key=sort_key)
        else:
            print("\n--- DEBUG: Top document not from a structured category or metadata missing. Skipping parent scope logic. ---")

        print(f"\n--- DEBUG: FINAL DOCUMENTS FOR CONTEXT ({len(final_docs)} docs) ---")
        context_parts = []
        for i, doc in enumerate(final_docs):
            print(f"\n[Document {i+1}]")
            print("Metadata:", doc.metadata)
            print("Content:", doc.page_content.strip()) # In ra nội dung
            context_parts.append(doc.page_content)
            print("---------------------------------------------------\n")

        context = "\n\n".join(context_parts)
        
        prompt_input = {
            "context": context,
            "input": query
        }
        
        chain = self.prompt | self.llm | StrOutputParser()
        response = await chain.ainvoke(prompt_input)

        # Check if user is eligible for SQL fallback
        eligible_for_sql_fallback = role.lower() in ['hr', 'c-level']

        if eligible_for_sql_fallback and self.sql_agent:
            print(f"Role '{role}' is eligible. Falling back to SQL Agent.")
            try:    
                # Call executor of SQL agent
                sql_response = await self.sql_agent.ainvoke(query)
                sql_response = sql_response.get("output")
                if sql_response == "I don't know":
                    return response
                return sql_response

            except Exception as e:
                # If there is an error during the conversion, return error message
                return f"An error occurred during the SQL fallback attempt: {str(e)}"
        return response

# async def test_rag_service():
#     """
#     Test RAG service with error handling and improved logging.
#     """
#     from pathlib import Path
    
#     # Configuration
#     MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
#     CATEGORIES = ["engineering", "hr", "finance", "general", "marketing"]
#     VECTOR_STORE_DIR = Path("resources/vector_store")
#     TEST_QUERY = "What are testing strategy include?"
#     TEST_ROLE = "c-level" # Test with role having wide access
    
#     # Initialize embedding model with appropriate configuration
#     embeddings = HuggingFaceEmbeddings(
#         model_name=MODEL_NAME,
#         model_kwargs={'device': 'cpu'},
#         encode_kwargs={'normalize_embeddings': True}
#     )

#     rag_service = RAGService(
#         categories=CATEGORIES,
#         embedding_model=embeddings,
#         persist_base_dir=str(VECTOR_STORE_DIR.absolute())
#     )
    
#     # Test RAG service
#     response = await rag_service.get_rag_response(
#         query=TEST_QUERY,
#         role=TEST_ROLE
#     )
    
#     # Print test results
#     print("\n--- TEST RESULTS ---")
#     print(f"Query: {TEST_QUERY}")
#     print(f"Role: {TEST_ROLE}")
#     print(f"Response: {response}")
#     print("--- END OF TEST ---")
    
#     return response
    
# if __name__ == "__main__":
#     # Run test asynchronously
#     import asyncio
#     try:
#         asyncio.run(test_rag_service())
#     except KeyboardInterrupt:
#         print("\nTest interrupted by user")
#     except Exception as e:
#         print(f"\nCritical error occurred: {str(e)}")
#         exit(1)