# DS RPC 01: RAG-based Internal Chatbot

A powerful Retrieval-Augmented Generation (RAG) chatbot with role-based access control, built with FastAPI (backend) and Streamlit (frontend). This application allows users to interact with documents through natural language queries, retrieving relevant information using semantic search and generating human-like responses.

## 📌 Problem Statement
The aim is to design a chatbot that enables different teams to access role-specific data while maintaining secure access for Finance, Marketing, HR, C-Level Executives, and Employees. 

-	Authentication and Role Assignment: The chatbot should authenticate users and assign them their roles.
-	 Data Handling: Respond to queries based on the corresponding department data (Finance, Marketing, HR, General), also providing reference to the source document.
-	NLP: Process and understand natural language queries.
-	Role-Based Access Control: Ensure role-based data access.
-	RAG: Retrieve data, augment it with context, and generate a clear, insightful response.

## 🚀 Features

- **Document Processing**: Upload and process various document formats
- **Semantic Search**: Find relevant information using vector embeddings
- **Natural Language Interaction**: Chat with your documents using natural language
- **Role-Based Access Control**: Secure access with different permission levels
- **Modern Web Interface**: Clean and responsive UI built with Streamlit
- **RESTful API**: Built with FastAPI for easy integration
- **Vector Database**: ChromaDB for efficient document retrieval
- **HuggingFace Integration**: State-of-the-art language models and embeddings

## 🛠️ Tech Stack

- **Backend**: FastAPI
- **Frontend**: Streamlit
- **Vector Database**: ChromaDB
- **Language Models**: Gemini 
- **Embeddings**: Qwen/Qwen3-Embedding
- **Data validation**: Pydantic BaseModel 

## 🚀 Setup and Installation
### Project Structure

```
ds-rpc-01/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI application
│   └── services/         # Services module
│       ├── document_loader
│       ├── vector_store
│       └── rag.py         
├── streamlit_app.py      # Streamlit frontend
├── requirements.txt      # Python dependencies
└── README.md            # This file
```
### Prerequisites
- Python 3.8+
- pip (Python package manager)
- Git

### 1. Clone the Repository
```bash
git clone <repository-url>
cd ds-rpc-01
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Variables
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your-gemini-api-key-here
```

## 🏃 Running the Application

### 1. Start the FastAPI Backend
Open a terminal and run:
```bash
cd app
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

### 2. Start the Streamlit Frontend
Open another terminal and run:
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## 🤖 How It Works

1. **Document Ingestion**: Upload documents which are processed and stored in the vector database
2. **Query Processing**: User queries are converted to embeddings
3. **Semantic Search**: The system retrieves the most relevant document chunks
4. **Response Generation**: The LLM generates a response based on the retrieved context
5. **Response Delivery**: The response is returned to the user

## Fictional user accounts are available

| Username | Password    | Role       |
|----------|-------------|------------|
| Tony     | password123 | engineering|
| Bruce    | securepass  | marketing  |
| Sam      | financepass | finance    |
| Peter    | pete123     | engineering|
| Sid      | sidpass123  | marketing  |
| Natasha  | hrpass123   | hr         |

## API Endpoints

- `GET /login` - User login (Basic Auth required)
- `GET /query?message=your_message` - Send a message to the chatbot
- `GET /test` - Test endpoint for authentication

---
## Demo
https://github.com/user-attachments/assets/0629dcd7-80eb-4f87-812e-6c92d51edd45

This project was made as a part of the [Codebasics Resume Challenge] for educational and portfolio purposes.

All user data used is fictional for simulation purposes.

Visit the challenge page to learn more: [DS RPC-01](https://codebasics.io/challenge/codebasics-gen-ai-data-science-resume-project-challenge)
![alt text](resources/RPC_01_Thumbnail.jpg)
