# DS RPC 01: RAG-based Internal Chatbot

A powerful Retrieval-Augmented Generation (RAG) chatbot with role-based access control, built with FastAPI (backend) and Streamlit (frontend). This application allows users to interact with documents through natural language queries, retrieving relevant information using semantic search and generating human-like responses.

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
- **Language Models**: HuggingFace Transformers
- **Embeddings**: Sentence Transformers
- **Authentication**: JWT Tokens

## 🚀 Setup and Installation

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
Create a `.env` file in the root directory with the following variables:
```env
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
SECRET_KEY=your_secret_key_for_jwt
ADMIN_USERNAME=admin
ADMIN_PASSWORD=your_secure_password
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

## 🔍 API Endpoints

### Authentication
- `POST /token`: Get access token
- `POST /users/`: Create new user (admin only)
- `GET /users/me/`: Get current user info

### Document Management
- `POST /documents/`: Upload a new document
- `GET /documents/`: List all documents
- `GET /documents/{doc_id}`: Get document details
- `DELETE /documents/{doc_id}`: Delete a document

### Chat
- `POST /chat/`: Send a message to the chatbot
- `GET /chat/history`: Get chat history

## 🤖 How It Works

1. **Document Ingestion**: Upload documents which are processed and stored in the vector database
2. **Query Processing**: User queries are converted to embeddings
3. **Semantic Search**: The system retrieves the most relevant document chunks
4. **Response Generation**: The LLM generates a response based on the retrieved context
5. **Response Delivery**: The response is returned to the user

## 📚 Documentation

For detailed API documentation, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🤝 Contributing

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

The Streamlit app will open in your default browser at `http://localhost:8501`

## Available User Accounts

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

## Project Structure

```
ds-rpc-01/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI application
│   └── services/         # Business logic and services
├── streamlit_app.py      # Streamlit frontend
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Development

- The FastAPI backend uses automatic documentation available at `http://localhost:8000/docs`
- The Streamlit frontend supports hot-reloading - just save your changes

## Security Note

For production use, please:
1. Change the default passwords
2. Implement proper session management
3. Use HTTPS
4. Configure proper CORS settings
5. Consider using environment variables for sensitive data

---

Visit the challenge page to learn more: [DS RPC-01](https://codebasics.io/challenge/codebasics-gen-ai-data-science-resume-project-challenge)
![alt text](resources/RPC_01_Thumbnail.jpg)
