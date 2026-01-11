# Backend - FastAPI RAG Chatbot

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

- `GET /` - Health check
- `POST /chat` - Main chat endpoint (TODO: implement RAG here)
- `POST /ingest` - Document ingestion endpoint (TODO: implement document loading)

## Where to Add Your RAG Code

### 1. In `main.py` - Top of file (after imports)

Add your vector store initialization:
```python
# TODO: Initialize vector store here
# Example:
# from langchain.vectorstores import Chroma
# from langchain.embeddings import OpenAIEmbeddings
# 
# embeddings = OpenAIEmbeddings()
# vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
```

### 2. In `/chat` endpoint

Replace the placeholder response with your RAG chain:
- Load your vector store
- Create a ConversationalRetrievalChain
- Process the user query with conversation history
- Return the generated response

### 3. In `/ingest` endpoint

Implement document loading:
- Accept document files (PDF, text, etc.)
- Split into chunks
- Create embeddings
- Store in vector database

## Testing

You can test the API with curl:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!", "conversation_history": []}'
```

Or use the Chrome extension frontend once it's built and loaded.
