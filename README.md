# RAG Chatbot - Learning Template

A basic template for learning how to build a chatbot with RAG (Retrieval-Augmented Generation) using LangChain, FastAPI, and React.

## Project Structure

```
.
├── backend/           # FastAPI server
│   ├── main.py       # Main API with RAG implementation
│   └── requirements.txt
├── web/               # Web frontend
│   └── index.html    # Simple standalone HTML page
└── README.md
```

## Setup Instructions

### Backend (FastAPI)

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start the server:
```bash
python main.py
```

The API will run on `http://localhost:8000`

### Frontend (Web App)

1. Make sure your backend is running on `http://localhost:8000`

2. Open the web app:
   - Simply open `web/index.html` in your browser
   - Double-click the file, or drag it into your browser
   - No build step required!

3. The web app allows you to:
   - Upload PDF or TXT documents to ingest into your vector store
   - Chat with the RAG chatbot about your documents

## Learning Path - Where to Implement RAG

### Step 1: Set up LangChain and Vector Store

In `backend/main.py`, you'll need to:

1. **Install LangChain dependencies** (uncomment in `requirements.txt`):
   ```bash
   pip install langchain langchain-openai chromadb
   # or: pip install langchain langchain-community faiss-cpu
   ```

2. **Create a vector store** (add this to `main.py`):
   ```python
   # TODO: Initialize your vector store here
   # Example with Chroma:
   # from langchain.vectorstores import Chroma
   # from langchain.embeddings import OpenAIEmbeddings
   # 
   # embeddings = OpenAIEmbeddings()
   # vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
   ```

### Step 2: Implement Document Ingestion

In the `/ingest` endpoint (`backend/main.py`):

1. **Load documents** (PDF, text files, etc.)
2. **Split into chunks** using LangChain's `TextSplitter`
3. **Create embeddings** and store in vector database

Example structure:
```python
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import PyPDFLoader  # or TextLoader
# 
# loader = PyPDFLoader("your_document.pdf")
# documents = loader.load()
# 
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# splits = text_splitter.split_documents(documents)
# 
# vectorstore.add_documents(splits)
```

### Step 3: Implement RAG Chain

In the `/chat` endpoint (`backend/main.py`):

1. **Set up the retrieval chain**:
   ```python
   # from langchain.chains import ConversationalRetrievalChain
   # from langchain.chat_models import ChatOpenAI
   # 
   # llm = ChatOpenAI(temperature=0)
   # retriever = vectorstore.as_retriever()
   # 
   # chain = ConversationalRetrievalChain.from_llm(
   #     llm=llm,
   #     retriever=retriever,
   #     return_source_documents=True
   # )
   ```

2. **Process the query**:
   ```python
   # result = chain({
   #     "question": user_message,
   #     "chat_history": [(msg.role, msg.content) for msg in conversation_history]
   # })
   # response_text = result["answer"]
   ```

### Step 4: Test Your Implementation

1. Start the backend server
2. Open `web/index.html` in your browser
3. Upload a document and start chatting!

## Key Concepts to Learn

- **RAG (Retrieval-Augmented Generation)**: Combines retrieval of relevant documents with LLM generation
- **Vector Stores**: Store document embeddings for semantic search (Chroma, FAISS, Pinecone)
- **Embeddings**: Convert text to numerical vectors for similarity search
- **LangChain Chains**: Orchestrate the RAG pipeline (RetrievalQA, ConversationalRetrievalChain)
- **Document Chunking**: Split large documents into smaller pieces for better retrieval

## Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/use_cases/question_answering/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## Next Steps

1. ✅ Set up the basic structure (you're here!)
2. ⬜ Implement document ingestion
3. ⬜ Set up vector store and embeddings
4. ⬜ Implement RAG chain in `/chat` endpoint
5. ⬜ Test with real documents
6. ⬜ Add error handling and improvements

Happy learning! 🚀
