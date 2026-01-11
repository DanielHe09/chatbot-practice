from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from supabase import create_client, Client
import tempfile

# Load environment variables
load_dotenv()

app = FastAPI(title="Chatbot RAG API")

# Initialize Supabase connection
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")  # Use service role key for admin access
SUPABASE_TABLE_NAME = os.getenv("SUPABASE_TABLE_NAME", "documents")  # Default table name

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in .env file")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize embeddings
from langchain_openai import OpenAIEmbeddings

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY must be set in .env file for embeddings")

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

# Initialize vector store (will be created/connected on first use)
vector_store = None

def get_vector_store():
    """Get or create the Supabase vector store"""
    global vector_store
    if vector_store is None:
        vector_store = SupabaseVectorStore(
            client=supabase,
            embedding=embeddings,
            table_name=SUPABASE_TABLE_NAME,
        )
    return vector_store

agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[],
    system_prompt="You are a helpful assistant that can retreive context from a vector store to answer use questions",
)


def retrieve_context():
    return "This is the context of the document"

# Enable CORS for Chrome extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your extension's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[ChatMessage]] = []


class ChatResponse(BaseModel):
    response: str


@app.get("/")
def root():
    return {"message": "Chatbot RAG API is running"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    
    try:
        user_message = request.message
        conversation_history = request.conversation_history or []

        # Run the agent
        result = agent.invoke({"messages": [{"role": "user", "content": user_message}]})
        
        # Extract response from agent result
        # The exact structure depends on your agent implementation
        response_text = result.get("output", str(result)) if isinstance(result, dict) else str(result)
        
        return ChatResponse(response=response_text)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest")
async def ingest_documents(file: UploadFile = File(...)):
    """
    Endpoint to ingest documents into your Supabase vector store.
    
    Accepts PDF or text files, splits them into chunks, creates embeddings,
    and stores them in Supabase.
    
    Example usage:
    curl -X POST "http://localhost:8000/ingest" \
         -H "accept: application/json" \
         -H "Content-Type: multipart/form-data" \
         -F "file=@document.pdf"
    """
    try:
        # Check file type
        file_extension = file.filename.split('.')[-1].lower() if file.filename else ''
        
        if file_extension not in ['pdf', 'txt']:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Please upload a PDF or TXT file."
            )
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Load document based on file type
            if file_extension == 'pdf':
                loader = PyPDFLoader(tmp_file_path)
            else:  # txt
                loader = TextLoader(tmp_file_path)
            
            documents = loader.load()
            
            # Split documents into chunks
            chunks = text_splitter.split_documents(documents)
            
            # Get vector store and add documents
            vector_store = get_vector_store()
            vector_store.add_documents(chunks)
            
            return {
                "message": "Documents ingested successfully",
                "filename": file.filename,
                "chunks_created": len(chunks),
                "total_documents": len(documents)
            }
        
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ingesting document: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
