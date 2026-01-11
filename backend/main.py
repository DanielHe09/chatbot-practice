from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
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
SUPABASE_KEY = os.getenv("SUPABASE_KEY")  # Must be the SERVICE ROLE KEY (secret), not the anon/public key
SUPABASE_TABLE_NAME = os.getenv("SUPABASE_TABLE_NAME", "embeddings")  # Default to "embeddings" (your actual table name)

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError(
        "SUPABASE_URL and SUPABASE_KEY must be set in .env file. "
        "SUPABASE_KEY must be the SERVICE ROLE KEY (secret key) from Supabase dashboard, "
        "not the anon/public key. Find it in Settings → API → service_role key."
    )

# Validate URL format - should be https://xxxxx.supabase.co, not a postgresql:// connection string
if not SUPABASE_URL.startswith("https://"):
    raise ValueError(
        f"SUPABASE_URL must be the Supabase project URL (e.g., https://xxxxx.supabase.co), "
        f"not a PostgreSQL connection string. Current value: {SUPABASE_URL[:50]}..."
    )

# Validate key format
if not SUPABASE_KEY:
    raise ValueError("SUPABASE_KEY is empty")
if len(SUPABASE_KEY) < 20:
    raise ValueError(f"SUPABASE_KEY seems too short ({len(SUPABASE_KEY)} chars). Make sure you copied the full key.")
if SUPABASE_KEY.startswith("anon.") or SUPABASE_KEY.startswith("eyJ") and "anon" in SUPABASE_KEY:
    raise ValueError(
        "SUPABASE_KEY appears to be the anon/public key. You need the SERVICE ROLE KEY (secret key). "
        "Find it in Supabase Dashboard → Settings → API → service_role key"
    )

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    if "Invalid API key" in str(e) or "Invalid" in str(e):
        raise ValueError(
            f"Invalid Supabase API key. Make sure you're using the SERVICE ROLE KEY (secret key), "
            f"not the anon/public key. Key starts with: {SUPABASE_KEY[:15]}... "
            f"Error: {str(e)}"
        ) from e
    raise

# Initialize embeddings
from langchain_openai import OpenAIEmbeddings

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY must be set in .env file for embeddings")

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")

# Initialize Anthropic API key
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY must be set in .env file")

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

def add_documents_with_int_ids(documents):
    """
    Add documents to Supabase with integer IDs instead of UUIDs.
    This works around the limitation where the id column must be an integer.
    Uses BIGSERIAL auto-increment, so we don't specify the id - PostgreSQL will assign it.
    """
    # Generate embeddings for all documents
    texts = [doc.page_content for doc in documents]
    doc_embeddings = embeddings.embed_documents(texts)
    
    # Prepare records (without id - let PostgreSQL auto-increment)
    records = []
    for doc, embedding in zip(documents, doc_embeddings):
        # Ensure metadata is a valid JSON object
        doc_metadata = {}
        if hasattr(doc, 'metadata') and doc.metadata:
            # Convert metadata to a JSON-serializable format
            doc_metadata = {k: str(v) for k, v in doc.metadata.items()}
        
        record = {
            "content": doc.page_content,
            "embedding": embedding,  # This should be a list of floats
            "metadata": doc_metadata
        }
        records.append(record)
    
    # Insert in batches (PostgreSQL will auto-assign integer IDs)
    batch_size = 100
    total_inserted = 0
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        try:
            result = supabase.table(SUPABASE_TABLE_NAME).insert(batch).execute()
            total_inserted += len(batch)
        except Exception as e:
            # Provide more detailed error information
            error_msg = str(e)
            if hasattr(e, 'message'):
                error_msg = e.message
            raise Exception(
                f"Failed to insert documents into Supabase. "
                f"Error: {error_msg}. "
                f"Make sure your table has columns: id (BIGSERIAL), content (TEXT), "
                f"metadata (JSONB), embedding (vector(1536)). "
                f"Batch index: {i}, Records in batch: {len(batch)}"
            ) from e
    
    return total_inserted

# Initialize LLM - Switch between Anthropic and OpenAI
USE_ANTHROPIC = os.getenv("USE_ANTHROPIC", "false").lower() == "true"

if USE_ANTHROPIC:
    # Use Claude (requires ANTHROPIC_API_KEY)
    from langchain_anthropic import ChatAnthropic
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20240620",
        anthropic_api_key=ANTHROPIC_API_KEY,
        temperature=0.7,
    )
else:
    # Use OpenAI (cheaper, good for testing)
    from langchain_openai import ChatOpenAI
    MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")  # or "gpt-4" for better quality
    llm = ChatOpenAI(
        model=MODEL_NAME,
        openai_api_key=OPENAI_API_KEY,
        temperature=0.7,
    )

def retrieve_context(query: str, k: int = 4):
    """
    Retrieve relevant context from the vector store based on the user's query.
    Uses the built-in SupabaseVectorStore retriever.
    
    Args:
        query: The user's question/query
        k: Number of relevant documents to retrieve (default: 4)
    
    Returns:
        A string containing the retrieved context
    """
    try:
        vector_store = get_vector_store()
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
        documents = retriever.invoke(query)
        
        if not documents:
            return "No relevant documents found in the knowledge base."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            context_parts.append(f"[Document {i}]\n{content}")
        
        context = "\n\n".join(context_parts)
        print(f"Retrieved {len(documents)} documents for query: {query[:50]}...")
        return context
        
    except Exception as e:
        # If retrieval fails, log the error and return helpful message
        import traceback
        error_details = traceback.format_exc()
        print(f"Error retrieving context: {e}")
        print(f"Traceback: {error_details}")
        return f"Unable to retrieve context from the knowledge base. Error: {str(e)}"

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

        # Step 1: Retrieve relevant context from vector store
        context = retrieve_context(user_message, k=4)
        
        # Step 2: Build the prompt with context
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
        
        # System message that instructs the agent to use the retrieved context
        system_prompt = """You are a helpful assistant that answers questions based on the provided context from a knowledge base.

When answering:
- Use the retrieved context to provide accurate, detailed answers
- If the context contains relevant information, cite it in your response
- If the context doesn't contain enough information to answer the question, say so
- You can also use your general knowledge, but prioritize the provided context
- Be concise but thorough"""
        
        messages = [SystemMessage(content=system_prompt)]
        
        # Add conversation history
        for msg in conversation_history:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                messages.append(AIMessage(content=msg.content))
        
        # Add current user message with context
        user_prompt = f"""Context from knowledge base:
{context}

User question: {user_message}

Please answer the user's question using the context above when available."""
        
        messages.append(HumanMessage(content=user_prompt))
        
        # Step 3: Get response from LLM (the "agent")
        response = llm.invoke(messages)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
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
            
            # Add documents with integer IDs (workaround for bigint id column)
            chunks_created = add_documents_with_int_ids(chunks)
            
            return {
                "message": "Documents ingested successfully",
                "filename": file.filename,
                "chunks_created": chunks_created,
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
