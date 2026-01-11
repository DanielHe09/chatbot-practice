# Environment Variables Setup

Create a `.env` file in the `backend/` directory with the following variables:

```env
# Supabase Configuration
SUPABASE_URL=your_supabase_project_url_here
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key_here
SUPABASE_TABLE_NAME=documents

# OpenAI Configuration (for embeddings)
# Anthropic doesn't provide embeddings, so we use OpenAI for vector embeddings
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic Configuration (for the LLM/agent)
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

## How to Get These Values

### Supabase
1. Go to your Supabase project dashboard
2. **SUPABASE_URL**: Found in Settings > API > Project URL
3. **SUPABASE_SERVICE_ROLE_KEY**: Found in Settings > API > service_role key (keep this secret!)
4. **SUPABASE_TABLE_NAME**: The name of your table (default: "documents")

### OpenAI
1. Go to https://platform.openai.com/api-keys
2. Create a new API key
3. Copy it to `OPENAI_API_KEY`

### Anthropic
1. Go to https://console.anthropic.com/
2. Create an API key
3. Copy it to `ANTHROPIC_API_KEY`

## Setting Up Supabase Table

You'll need to create a table in Supabase for storing vectors. Run this SQL in your Supabase SQL editor:

```sql
-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the documents table
CREATE TABLE IF NOT EXISTS documents (
  id BIGSERIAL PRIMARY KEY,
  content TEXT,
  metadata JSONB,
  embedding vector(1536)  -- 1536 is the dimension for OpenAI embeddings
);

-- Create an index for vector similarity search
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops);
```

Note: The embedding dimension (1536) is for OpenAI's text-embedding-ada-002. If you use a different embedding model, adjust the dimension accordingly.
