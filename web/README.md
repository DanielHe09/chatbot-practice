# RAG Chatbot Frontend

A modern React frontend for the RAG Chatbot application.

## Features

- **Document Upload**: Upload PDF or TXT files to ingest into the vector store
- **Chat Interface**: Clean, modern chat UI for interacting with the AI agent
- **Real-time Updates**: See messages as they're sent and received
- **Error Handling**: User-friendly error messages

## Setup

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

The app will be available at `http://localhost:3000`

## API Endpoints

The frontend connects to the backend API at `http://localhost:8000`:

- **POST /chat**: Send messages to the AI agent
- **POST /ingest**: Upload documents for ingestion

## Environment Variables

Create a `.env` file in the `web` directory to customize the API URL:

```
VITE_API_URL=http://localhost:8000
```

## Build for Production

```bash
npm run build
```

The built files will be in the `dist` directory.
