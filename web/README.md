# Web Frontend

A simple standalone HTML page for the RAG Chatbot.

## Usage

1. Make sure your backend is running on `http://localhost:8000`

2. Open `index.html` in your web browser:
   - Double-click the file, or
   - Right-click → Open With → Your Browser, or
   - Drag and drop into your browser

3. The page will:
   - Allow you to upload PDF or TXT documents
   - Chat with the RAG chatbot
   - Display conversation history

## Features

- **Document Upload**: Upload PDF or text files to ingest into your vector store
- **Chat Interface**: Simple, clean chat UI
- **Real-time Updates**: See messages as they're sent and received
- **No Build Required**: Just open the HTML file - uses React from CDN

## API Connection

The frontend connects to:
- `http://localhost:8000/chat` - For chat messages
- `http://localhost:8000/ingest` - For document uploads

If your backend is hosted elsewhere, edit the `API_URL` constant in the HTML file.
