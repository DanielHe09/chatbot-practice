import { useState } from 'react';
import DocumentUpload from './components/DocumentUpload';
import Chat from './components/Chat';
import './App.css';

function App() {
  return (
    <div className="app">
      <div className="header">
        <h1>RAG Chatbot</h1>
        <p>Chat with your documents using RAG (Retrieval-Augmented Generation)</p>
      </div>
      
      <DocumentUpload />
      <Chat />
    </div>
  );
}

export default App;
