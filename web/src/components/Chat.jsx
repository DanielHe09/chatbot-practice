import { useState, useRef, useEffect } from 'react';
import { sendChatMessage } from '../services/api';

const Chat = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMessage = { role: 'user', content: input.trim() };
    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    setInput('');
    setLoading(true);

    try {
      const response = await sendChatMessage(
        userMessage.content,
        messages.map(msg => ({ role: msg.role, content: msg.content }))
      );

      setMessages([...newMessages, { 
        role: 'assistant', 
        content: response.response || response.output || 'No response received'
      }]);
    } catch (error) {
      console.error('Error sending message:', error);
      setMessages([...newMessages, { 
        role: 'assistant', 
        content: `Error: ${error.message}. Make sure the backend is running on http://localhost:8000` 
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="chat-container">
      <div className="messages">
        {messages.length === 0 && (
          <div className="welcome-message">
            <h3>RAG Chatbot</h3>
            <p>Start a conversation with the AI agent!</p>
            <p style={{ marginTop: '10px', fontSize: '14px', color: '#666' }}>
              Upload a document above, then ask questions about it.
            </p>
          </div>
        )}
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.role}`}>
            <div className="message-content">{msg.content}</div>
          </div>
        ))}
        {loading && (
          <div className="message assistant">
            <div className="message-content">Thinking...</div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      <div className="input-container">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Type your message..."
          disabled={loading}
        />
        <button onClick={handleSendMessage} disabled={loading || !input.trim()}>
          Send
        </button>
      </div>
    </div>
  );
};

export default Chat;
