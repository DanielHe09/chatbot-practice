const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * Send a chat message to the AI agent
 * @param {string} message - The user's message
 * @param {Array} conversationHistory - Previous messages in the conversation
 * @returns {Promise<Object>} Response from the API
 */
export const sendChatMessage = async (message, conversationHistory = []) => {
  try {
    const response = await fetch(`${API_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message,
        conversation_history: conversationHistory,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Chat API error:', error);
    throw error;
  }
};

/**
 * Upload a document to the backend for ingestion
 * @param {File} file - The file to upload (PDF or TXT)
 * @returns {Promise<Object>} Response from the API
 */
export const uploadDocument = async (file) => {
  try {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_URL}/ingest`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Upload API error:', error);
    throw error;
  }
};
