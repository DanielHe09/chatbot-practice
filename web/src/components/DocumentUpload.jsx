import { useState, useRef } from 'react';
import { uploadDocument } from '../services/api';

const DocumentUpload = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      // Validate file type
      const fileExtension = file.name.split('.').pop().toLowerCase();
      if (!['pdf', 'txt'].includes(fileExtension)) {
        setUploadStatus({
          type: 'error',
          message: 'Please select a PDF or TXT file.',
        });
        setSelectedFile(null);
        if (fileInputRef.current) {
          fileInputRef.current.value = '';
        }
        return;
      }
      setSelectedFile(file);
      setUploadStatus(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setUploading(true);
    setUploadStatus(null);

    try {
      const data = await uploadDocument(selectedFile);
      setUploadStatus({
        type: 'success',
        message: `Successfully uploaded ${data.filename}! Created ${data.chunks_created} chunks from ${data.total_documents} document(s).`,
      });
      setSelectedFile(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    } catch (error) {
      console.error('Upload error:', error);
      setUploadStatus({
        type: 'error',
        message: `Upload failed: ${error.message}. Make sure the backend is running.`,
      });
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="upload-section">
      <h3>Upload Documents</h3>
      <p style={{ fontSize: '14px', color: '#666', marginBottom: '10px' }}>
        Upload PDF or TXT files to add them to the knowledge base
      </p>
      <div className="upload-container">
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.txt"
          onChange={handleFileSelect}
          disabled={uploading}
        />
        <button onClick={handleUpload} disabled={uploading || !selectedFile}>
          {uploading ? 'Uploading...' : 'Upload'}
        </button>
      </div>
      {uploadStatus && (
        <div className={`upload-status ${uploadStatus.type}`}>
          {uploadStatus.message}
        </div>
      )}
    </div>
  );
};

export default DocumentUpload;
