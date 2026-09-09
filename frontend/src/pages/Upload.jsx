import { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../services/api';
import Dropzone from '../components/upload/Dropzone';
import JobStatus from '../components/upload/JobStatus';
import FileList from '../components/upload/FileList';

export default function Upload() {
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [jobId, setJobId] = useState(null);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  const handleFilesChange = useCallback((newFiles) => {
    setFiles(newFiles);
    setError(null);
  }, []);

  const handleUpload = async () => {
    if (files.length === 0) {
      setError('Please select at least one photo');
      return;
    }

    setUploading(true);
    setError(null);

    try {
      const response = await api.uploadPhotos(files);
      setJobId(response.job_id);
      // Poll for job completion
      pollJobStatus(response.job_id);
    } catch (err) {
      setError(err.message || 'Upload failed');
      setUploading(false);
    }
  }, [files]);

  const pollJobStatus = async (jobId) => {
    const checkStatus = async () => {
      try {
        const job = await api.getJobStatus(jobId);
        if (job.status === 'completed') {
          navigate(`/results/${jobId}`);
        } else if (job.status === 'failed') {
          setError(job.error || 'Processing failed');
          setUploading(false);
        } else {
          // Still processing, check again in 2 seconds
          setTimeout(checkStatus, 2000);
        }
      } catch (err) {
        setError(err.message);
        setUploading(false);
      }
    };
    checkStatus();
  };

  const handleRemoveFile = useCallback((index) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  }, []);

  const handleClearAll = useCallback(() => {
    setFiles([]);
    setJobId(null);
    setError(null);
  }, []);

  return (
    <div className="max-w-4xl mx-auto">
      <div className="mb-8">
        <h2 className="text-3xl font-bold text-gray-900 mb-2">Upload Photos</h2>
        <p className="text-gray-600">
          Upload photos to organize them by person using AI face recognition.
        </p>
      </div>

      {error && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg text-red-800" role="alert">
          {error}
        </div>
      )}

      <Dropzone
        files={files}
        onFilesChange={handleFilesChange}
        onRemoveFile={(index) => setFiles(prev => prev.filter((_, i) => i !== index))}
        disabled={uploading}
      />

      <FileList files={files} onRemove={handleRemoveFile} disabled={uploading} />

      <div className="mt-6 flex flex-wrap gap-4">
        <button
          onClick={handleUpload}
          disabled={uploading || files.length === 0}
          className="btn btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {uploading ? 'Processing...' : 'Upload & Organize'}
        </button>

        {files.length > 0 && (
          <button
            onClick={handleClearAll}
            disabled={uploading}
            className="btn btn-secondary disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Clear All
          </button>
        )}
      </div>

      {jobId && !uploading && (
        <JobStatus jobId={jobId} onComplete={() => navigate(`/results/${jobId}`)} />
      )}
    </div>
  );
}

export default Upload;