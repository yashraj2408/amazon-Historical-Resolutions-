import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../../services/api';

export default function JobStatus({ jobId, onComplete }) {
  const [job, setJob] = useState(null);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    if (!jobId) return;

    const checkStatus = async () => {
      try {
        const job = await api.getJobStatus(jobId);
        setJob(job);
        
        if (job.status === 'completed') {
          if (onComplete) {
            onComplete();
          } else {
            navigate(`/results/${jobId}`);
          }
        } else if (job.status === 'failed') {
          setError(job.error || 'Processing failed');
        } else {
          // Still processing, check again in 2 seconds
          setTimeout(checkStatus, 2000);
        }
      } catch (err) {
        setError(err.message);
      }
    };

    checkStatus();
  }, [jobId, navigate]);

  if (!job) {
    return (
      <div className="mt-6 p-4 bg-gray-50 border border-gray-200 rounded-lg">
        <div className="flex items-center space-x-3">
          <div className="w-6 h-6 border-2 border-amber-500 border-t-transparent rounded-full animate-spin" />
          <p className="text-gray-700">Processing your photos...</p>
        </div>
      </div>
    );
  }

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed': return 'text-green-600 bg-green-100';
      case 'failed': return 'text-red-600 bg-red-100';
      case 'processing': return 'text-amber-600 bg-amber-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  return (
    <div className="mt-6 p-4 bg-gray-50 border border-gray-200 rounded-lg">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-medium text-gray-900">Processing Status</h3>
        <span className={`badge ${getStatusColor(job.status)}`}>
          {job.status.charAt(0).toUpperCase() + job.status.slice(1)}
        </span>
      </div>

      <div className="mb-4">
        <div className="flex justify-between text-sm mb-1">
          <span className="text-gray-600">Progress</span>
          <span className="font-medium">{job.progress}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div 
            className="bg-amber-500 h-2 rounded-full transition-all duration-300"
            style={{ width: `${job.progress}%` }}
          />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4 text-sm">
        <div>
          <span className="text-gray-500">Total Photos</span>
          <p className="font-medium">{job.total_files}</p>
        </div>
        <div>
          <span className="text-gray-500">Processed</span>
          <p className="font-medium">{job.processed_files}</p>
        </div>
      </div>

      {job.error && (
        <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-800">
          <p className="font-medium">Error:</p>
          <p>{job.error}</p>
        </div>
      )}
    </div>
  );
}

export default JobStatus;