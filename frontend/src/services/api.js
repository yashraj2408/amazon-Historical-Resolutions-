const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

class ApiService {
  constructor() {
    this.baseURL = API_URL;
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    // Don't set Content-Type for FormData
    if (options.body instanceof FormData) {
      delete config.headers['Content-Type'];
    }

    const response = await fetch(url, config);
    
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Request failed' }));
      throw new Error(error.detail || `HTTP ${response.status}`);
    }

    return response.json();
  }

  // Health check
  async healthCheck() {
    return this.request('/api/health');
  }

  // Upload photos
  async uploadPhotos(files) {
    const formData = new FormData();
    files.forEach(file => {
      formData.append('files', file);
    });

    return this.request('/api/upload/', {
      method: 'POST',
      body: formData,
      headers: {}, // Let browser set Content-Type with boundary
    });
  }

  // Get job status
  async getJobStatus(jobId) {
    return this.request(`/api/jobs/${jobId}`);
  }

  // Get job results
  async getJobResults(jobId) {
    return this.request(`/api/results/${jobId}`);
  }

  // List jobs
  async listJobs(params = {}) {
    const query = new URLSearchParams(params).toString();
    return this.request(`/api/jobs?${query}`);
  }

  // Delete job
  async deleteJob(jobId) {
    return this.request(`/api/jobs/${jobId}`, {
      method: 'DELETE',
    });
  }
}

export const api = new ApiService();
export default api;