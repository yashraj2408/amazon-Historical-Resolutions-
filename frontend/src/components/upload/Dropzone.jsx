import { useCallback, useState } from 'react';

export default function Dropzone({ files, onFilesChange, onRemoveFile, disabled }) {
  const [isDragActive, setIsDragActive] = useState(false);

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!disabled) {
      setIsDragActive(e.type === 'dragenter' || e.type === 'dragover');
    }
  }, [disabled]);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
    
    if (disabled) return;
    
    const droppedFiles = Array.from(e.dataTransfer.files);
    handleFiles(droppedFiles);
  }, []);

  const handleFileSelect = useCallback((e) => {
    if (disabled) return;
    const selectedFiles = Array.from(e.target.files);
    handleFiles(selectedFiles);
    e.target.value = ''; // Reset input
  }, []);

  const handleFiles = useCallback((newFiles) => {
    const validFiles = newFiles.filter(file => 
      ['image/jpeg', 'image/png', 'image/webp'].includes(file.type)
    );
    
    if (validFiles.length !== newFiles.length) {
      alert('Some files were skipped. Only JPEG, PNG, and WebP files are allowed.');
    }
    
    onFilesChange(prev => [...prev, ...validFiles]);
  }, [onFilesChange]);

  return (
    <div className="relative">
      <div
        className={`border-2 border-dashed rounded-xl p-8 text-center transition-colors ${
          isDragActive 
            ? 'border-amber-500 bg-amber-50' 
            : 'border-gray-300 hover:border-amber-400'
        } ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        role="button"
        tabIndex={disabled ? -1 : 0}
        onKeyDown={(e) => {
          if ((e.key === 'Enter' || e.key === ' ') && !disabled) {
            e.preventDefault();
            document.getElementById('file-input')?.click();
          }
        }}
      >
        <input
          id="file-input"
          type="file"
          multiple
          accept="image/jpeg,image/png,image/webp"
          onChange={handleFileSelect}
          className="hidden"
          disabled={disabled}
        />
        
        <div className="space-y-4">
          <svg 
            className="mx-auto h-12 w-12 text-gray-400" 
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24"
            aria-hidden="true"
          >
            <path 
              strokeLinecap="round" 
              strokeLinejoin="round" 
              strokeWidth={2} 
              d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" 
            />
          </svg>
          
          <div>
            <p className="text-lg font-medium text-gray-900">Drag & drop photos here</p>
            <p className="text-sm text-gray-500 mt-1">or click to browse</p>
            <p className="text-xs text-gray-400 mt-2">JPEG, PNG, WebP · Max 10MB each · Up to 50 files</p>
          </div>
        </div>
      </div>

      <input
        type="file"
        multiple
        accept="image/jpeg,image/png,image/webp"
        onChange={handleFileSelect}
        className="hidden"
        id="file-input-hidden"
        disabled={disabled}
      />
    </div>
  );
}

export default Dropzone;