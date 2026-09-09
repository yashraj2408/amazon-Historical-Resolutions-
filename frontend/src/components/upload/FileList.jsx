import ImagePreview from './ImagePreview';

export default function FileList({ files, onRemove, disabled }) {
  if (files.length === 0) return null;

  return (
    <div className="mt-6">
      <h3 className="text-sm font-medium text-gray-900 mb-3">
        Selected Files ({files.length})
      </h3>
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3">
        {files.map((file, index) => (
          <div key={`${file.name}-${index}`} className="relative group">
            <ImagePreview file={file} />
            {!disabled && (
              <button
                onClick={() => onRemove(index)}
                className="absolute top-1 right-1 bg-red-500 text-white rounded-full p-1 opacity-0 group-hover:opacity-100 transition-opacity hover:bg-red-600"
                aria-label={`Remove ${file.name}`}
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            )}
            <div className="mt-2 px-1">
              <p className="text-xs font-medium text-gray-900 truncate">{file.name}</p>
              <p className="text-xs text-gray-500">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default FileList;