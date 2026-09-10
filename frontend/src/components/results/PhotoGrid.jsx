export default function PhotoGrid({ photos }) {
  if (!photos || photos.length === 0) {
    return (
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-8 text-center">
        <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
        </svg>
        <h3 className="mt-4 text-lg font-medium text-gray-900">No photos</h3>
        <p className="mt-2 text-gray-600">No photos to display.</p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-3">
      {photos.map((photo, index) => (
        <div key={`${photo.photo_id}-${index}`} className="relative group">
          <div className="aspect-square bg-gray-100 rounded-lg overflow-hidden">
            <img
              src={`/api/placeholder/${photo.photo_id}`}
              alt={photo.photo_id}
              className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
              loading="lazy"
            />
          </div>
          <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300">
            <div className="absolute bottom-2 left-2 right-2 flex flex-wrap gap-1">
              {photo.faces.map((face, i) => (
                <span
                  key={i}
                  className="bg-amber-500 text-white text-xs px-2 py-1 rounded-full"
                >
                  {face.person_id}
                </span>
              ))}
            </div>
          </div>
          <div className="mt-2 px-1">
            <p className="text-xs font-medium text-gray-900 truncate">
              {photo.photo_id}
            </p>
            <p className="text-xs text-gray-500">
              {photo.faces.length} face{photo.faces.length !== 1 ? 's' : ''}
            </p>
          </div>
        </div>
      ))}
    </div>
  );
}

