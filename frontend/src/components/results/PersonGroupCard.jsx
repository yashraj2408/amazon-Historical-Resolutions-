export default function PersonGroupCard({ group }) {
  const handleClick = () => {
    // Navigate to person detail view or filter photos
    console.log('View person:', group.person_id);
  };

  return (
    <button
      onClick={handleClick}
      className="card group cursor-pointer h-full flex flex-col"
    >
      <div className="relative aspect-square overflow-hidden rounded-lg bg-gray-100">
        {group.representative_face ? (
          <img
            src={group.representative_face}
            alt={`Person ${group.person_id}`}
            className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center bg-gray-100">
            <svg className="w-16 h-16 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3.135 3.135 0 011.452 0 5.002 5.002 0 015.497 5.497" />
            </svg>
          )}
        )}
        <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
      </div>
      <div className="p-4 flex-1 flex flex-col">
        <h3 className="font-semibold text-gray-900 truncate">
          {group.person_id.replace('_', ' ')}
        </h3>
        <p className="text-sm text-gray-500 mt-1">
          {group.face_count} {group.face_count === 1 ? 'photo' : 'photos'}
        </p>
      </div>
      <div className="p-4 border-t border-gray-100">
        <div className="flex items-center justify-between text-sm">
          <span className="text-gray-500">{group.photo_ids.length} photos</span>
          <button
            onClick={(e) => e.stopPropagation()}
            className="text-amber-600 hover:text-amber-700 font-medium"
          >
            View →
          </button>
        </div>
      </div>
    </button>
  );
}

