"""
Face clustering service for grouping faces into person groups.
Uses DBSCAN clustering on face embeddings.
"""
import numpy as np
from typing import List, Dict, Any
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize

from app.models.photo import FaceDetection
from app.config import get_settings


class ClusteringService:
    """
    Clusters face embeddings into person groups using DBSCAN.
    """
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
    
    async def cluster_faces(self, faces: List[FaceDetection]) -> List[Dict[str, Any]]:
        """
        Cluster faces into person groups using DBSCAN.
        Returns list of person groups with metadata.
        """
        # Extract embeddings
        embeddings = []
        valid_faces = []
        
        for face in faces:
            if face.embedding and len(face.embedding) > 0:
                embeddings.append(face.embedding)
                valid_faces.append(face)
            else:
                # Assign noise label to faces without embeddings
                face.person_id = "unknown"
        
        if not embeddings:
            return []
        
        # Convert to numpy array
        X = np.array(embeddings)
        
        # Normalize embeddings
        X = normalize(X, norm='l2')
        
        # DBSCAN clustering
        clustering = DBSCAN(
            eps=self.settings.clustering_eps,
            min_samples=self.settings.clustering_min_samples,
            metric='cosine'
        ).fit(X)
        
        labels = clustering.labels_
        
        # Group faces by cluster
        clusters = {}
        for idx, label in enumerate(labels):
            if label == -1:
                # Noise points get individual IDs
                face = valid_faces[idx]
                face.person_id = f"unknown_{idx}"
                continue
            
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(valid_faces[idx])
        
        # Build person groups
        person_groups = []
        for cluster_id, faces_in_cluster in clusters.items():
            if len(faces_in_cluster) == 0:
                continue
            
            person_id = f"person_{cluster_id + 1}"
            
            # Assign person_id to faces
            for face in faces_in_cluster:
                face.person_id = person_id
            
            # Select representative face (first one)
            rep_face = faces_in_cluster[0]
            
            # Get photo IDs for this person
            photo_ids = list(set(f.photo_id for f in faces_in_cluster if hasattr(f, 'photo_id')))
            
            person_groups.append({
                "person_id": person_id,
                "face_count": len(faces_in_cluster),
                "representative_face": "",  # Will be filled by face service
                "photo_ids": photo_ids,
                "embeddings": [f.embedding for f in faces_in_cluster]
            })
        
        # Sort by face count descending
        person_groups.sort(key=lambda g: g["face_count"], reverse=True)
        
        return person_groups


async def cluster_faces(faces: List[FaceDetection]) -> List[Dict[str, Any]]:
    """Convenience function for clustering faces."""
    service = ClusteringService()
    return await service.cluster_faces(faces)