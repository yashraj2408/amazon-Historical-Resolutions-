"""
Face detection and embedding service.
Uses OpenCV for lightweight face detection and embeddings.
"""
import cv2
import numpy as np
from typing import List, Optional, Tuple
from app.models.photo import FaceDetection
from app.config import get_settings


class FaceService:
    """
    Lightweight face detection and embedding service using OpenCV.
    Uses FaceDetectorYN for detection (OpenCV 5.x) and LBPH for embeddings.
    """
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self._face_detector = None
        self._initialized = False
    
    def _initialize(self):
        """Lazy initialization of face detection models."""
        if self._initialized:
            return
        
        # Load YuNet face detector (OpenCV 5.x)
        # Download the model if not present
        model_path = cv2.data.haarcascades + "face_detection_yunet_2023mar.onnx"
        try:
            self._face_detector = cv2.FaceDetectorYN_create(
                model_path,
                "",
                (320, 320),
                score_threshold=0.5,
                nms_threshold=0.3,
                top_k=5000,
                backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
                target_id=cv2.dnn.DNN_TARGET_CPU
            )
        except:
            # Fallback: try to find the model file
            import os
            model_dir = cv2.data.haarcascades
            for f in os.listdir(model_dir):
                if f.endswith(".onnx") and "yunet" in f.lower():
                    model_path = os.path.join(model_dir, f)
                    self._face_detector = cv2.FaceDetectorYN_create(
                        model_path,
                        "",
                        (320, 320),
                        score_threshold=0.5,
                        nms_threshold=0.3,
                        top_k=5000,
                        backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
                        target_id=cv2.dnn.DNN_TARGET_CPU
                    )
                    break
            else:
                # If no model found, create a simple detector using the built-in method
                self._face_detector = None
        
        self._initialized = True
    
    async def detect_faces(self, image_data: bytes) -> List[FaceDetection]:
        """
        Detect faces in image.
        Returns list of FaceDetection with bounding boxes.
        """
        self._initialize()
        
        # Decode image
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return []
        
        # Resize if too large
        height, width = img.shape[:2]
        settings = get_settings()
        if max(width, height) > settings.max_image_dimension:
            scale = settings.max_image_dimension / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height))
        
        if self._face_detector is None:
            return []
        
        # Set input size for detector
        height, width = img.shape[:2]
        self._face_detector.setInputSize((width, height))
        
        # Detect faces
        _, faces = self._face_detector.detect(img)
        
        detections = []
        if faces is not None:
            for face in faces:
                x, y, w, h = face[:4].astype(int)
                confidence = face[4] if len(face) > 4 else 0.9
                
                detections.append(FaceDetection(
                    person_id="",  # Will be assigned after clustering
                    bbox=[x, y, w, h],
                    confidence=float(confidence)
                ))
        
        return detections
        
        detections = []
        for (x, y, w, h) in faces:
            # Convert to original image coordinates if scaled
            scale_x = width / img.shape[1] if img.shape[1] > 0 else 1
            scale_y = height / img.shape[0] if img.shape[0] > 0 else 1
            
            detections.append(FaceDetection(
                person_id="",  # Will be assigned after clustering
                bbox=[int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)],
                confidence=0.9  # Haar cascades don't provide confidence
            ))
        
        return detections
    
    async def get_embedding(
        self,
        image_data: bytes,
        bbox: List[int]
    ) -> List[float]:
        """
        Generate face embedding for clustering.
        Uses LBPH histogram as lightweight embedding.
        """
        self._initialize()
        
        # Decode image
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return [0.0] * 64  # Return zero vector on failure
        
        x, y, w, h = bbox
        
        # Extract face region with padding
        padding = 10
        x1 = max(0, x - 10)
        y1 = max(0, y - 10)
        x2 = min(image_data.shape[1] if hasattr(image_data, 'shape') else 9999, x + w + 10)
        y2 = min(image_data.shape[0] if hasattr(image_data, 'shape') else 9999, y + h + 10)
        
        # Re-decode for proper shape access
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return [0.0] * 64
        
        height, width = img.shape[:2]
        x1 = max(0, x - 10)
        y1 = max(0, y - 10)
        x2 = min(width, x + w + 10)
        y2 = min(height, y + h + 10)
        
        face_img = img[y1:y2, x1:x2]
        
        if face_img.size == 0:
            return [0.0] * 64
        
        # Convert to grayscale
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Resize to standard size for LBPH
        gray_face = cv2.resize(gray_face, (64, 64))
        
        # Compute LBPH histogram as embedding
        # LBPH divides image into grid and computes local binary patterns
        # We'll compute a simple histogram as lightweight embedding
        
        # Compute LBP manually for lightweight embedding
        lbp_hist = self._compute_lbp_histogram(gray_face)
        
        # Normalize
        lbp_hist = lbp_hist / (np.sum(lbp_hist) + 1e-7)
        
        return lbp_hist.tolist()
    
    def _compute_lbp_histogram(self, image: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
        """Compute Local Binary Pattern histogram as face embedding."""
        h, w = image.shape
        lbp = np.zeros((h, w), dtype=np.uint8)
        
        # Compute LBP for each pixel
        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                center = image[i, j]
                code = 0
                for k in range(n_points):
                    # Circular neighbors
                    angle = 2 * np.pi * k / n_points
                    ni = int(i + radius * np.sin(angle))
                    nj = int(j + radius * np.cos(angle))
                    if 0 <= ni < h and 0 <= nj < w:
                        if image[ni, nj] >= center:
                            code |= (1 << k)
                lbp[i, j] = code
        
        # Compute histogram
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        return hist.astype(np.float32)


# Global instance
_face_service = None


def get_face_service() -> FaceService:
    global _face_service
    if _face_service is None:
        _face_service = FaceService()
    return _face_service