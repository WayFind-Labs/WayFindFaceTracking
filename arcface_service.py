import os
import json
import numpy as np
import cv2
from typing import Union, Dict, Any, Tuple, Optional
import insightface
from insightface.app import FaceAnalysis

class ArcFaceService:
    """
    ArcFace Service for face embedding extraction, storage, and vector comparison.
    
    Primary Functions:
    1. image_to_vector: Extracts 512-D ArcFace feature vector from an image and optionally saves it.
    2. compare_vectors: Compares two feature vectors using cosine similarity and Euclidean distance.
    """
    
    def __init__(self, model_name: str = 'buffalo_sc', providers: Optional[list] = None):
        """
        Initialize the ArcFace model suite.
        
        Args:
            model_name: InsightFace model pack name ('buffalo_sc' for fast CPU, 'buffalo_l' for higher accuracy).
            providers: Execution providers for ONNX Runtime (e.g. ['CPUExecutionProvider']).
        """
        if providers is None:
            providers = ['CPUExecutionProvider']
            
        print(f"[ArcFaceService] Loading ArcFace model suite ({model_name})...")
        self.app = FaceAnalysis(name=model_name, providers=providers)
        # Prepare model with detection size (640, 640)
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        print("[ArcFaceService] Model initialized successfully!")

    def image_to_vector(
        self, 
        image_input: Union[str, np.ndarray], 
        save_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Function 1: Convert image to vector and optionally save it to disk.
        
        Args:
            image_input: Path to image file OR OpenCV BGR image matrix (np.ndarray).
            save_path: Optional path to save vector (supports .npy or .json format).
            metadata: Optional dictionary of metadata (e.g. person_id, name) to store alongside vector.
            
        Returns:
            np.ndarray: 512-dimensional normalized face feature vector.
        """
        # Load image if string path is provided
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image file not found: {image_input}")
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"Failed to read image file: {image_input}")
        elif isinstance(image_input, np.ndarray):
            img = image_input
        else:
            raise TypeError("image_input must be a file path string or numpy ndarray.")

        # Detect faces and extract embeddings
        faces = self.app.get(img)
        
        if len(faces) == 0:
            raise ValueError("No face detected in the provided image.")

        # Pick the largest face if multiple faces detected
        if len(faces) > 1:
            faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
            print(f"[ArcFaceService] Multiple faces detected ({len(faces)}). Selecting largest face.")

        # Extract normalized embedding vector (512-D float32 array)
        embedding = faces[0].normed_embedding.astype(np.float32)

        # Save vector if save_path is provided
        if save_path:
            self.save_vector(embedding, save_path, metadata=metadata)

        return embedding

    @staticmethod
    def save_vector(
        vector: np.ndarray, 
        save_path: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save feature vector to disk (.npy array or .json with metadata).
        
        Args:
            vector: 1D numpy array of face embedding.
            save_path: Destination file path (.npy or .json).
            metadata: Optional metadata dict.
            
        Returns:
            str: Absolute path of saved file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        ext = os.path.splitext(save_path)[1].lower()

        if ext == '.npy':
            np.save(save_path, vector)
            print(f"[ArcFaceService] Saved vector array to {save_path}")
        elif ext == '.json':
            data = {
                "vector": vector.tolist(),
                "dimension": len(vector),
                "metadata": metadata or {}
            }
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            print(f"[ArcFaceService] Saved vector + metadata JSON to {save_path}")
        else:
            # Default to npy if unspecified
            full_path = save_path + '.npy' if not ext else save_path
            np.save(full_path, vector)
            print(f"[ArcFaceService] Saved vector array to {full_path}")
            return os.path.abspath(full_path)

        return os.path.abspath(save_path)

    @staticmethod
    def load_vector(vector_input: Union[str, list, np.ndarray]) -> np.ndarray:
        """
        Load vector from file (.npy or .json), list, or return numpy array.
        """
        if isinstance(vector_input, np.ndarray):
            vec = vector_input
        elif isinstance(vector_input, list):
            vec = np.array(vector_input, dtype=np.float32)
        elif isinstance(vector_input, str):
            if not os.path.exists(vector_input):
                raise FileNotFoundError(f"Vector file not found: {vector_input}")
            ext = os.path.splitext(vector_input)[1].lower()
            if ext == '.npy':
                vec = np.load(vector_input)
            elif ext == '.json':
                with open(vector_input, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                vec = np.array(data["vector"], dtype=np.float32)
            else:
                raise ValueError(f"Unsupported vector file format: {ext}")
        else:
            raise TypeError("Unsupported vector input type.")

        # Ensure vector is 1D float32 and normalized
        vec = vec.astype(np.float32).flatten()
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    @classmethod
    def compare_vectors(
        cls, 
        vector1_input: Union[str, list, np.ndarray], 
        vector2_input: Union[str, list, np.ndarray], 
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Function 2: Compare two vectors and evaluate similarity.
        
        Args:
            vector1_input: First vector (numpy array, list, or path to .npy/.json file).
            vector2_input: Second vector (numpy array, list, or path to .npy/.json file).
            threshold: Similarity threshold (default 0.5 for ArcFace cosine similarity).
            
        Returns:
            Dict containing similarity metrics and match result:
            - similarity_score (cosine similarity, range -1.0 to 1.0, typical 0 to 1)
            - cosine_distance (1 - cosine similarity)
            - euclidean_distance (L2 distance)
            - is_match (bool)
            - match_confidence_percent (float 0-100%)
        """
        v1 = cls.load_vector(vector1_input)
        v2 = cls.load_vector(vector2_input)

        if v1.shape != v2.shape:
            raise ValueError(f"Vector dimension mismatch: {v1.shape} vs {v2.shape}")

        # Cosine Similarity: dot product of normalized vectors
        cosine_similarity = float(np.dot(v1, v2))
        
        # Cosine Distance: 1 - cosine_similarity
        cosine_distance = float(1.0 - cosine_similarity)
        
        # Euclidean Distance: L2 distance between normalized vectors sqrt(2 - 2*cos_sim)
        euclidean_distance = float(np.linalg.norm(v1 - v2))

        # Match status
        is_match = bool(cosine_similarity >= threshold)

        # Approximate confidence percentage bounded between 0% and 100%
        # ArcFace cosine similarity > threshold (0.5) is standard for same identity.
        confidence = float(np.clip(cosine_similarity * 100, 0.0, 100.0))

        result = {
            "similarity_score": round(cosine_similarity, 4),
            "cosine_distance": round(cosine_distance, 4),
            "euclidean_distance": round(euclidean_distance, 4),
            "threshold": threshold,
            "is_match": is_match,
            "match_confidence_percent": round(confidence, 2)
        }

        return result
