import numpy as np
from insightface.app import FaceAnalysis

class FaceEngine:
    def __init__(self, model_name='buffalo_l'):
        self.app = FaceAnalysis(name=model_name, providers=['CUDAExecutionProvider','CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def get_face_features(self, frame):
        """
        Detects faces and returns a list of dictionaries containing:
        - 'bbox': [x1, y1, x2, y2]
        - 'embedding': 512-d vector
        - 'gender': gender prediction
        - 'age': age prediction
        """
        faces = self.app.get(frame)
        results = []
        for face in faces:
            results.append({
                'bbox': face['bbox'].astype(int),
                'embedding': face['embedding'],
                'gender': face['gender'],
                'age': face['age']
            })
        return results

    @staticmethod
    def compute_similarity(feat1, feat2):
        """Computes Cosine Similarity between two embeddings"""
        return np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))