import numpy as np
from insightface.app import FaceAnalysis
import faiss

class FaceEngine:
    def __init__(self, model_name='buffalo_s'):
        # Initialize InsightFace
        self.app = FaceAnalysis(name=model_name, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Initialize FAISS index (512-d for ArcFace)
        # IndexFlatIP uses Inner Product, which is Cosine Similarity for normalized vectors
        self.index = faiss.IndexFlatIP(512)
        self.id_map = []  # Maps FAISS index position to Database person_id

    def get_face_features(self, frame):
        faces = self.app.get(frame)
        results = []
        for face in faces:
            # Normalize embedding for Cosine Similarity via Dot Product
            feat = face['embedding']
            norm_feat = feat / np.linalg.norm(feat)
                        
            results.append({
                'bbox': face['bbox'].astype(int),
                'embedding': norm_feat,
                'gender': face['gender'],
                'age': face['age'],
                'kps': face['kps'],
                'det_score': face['det_score'],
                'landmark_3d_68': face['landmark_3d_68'],
                'pose': face['pose'],
                'landmark_2d_106': face['landmark_2d_106'],
            })
        return results

    def update_search_index(self, known_faces):
        """Rebuilds the FAISS index from the database records."""
        if not known_faces:
            return
        
        embeddings = np.array([f[1] for f in known_faces]).astype('float32')
        # Normalize all for dot-product similarity
        faiss.normalize_L2(embeddings)
        
        self.index = faiss.IndexFlatIP(512)
        self.index.add(embeddings)
        self.id_map = [f[0] for f in known_faces]
        
    def compute_similarity(self, emb1, emb2):
        """Computes cosine similarity between two embeddings."""
        emb1_norm = emb1 / np.linalg.norm(emb1)
        emb2_norm = emb2 / np.linalg.norm(emb2)
        return np.dot(emb1_norm, emb2_norm)

    def search_face(self, query_embedding, threshold=0.45):
        """Returns (person_id, score) using FAISS"""
        if self.index.ntotal == 0:
            return None, 0
        
        query_vec = query_embedding.reshape(1, -1).astype('float32')
        distances, indices = self.index.search(query_vec, 1)
        
        score = distances[0][0]
        idx = indices[0][0]
        
        if idx != -1 and score >= threshold:
            return self.id_map[idx], score
        return None, score