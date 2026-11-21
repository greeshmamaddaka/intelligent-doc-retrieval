"""
Embeddings & Vector Search Module.
Integrates SentenceTransformers & FAISS vector index search with robust NumPy fallback.
"""
import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from app.ingest import DocumentChunk

class VectorStoreEngine:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.vectors: List[np.ndarray] = []
        self.chunks: List[DocumentChunk] = []
        self._faiss_index = None
        self._try_init_faiss()

    def _try_init_faiss(self):
        try:
            import faiss
            self._faiss_index = faiss.IndexFlatIP(self.dimension)
            self.use_faiss = True
        except ImportError:
            self.use_faiss = False

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: np.ndarray):
        """Add text chunks and their normalized embedding vectors to index."""
        if len(chunks) == 0:
            return
        
        # Ensure L2 normalized vectors for Cosine Similarity (Inner Product)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        norm_embeddings = (embeddings / norms).astype(np.float32)

        for chunk, vec in zip(chunks, norm_embeddings):
            self.chunks.append(chunk)
            self.vectors.append(vec)

        if self.use_faiss and self._faiss_index is not None:
            self._faiss_index.add(norm_embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int = 3, threshold: float = 0.0) -> List[Tuple[DocumentChunk, float]]:
        """Search top-K nearest neighbors using cosine similarity / vector inner product."""
        if len(self.chunks) == 0:
            return []

        # Normalize query vector
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_vec = (query_embedding / norm).astype(np.float32)
        else:
            query_vec = query_embedding.astype(np.float32)

        if self.use_faiss and self._faiss_index is not None and self._faiss_index.ntotal > 0:
            query_arr = np.expand_dims(query_vec, axis=0)
            scores, indices = self._faiss_index.search(query_arr, min(top_k, len(self.chunks)))
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and score >= threshold:
                    results.append((self.chunks[idx], float(score)))
            return results
        else:
            # NumPy vector similarity computation
            matrix = np.array(self.vectors) # shape: (N, D)
            sim_scores = np.dot(matrix, query_vec) # shape: (N,)
            top_indices = np.argsort(sim_scores)[::-1][:top_k]

            results = []
            for idx in top_indices:
                score = float(sim_scores[idx])
                if score >= threshold:
                    results.append((self.chunks[idx], score))
            return results

    def save(self, index_path: str, metadata_path: str):
        """Serialize index vectors and chunk metadata."""
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        if len(self.vectors) > 0:
            np.save(index_path, np.array(self.vectors))
        meta_data = [chunk.model_dump() for chunk in self.chunks]
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(meta_data, f, indent=2)

    def load(self, index_path: str, metadata_path: str):
        """Load vector index and metadata from disk."""
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            vec_arr = np.load(index_path)
            with open(metadata_path, 'r', encoding='utf-8') as f:
                meta_data = json.load(f)
            
            self.chunks = [DocumentChunk(**item) for item in meta_data]
            self.vectors = [vec for vec in vec_arr]
            if self.use_faiss and self._faiss_index is not None and len(vec_arr) > 0:
                self._faiss_index.reset()
                self._faiss_index.add(vec_arr.astype(np.float32))

class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dimension: int = 384):
        self.model_name = model_name
        self.dimension = dimension
        self.st_model = None
        self._try_load_st()

    def _try_load_st(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.st_model = SentenceTransformer(self.model_name)
        except Exception:
            self.st_model = None

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode list of text strings into vector embeddings."""
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        if self.st_model is not None:
            embeddings = self.st_model.encode(texts, convert_to_numpy=True)
            return embeddings.astype(np.float32)
        else:
            # High-performance semantic hash & word vector fallback
            return self._fallback_encode(texts)

    def _fallback_encode(self, texts: List[str]) -> np.ndarray:
        """Determinstic pseudo-semantic embedding vector representation for local execution."""
        embeddings = []
        for text in texts:
            vec = np.zeros(self.dimension, dtype=np.float32)
            words = text.lower().split()
            if not words:
                embeddings.append(vec)
                continue
            
            for i, word in enumerate(words):
                # Hash characters to vector dimensions
                h1 = hash(word) % self.dimension
                h2 = hash(word[::-1]) % self.dimension
                vec[h1] += 1.0
                vec[h2] += 0.5

            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            embeddings.append(vec)
        return np.array(embeddings, dtype=np.float32)
