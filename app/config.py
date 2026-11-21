"""
Configuration settings for Intelligent Document Search and Retrieval System.
"""
import os
from pydantic import BaseModel

class SystemConfig(BaseModel):
    # Chunking settings
    CHUNK_SIZE: int = 400
    CHUNK_OVERLAP: int = 80
    
    # Vector store settings
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    VECTOR_DIMENSION: int = 384
    TOP_K_RETRIEVAL: int = 3
    SIMILARITY_THRESHOLD: float = 0.25
    
    # Storage paths
    INDEX_SAVE_PATH: str = "/workspace/scratch/intelligent_doc_retrieval/data/vector_store.idx"
    METADATA_SAVE_PATH: str = "/workspace/scratch/intelligent_doc_retrieval/data/metadata.json"
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000

config = SystemConfig()
