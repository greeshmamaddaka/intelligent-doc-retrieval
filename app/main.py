"""
FastAPI Application Entry Point.
Provides RESTful APIs for document ingestion, semantic search, RAG, and system evaluation.
"""
import os
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from app.config import config
from app.ingest import DocumentIngestor, DocumentChunk
from app.embeddings import VectorStoreEngine, EmbeddingModel
from app.rag import RAGPipeline, RAGResult
from app.eval import RetrievalEvaluator, BenchmarkQuery, EvaluationMetrics

app = FastAPI(
    title="Intelligent Document Search and Retrieval System",
    description="Vector Similarity Search & Retrieval-Augmented Generation (RAG) REST API",
    version="1.0.0"
)

# Global core system instances
ingestor = DocumentIngestor(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
embedder = EmbeddingModel(model_name=config.EMBEDDING_MODEL_NAME, dimension=config.VECTOR_DIMENSION)
vector_store = VectorStoreEngine(dimension=config.VECTOR_DIMENSION)

# Load existing index if present
if os.path.exists(config.INDEX_SAVE_PATH) and os.path.exists(config.METADATA_SAVE_PATH):
    try:
        vector_store.load(config.INDEX_SAVE_PATH, config.METADATA_SAVE_PATH)
    except Exception:
        pass

rag_pipeline = RAGPipeline(vector_store=vector_store, embedder=embedder)
evaluator = RetrievalEvaluator(rag_pipeline=rag_pipeline)

# Request / Response Models
class SingleDocumentRequest(BaseModel):
    doc_id: str = Field(..., example="doc_001")
    title: str = Field(..., example="FastAPI & Vector Search Overview")
    text: str = Field(..., example="FastAPI is a modern high-performance web framework for Python...")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class BatchIngestRequest(BaseModel):
    documents: List[SingleDocumentRequest]

class SearchQueryRequest(BaseModel):
    query: str = Field(..., example="How does vector similarity search work?")
    top_k: Optional[int] = Field(default=3, ge=1, le=20)
    threshold: Optional[float] = Field(default=0.0, ge=0.0, le=1.0)

class RAGQueryRequest(BaseModel):
    query: str = Field(..., example="Explain RAG context integration.")
    top_k: Optional[int] = Field(default=3, ge=1, le=20)

class BenchmarkEvalRequest(BaseModel):
    benchmarks: List[BenchmarkQuery]
    top_k: Optional[int] = Field(default=3)

@app.get("/health", tags=["System"])
def health_check():
    """Check system health, vector index size, and loaded model metadata."""
    return {
        "status": "online",
        "vector_store": {
            "total_chunks": len(vector_store.chunks),
            "faiss_enabled": vector_store.use_faiss,
            "dimension": vector_store.dimension
        },
        "embedding_model": {
            "name": embedder.model_name,
            "backend": "SentenceTransformers" if embedder.st_model is not None else "NumPy Semantic Engine"
        }
    }

@app.post("/documents/ingest", tags=["Ingestion"], status_code=status.HTTP_201_CREATED)
def ingest_documents(payload: BatchIngestRequest):
    """Process, clean, chunk, embed, and index a batch of raw documents."""
    all_chunks: List[DocumentChunk] = []
    
    for doc in payload.documents:
        chunks = ingestor.process_document(
            doc_id=doc.doc_id, title=doc.title, text=doc.text, metadata=doc.metadata
        )
        all_chunks.extend(chunks)

    if not all_chunks:
        raise HTTPException(status_code=400, detail="No text content extracted from documents.")

    # Encode chunks into embeddings
    texts = [c.content for c in all_chunks]
    embeddings = embedder.encode(texts)
    
    # Add to vector index
    vector_store.add_chunks(all_chunks, embeddings)
    
    # Save index to disk
    vector_store.save(config.INDEX_SAVE_PATH, config.METADATA_SAVE_PATH)

    return {
        "message": f"Successfully ingested {len(payload.documents)} document(s).",
        "created_chunks_count": len(all_chunks),
        "total_index_size": len(vector_store.chunks)
    }

@app.post("/search", tags=["Search"])
def search_documents(payload: SearchQueryRequest):
    """Perform top-K semantic vector search across indexed document chunks."""
    results = rag_pipeline.retrieve_context(
        query=payload.query, top_k=payload.top_k, threshold=payload.threshold
    )
    
    formatted_results = []
    for chunk, score in results:
        formatted_results.append({
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "doc_title": chunk.doc_title,
            "similarity_score": round(score, 4),
            "content": chunk.content,
            "metadata": chunk.metadata
        })

    return {
        "query": payload.query,
        "results_count": len(formatted_results),
        "results": formatted_results
    }

@app.post("/rag/query", response_model=RAGResult, tags=["RAG"])
def rag_query(payload: RAGQueryRequest):
    """Execute complete Retrieval-Augmented Generation query workflow."""
    result = rag_pipeline.generate_answer(query=payload.query, top_k=payload.top_k)
    return result

@app.post("/eval/benchmark", response_model=EvaluationMetrics, tags=["Evaluation"])
def run_evaluation(payload: BenchmarkEvalRequest):
    """Run evaluation benchmark suite calculating Precision@K, Recall@K, and MRR."""
    metrics = evaluator.run_benchmark(payload.benchmarks, top_k=payload.top_k)
    return metrics
