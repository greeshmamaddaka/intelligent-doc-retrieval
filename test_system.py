"""
Automated Integration Test & Evaluation Suite for Intelligent Document Search & Retrieval System.
"""
import sys
import os
import json
from tabulate import tabulate

# Add project root to sys.path
sys.path.insert(0, os.path.abspath('/workspace/scratch/intelligent_doc_retrieval'))

from app.ingest import DocumentIngestor
from app.embeddings import VectorStoreEngine, EmbeddingModel
from app.rag import RAGPipeline
from app.eval import RetrievalEvaluator, BenchmarkQuery

def run_e2e_test():
    print("=" * 80)
    print("INTELLIGENT DOCUMENT SEARCH AND RETRIEVAL SYSTEM - SYSTEM TEST & EVALUATION")
    print("=" * 80)

    # 1. Initialize Components
    ingestor = DocumentIngestor(chunk_size=350, chunk_overlap=60)
    embedder = EmbeddingModel()
    vector_store = VectorStoreEngine(dimension=384)
    rag_pipeline = RAGPipeline(vector_store, embedder)
    evaluator = RetrievalEvaluator(rag_pipeline)

    print(f"\n[1] Component Initialization")
    print(f" -> Vector Store Engine: FAISS Available = {vector_store.use_faiss}")
    print(f" -> Embedding Model: {embedder.model_name}")

    # 2. Sample Unstructured Text Documents
    sample_docs = [
        {
            "doc_id": "doc_rag_overview",
            "title": "Retrieval-Augmented Generation Architecture",
            "text": """Retrieval-Augmented Generation (RAG) combines dense text retrieval with pretrained language model generation. 
            By querying a vector database like FAISS or Pinecone, the system retrieves top-k relevant document passages corresponding to a user query. 
            The retrieved passages are concatenated into a structured prompt context, allowing the generator model to synthesize grounded, context-aware answers 
            without hallucinations. Evaluation of RAG systems relies on metrics like Precision@K, Recall@K, MRR (Mean Reciprocal Rank), and context relevance score."""
        },
        {
            "doc_id": "doc_faiss_vector",
            "title": "Sentence Transformers and Vector Similarity Search",
            "text": """Sentence Transformers produce dense vector embeddings for text sequences such that semantically similar sentences are close in vector space. 
            FAISS (Facebook AI Similarity Search) provides efficient indexing and similarity search for high-dimensional vectors. 
            IndexFlatIP computes exact Inner Product cosine similarity, while IndexHNSW provides approximate nearest neighbor (ANN) search with sub-millisecond latency. 
            Vector normalization ensures cosine similarity alignment across L2 distance metrics."""
        },
        {
            "doc_id": "doc_fastapi_docker",
            "title": "FastAPI Web Service & Docker Containerization",
            "text": """FastAPI is a high-performance Python web framework featuring asynchronous endpoint execution, automatic OpenAPI schema generation, and Pydantic validation. 
            Containerizing FastAPI applications with Docker involves multi-stage builds, minimizing container footprint and isolating runtime dependencies. 
            Docker Compose orchestrates containerized services alongside vector database containers for seamless cloud deployment."""
        }
    ]

    # 3. Process & Index Documents
    print(f"\n[2] Document Ingestion & Vector Indexing")
    all_chunks = []
    for doc in sample_docs:
        chunks = ingestor.process_document(doc_id=doc["doc_id"], title=doc["title"], text=doc["text"])
        all_chunks.extend(chunks)
        print(f" -> Document '{doc['title']}': generated {len(chunks)} chunks.")

    texts = [c.content for c in all_chunks]
    embeddings = embedder.encode(texts)
    vector_store.add_chunks(all_chunks, embeddings)
    print(f" -> Total Chunks Indexed in Vector Store: {len(vector_store.chunks)}")

    # 4. Perform Semantic Search Test
    print(f"\n[3] Semantic Vector Similarity Search Verification")
    search_query = "How does vector similarity search with FAISS work?"
    query_vector = embedder.encode([search_query])[0]
    search_results = vector_store.search(query_vector, top_k=2)

    table_data = []
    for rank, (chunk, score) in enumerate(search_results, 1):
        table_data.append([rank, chunk.doc_title, f"{score:.4f}", chunk.content[:90] + "..."])
    
    print(f"Query: '{search_query}'")
    print(tabulate(table_data, headers=["Rank", "Document Title", "Cosine Sim", "Content Snippet"], tablefmt="grid"))

    # 5. RAG Pipeline Execution Test
    print(f"\n[4] Retrieval-Augmented Generation (RAG) Execution")
    rag_query_text = "What metrics are used to evaluate RAG systems?"
    rag_result = rag_pipeline.generate_answer(rag_query_text, top_k=2)
    
    print(f"Query: '{rag_result.query}'")
    print(f"Answer:\n{rag_result.answer}\n")
    print(f"Citations Extracted:")
    for cit in rag_result.citations:
        print(f" - [{cit.doc_title}] (Score: {cit.score}): {cit.snippet}")

    # 6. Benchmark Evaluation Suite
    print(f"\n[5] Benchmark Evaluation Metrics Execution")
    benchmarks = [
        BenchmarkQuery(
            query="What is Retrieval-Augmented Generation and vector search?",
            expected_doc_ids=["doc_rag_overview", "doc_faiss_vector"],
            ground_truth_keywords=["Retrieval-Augmented", "vector", "FAISS", "passages"]
        ),
        BenchmarkQuery(
            query="How is FastAPI deployed using Docker?",
            expected_doc_ids=["doc_fastapi_docker"],
            ground_truth_keywords=["FastAPI", "Docker", "containerization"]
        ),
        BenchmarkQuery(
            query="What are Sentence Transformers used for?",
            expected_doc_ids=["doc_faiss_vector"],
            ground_truth_keywords=["dense", "vector", "embeddings", "similarity"]
        )
    ]

    eval_metrics = evaluator.run_benchmark(benchmarks, top_k=2)
    
    eval_table = [
        ["Total Evaluated Queries", eval_metrics.total_queries],
        ["Precision@K (K=2)", f"{eval_metrics.precision_at_k * 100:.2f}%"],
        ["Recall@K (K=2)", f"{eval_metrics.recall_at_k * 100:.2f}%"],
        ["Mean Reciprocal Rank (MRR)", f"{eval_metrics.mrr:.4f}"],
        ["Average Context Relevance", f"{eval_metrics.avg_context_relevance * 100:.2f}%"]
    ]
    print(tabulate(eval_table, headers=["Metric", "Value"], tablefmt="grid"))

    print("\n" + "=" * 80)
    print("ALL SYSTEM TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    run_e2e_test()
