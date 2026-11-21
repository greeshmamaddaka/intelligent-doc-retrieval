"""
Evaluation Module for Retrieval & RAG System.
Computes Precision@K, Recall@K, MRR, and Context Relevance Metrics.
"""
import numpy as np
from typing import List, Dict, Any, Set
from pydantic import BaseModel
from app.rag import RAGPipeline

class BenchmarkQuery(BaseModel):
    query: str
    expected_doc_ids: List[str]
    ground_truth_keywords: List[str]

class EvaluationMetrics(BaseModel):
    total_queries: int
    precision_at_k: float
    recall_at_k: float
    mrr: float
    avg_context_relevance: float

class RetrievalEvaluator:
    def __init__(self, rag_pipeline: RAGPipeline):
        self.rag_pipeline = rag_pipeline

    def evaluate_query(self, benchmark: BenchmarkQuery, top_k: int = 3) -> Dict[str, float]:
        """Evaluate a single query against expected documents and keywords."""
        search_results = self.rag_pipeline.retrieve_context(benchmark.query, top_k=top_k, threshold=0.0)
        
        retrieved_doc_ids = [chunk.doc_id for chunk, _ in search_results]
        
        # Calculate Precision@K and Recall@K based on document IDs
        expected_set = set(benchmark.expected_doc_ids)
        retrieved_hits = [1 if doc_id in expected_set else 0 for doc_id in retrieved_doc_ids]
        
        precision = sum(retrieved_hits) / top_k if top_k > 0 else 0.0
        matching_docs = len(set(retrieved_doc_ids).intersection(expected_set))
        recall = matching_docs / len(expected_set) if expected_set else 0.0

        # Calculate Reciprocal Rank (MRR component)
        rr = 0.0
        for rank, hit in enumerate(retrieved_hits, 1):
            if hit == 1:
                rr = 1.0 / rank
                break

        # Calculate Context Keyword Relevance
        context_text = " ".join([chunk.content.lower() for chunk, _ in search_results])
        if benchmark.ground_truth_keywords:
            kw_hits = sum(1 for kw in benchmark.ground_truth_keywords if kw.lower() in context_text)
            relevance = kw_hits / len(benchmark.ground_truth_keywords)
        else:
            relevance = 1.0 if precision > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "reciprocal_rank": rr,
            "relevance": relevance
        }

    def run_benchmark(self, benchmarks: List[BenchmarkQuery], top_k: int = 3) -> EvaluationMetrics:
        """Run complete evaluation suite over a set of benchmark queries."""
        if not benchmarks:
            return EvaluationMetrics(
                total_queries=0, precision_at_k=0.0, recall_at_k=0.0, mrr=0.0, avg_context_relevance=0.0
            )

        precisions = []
        recalls = []
        rrs = []
        relevances = []

        for bm in benchmarks:
            res = self.evaluate_query(bm, top_k=top_k)
            precisions.append(res["precision"])
            recalls.append(res["recall"])
            rrs.append(res["reciprocal_rank"])
            relevances.append(res["relevance"])

        return EvaluationMetrics(
            total_queries=len(benchmarks),
            precision_at_k=round(float(np.mean(precisions)), 4),
            recall_at_k=round(float(np.mean(recalls)), 4),
            mrr=round(float(np.mean(rrs)), 4),
            avg_context_relevance=round(float(np.mean(relevances)), 4)
        )
