"""
Retrieval-Augmented Generation (RAG) Module.
Coordinates vector retrieval, context formatting, and grounded answer synthesis.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from app.ingest import DocumentChunk
from app.embeddings import VectorStoreEngine, EmbeddingModel

class Citation(BaseModel):
    chunk_id: str
    doc_title: str
    score: float
    snippet: str

class RAGResult(BaseModel):
    query: str
    answer: str
    retrieved_chunks_count: int
    citations: List[Citation]
    context_used: str

class RAGPipeline:
    def __init__(self, vector_store: VectorStoreEngine, embedder: EmbeddingModel):
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve_context(self, query: str, top_k: int = 3, threshold: float = 0.1) -> List[tuple[DocumentChunk, float]]:
        """Retrieve relevant document chunks for a query."""
        query_vector = self.embedder.encode([query])[0]
        results = self.vector_store.search(query_vector, top_k=top_k, threshold=threshold)
        return results

    def format_context(self, search_results: List[tuple[DocumentChunk, float]]) -> str:
        """Format retrieved chunks into structured context blocks."""
        if not search_results:
            return "No relevant context found."
            
        context_blocks = []
        for idx, (chunk, score) in enumerate(search_results, 1):
            block = (
                f"[Source {idx} | Document: '{chunk.doc_title}' | Similarity: {score:.3f}]\n"
                f"{chunk.content}"
            )
            context_blocks.append(block)
            
        return "\n\n".join(context_blocks)

    def generate_answer(self, query: str, top_k: int = 3, threshold: float = 0.1) -> RAGResult:
        """Execute complete RAG workflow: Retrieve context -> Synthesize answer."""
        search_results = self.retrieve_context(query, top_k=top_k, threshold=threshold)
        
        citations = []
        for chunk, score in search_results:
            citations.append(Citation(
                chunk_id=chunk.chunk_id,
                doc_title=chunk.doc_title,
                score=round(score, 4),
                snippet=chunk.content[:150] + ("..." if len(chunk.content) > 150 else "")
            ))

        context_str = self.format_context(search_results)

        if not search_results:
            answer = "I could not find sufficient information in the indexed documents to answer your query."
        else:
            # Extractive & Synthesized Answer Generation grounded in context
            matched_sentences = []
            query_words = set(query.lower().split()) - {"what", "is", "the", "a", "an", "how", "why", "in", "of", "and", "or", "to", "for"}
            
            for chunk, score in search_results:
                sentences = [s.strip() for s in chunk.content.split('.') if len(s.strip()) > 10]
                for sent in sentences:
                    sent_words = set(sent.lower().split())
                    if len(query_words.intersection(sent_words)) > 0 or score > 0.4:
                        matched_sentences.append((sent, chunk.doc_title))

            if matched_sentences:
                unique_sents = []
                seen = set()
                for s, title in matched_sentences:
                    if s not in seen:
                        seen.add(s)
                        unique_sents.append(f"{s}. (Source: '{title}')")
                
                synthesis = " ".join(unique_sents[:3])
                answer = f"Based on the retrieved documents:\n\n{synthesis}"
            else:
                top_doc = search_results[0][0]
                answer = f"Based on document '{top_doc.doc_title}':\n\n{top_doc.content[:300]}..."

        return RAGResult(
            query=query,
            answer=answer,
            retrieved_chunks_count=len(search_results),
            citations=citations,
            context_used=context_str
        )
