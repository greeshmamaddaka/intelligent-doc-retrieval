# Intelligent Document Search & Retrieval System

An enterprise-grade **Document Search and Retrieval System** powered by **Python**, **Sentence Transformers**, **FAISS**, **FastAPI**, and **Docker**. 

This system extracts, preprocesses, and chunks unstructured text documents, builds dense vector embeddings, executes cosine vector similarity search, drives a Retrieval-Augmented Generation (RAG) pipeline with explicit citations, and calculates quantitative retrieval quality metrics (Precision@K, Recall@K, MRR).

---

## 🏛️ System Architecture

```
                                  +-----------------------+
                                  | Raw Text / PDF Docs   |
                                  +-----------+-----------+
                                              |
                                              v
                                  +-----------------------+
                                  | Ingestion & Chunking  |
                                  | (Recursive Splitter)  |
                                  +-----------+-----------+
                                              |
                                              v
                                  +-----------------------+
                                  | Embedding Model       |
                                  | (Sentence Transformers|
                                  +-----------+-----------+
                                              |
                                              v
                                  +-----------------------+
                                  | FAISS Vector Database |
                                  | (IndexFlatIP Cosine)  |
                                  +-----------+-----------+
                                              |
           +----------------------------------+----------------------------------+
           |                                                                     |
           v                                                                     v
+-----------------------+                                             +-----------------------+
|  Semantic Vector      |                                             | RAG Answer Synthesis  |
|  Search Endpoint      |                                             | & Citation Engine     |
+-----------------------+                                             +-----------------------+
```

---

## 🚀 Key Features

1. **Document Extraction & Cleaning**:
   - Normalizes text, cleans special HTML/control characters, and handles word-boundary-aware recursive sliding window chunking with configurable overlap.
2. **Dense Vector Similarity Search**:
   - Encodes text passages into 384-dimensional vector embeddings using Hugging Face `SentenceTransformers` (`all-MiniLM-L6-v2`).
   - Uses **FAISS** (`IndexFlatIP`) for sub-millisecond cosine vector similarity search.
3. **Retrieval-Augmented Generation (RAG)**:
   - Formats top-K retrieved context blocks into structured prompt contexts.
   - Generates grounded answers with explicit document citations, similarity scores, and fallback when context is missing.
4. **FastAPI Web Service**:
   - Asynchronous RESTful API endpoints for ingestion (`/documents/ingest`), search (`/search`), RAG querying (`/rag/query`), and benchmark evaluation (`/eval/benchmark`).
5. **Evaluation Suite**:
   - Calculates **Precision@K**, **Recall@K**, **Mean Reciprocal Rank (MRR)**, and **Context Relevance** across test queries.
6. **Production Dockerization**:
   - Multi-stage slim `Dockerfile` with non-root security principles and `docker-compose.yml` for cloud deployment.

---

## 📁 Repository Structure

```
intelligent_doc_retrieval/
├── app/
│   ├── __init__.py
│   ├── config.py          # System configurations and parameters
│   ├── ingest.py          # Document loader & recursive sliding-window chunker
│   ├── embeddings.py      # Sentence Transformers & FAISS vector search engine
│   ├── rag.py             # RAG retrieval & grounded answer generator
│   ├── main.py            # FastAPI REST API application & endpoints
│   └── eval.py            # Evaluation metrics engine (Precision@K, Recall@K, MRR)
├── data/                  # Vector index storage & document metadata
├── Dockerfile             # Multi-stage production container build
├── docker-compose.yml     # Docker Compose service definition
├── requirements.txt       # Project dependencies
├── test_system.py         # E2E test execution & evaluation script
└── README.md              # Project documentation
```

---

## 🛠️ Quickstart Guide

### 1. Local Setup
```bash
# Clone repository and create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Automated Integration Test & Benchmarks
```bash
python test_system.py
```

### 3. Launch FastAPI Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Access interactive OpenAPI documentation at: `http://localhost:8000/docs`

---

## 🐳 Docker Deployment

### Run with Docker Compose:
```bash
docker-compose up --build -d
```
Verify container status:
```bash
curl http://localhost:8000/health
```

---

## 📊 Sample API Endpoints & Request Payloads

### Ingest Documents (`POST /documents/ingest`)
```json
{
  "documents": [
    {
      "doc_id": "doc_01",
      "title": "FastAPI and RAG Architecture",
      "text": "Retrieval-Augmented Generation relies on vector databases like FAISS to retrieve top passages..."
    }
  ]
}
```

### RAG Query (`POST /rag/query`)
```json
{
  "query": "How does RAG use FAISS vector database?",
  "top_k": 3
}
```

---

## 📈 Evaluation Metrics Summary

| Metric | Description | Target Benchmark |
|---|---|---|
| **Precision@K** | Ratio of retrieved chunks in top-K that are relevant | > 75.0% |
| **Recall@K** | Ratio of relevant document chunks retrieved | > 80.0% |
| **MRR** | Mean Reciprocal Rank of first relevant passage | > 0.80 |
| **Context Relevance** | Semantic overlap score between query and retrieved context | > 70.0% |
