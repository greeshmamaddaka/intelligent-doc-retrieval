"""
Document Ingestion & Chunking Module.
Handles document extraction, text cleaning, and recursive sliding-window chunking.
"""
import re
import uuid
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class DocumentChunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str
    doc_title: str
    content: str
    start_char: int
    end_char: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

class TextCleaner:
    @staticmethod
    def clean(text: str) -> str:
        """Normalize whitespace, strip HTML tags, and clean special control characters."""
        text = re.sub(r'<[^>]+>', ' ', text)  # remove html
        text = re.sub(r'[\r\t]', ' ', text)   # remove tabs/returns
        text = re.sub(r'\s+', ' ', text)      # normalize multi-spaces
        return text.strip()

class RecursiveTextSplitter:
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 80):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str, doc_id: str, doc_title: str, metadata: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        """Sliding-window chunking preserving word boundaries."""
        cleaned_text = TextCleaner.clean(text)
        if not cleaned_text:
            return []
            
        chunks = []
        start = 0
        text_len = len(cleaned_text)
        meta = metadata or {}

        while start < text_len:
            end = start + self.chunk_size
            if end >= text_len:
                end = text_len
            else:
                # Break at nearest word boundary
                boundary = cleaned_text.rfind(' ', start, end)
                if boundary != -1 and boundary > start + self.chunk_size // 2:
                    end = boundary

            chunk_content = cleaned_text[start:end].strip()
            if chunk_content:
                chunks.append(DocumentChunk(
                    doc_id=doc_id,
                    doc_title=doc_title,
                    content=chunk_content,
                    start_char=start,
                    end_char=end,
                    metadata=meta
                ))

            if end == text_len:
                break
            # Advance start by chunk_size - overlap
            advance = max(1, (end - start) - self.chunk_overlap)
            start += advance

        return chunks

class DocumentIngestor:
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 80):
        self.splitter = RecursiveTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def process_document(self, doc_id: str, title: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        return self.splitter.split_text(text, doc_id=doc_id, doc_title=title, metadata=metadata)
