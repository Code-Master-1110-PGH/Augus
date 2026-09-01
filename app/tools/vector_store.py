"""Hybrid vector store with document upload and RAG support."""

from __future__ import annotations

import io
import re
import uuid
from collections import Counter
from typing import Any

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


class DocumentProcessor:
    """Handles text extraction from various file formats."""

    @staticmethod
    def extract_text_from_pdf(file_bytes: bytes) -> str:
        """Extract text from PDF file.
        
        Args:
            file_bytes: Raw PDF file content as bytes
            
        Returns:
            Extracted text from all PDF pages
            
        Raises:
            ValueError: If PDF parsing fails or pypdf not installed
        """
        if not PdfReader:
            raise ValueError(
                "pypdf is required for PDF processing. "
                "Install with: pip install pypdf"
            )
        
        try:
            pdf_file = io.BytesIO(file_bytes)
            reader = PdfReader(pdf_file)
            text_parts = []
            
            for page_num, page in enumerate(reader.pages, 1):
                page_text = page.extract_text()
                if page_text.strip():
                    # Add page marker for reference
                    text_parts.append(f"[Page {page_num}]\n{page_text}")
            
            return "\n".join(text_parts)
        except Exception as e:
            raise ValueError(f"Failed to parse PDF: {str(e)}")

    @staticmethod
    def extract_text_from_txt(file_bytes: bytes, encoding: str = "utf-8") -> str:
        """Extract text from TXT file.
        
        Args:
            file_bytes: Raw TXT file content as bytes
            encoding: Text encoding (default: utf-8)
            
        Returns:
            Decoded text content
            
        Raises:
            ValueError: If decoding fails
        """
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            # Fallback to latin-1 if utf-8 fails
            try:
                return file_bytes.decode("latin-1")
            except UnicodeDecodeError as e:
                raise ValueError(
                    f"Could not decode text file with utf-8 or latin-1: {str(e)}"
                )

    @staticmethod
    def extract_text_from_csv(file_bytes: bytes) -> str:
        """Extract text from CSV file.
        
        Args:
            file_bytes: Raw CSV file content as bytes
            
        Returns:
            CSV content as text
        """
        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return file_bytes.decode("latin-1")


class TextChunker:
    """Splits text into overlapping chunks of approximately target token count."""

    def __init__(self, chunk_size: int = 512, overlap: int = 100):
        """Initialize chunker.
        
        Args:
            chunk_size: Target number of tokens per chunk (~4 chars per token)
            overlap: Number of tokens to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        """Split text into overlapping chunks.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks with preserved context
        """
        # Approximate token count as words (1 word ≈ 1.3 tokens)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk_sentences = []
        current_size = 0
        
        overlap_sentences = []
        
        for sentence in sentences:
            sentence_size = len(sentence.split())
            
            # Check if adding this sentence exceeds chunk size
            if current_size + sentence_size > self.chunk_size and current_chunk_sentences:
                # Save current chunk
                chunk_text = " ".join(current_chunk_sentences)
                chunks.append(chunk_text)
                
                # Preserve overlap for context continuity
                words_to_overlap = sum(len(s.split()) for s in current_chunk_sentences)
                running_overlap = 0
                for s in reversed(current_chunk_sentences):
                    s_len = len(s.split())
                    if running_overlap + s_len <= self.overlap:
                        overlap_sentences.insert(0, s)
                        running_overlap += s_len
                    else:
                        break
                
                current_chunk_sentences = overlap_sentences.copy()
                current_size = sum(len(s.split()) for s in current_chunk_sentences)
                overlap_sentences = []
            
            current_chunk_sentences.append(sentence)
            current_size += sentence_size
        
        # Add final chunk
        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))
        
        return [chunk.strip() for chunk in chunks if chunk.strip()]


class HybridVectorStore:
    """In-memory dense-like token search with BM25-style lexical scoring.

    The interface is intentionally backend-neutral so Qdrant can be introduced
    behind the same contract without changing agent orchestration.
    """

    def __init__(self, embeddings: Any | None = None) -> None:
        self._documents: list[dict[str, Any]] = []
        self._embeddings = embeddings
        self._vectors: list[list[float]] = []

    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        self._documents.extend(documents)
        if self._embeddings:
            self._vectors.extend(self._embeddings.embed_documents([document.get("text", "") for document in documents]))

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def hybrid_search(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        """Rank documents with local dense similarity plus lexical overlap."""
        query_tokens = self._tokens(query)
        query_counts = Counter(query_tokens)
        query_vector = self._embeddings.embed_query(query) if self._embeddings else None
        scored = []
        for index, document in enumerate(self._documents):
            tokens = self._tokens(document.get("text", ""))
            counts = Counter(tokens)
            lexical = sum(min(counts[token], frequency) for token, frequency in query_counts.items())
            phrase = 0.25 if query.lower() in document.get("text", "").lower() else 0.0
            dense = 0.0
            if query_vector and index < len(self._vectors):
                dense = sum(left * right for left, right in zip(query_vector, self._vectors[index]))
            scored.append((lexical + phrase + dense, document))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [document for score, document in scored[:limit] if score > 0] or [
            document for _, document in scored[:limit]
        ]

    async def upload_documents(
        self,
        file_bytes: bytes,
        filename: str,
        company_ticker: str = "CUSTOM",
        chunk_size: int = 512,
        overlap: int = 100,
    ) -> dict[str, Any]:
        """Process and store uploaded document in vector store.
        
        Args:
            file_bytes: Raw file content
            filename: Original filename (used to determine file type)
            company_ticker: Company identifier for metadata
            chunk_size: Target tokens per chunk
            overlap: Token overlap between chunks
            
        Returns:
            Dictionary with processing results:
            {
                "status": "success",
                "filename": str,
                "ticker": str,
                "chunks_created": int,
                "documents_stored": int,
                "chunk_ids": list[str]
            }
            
        Raises:
            ValueError: If file type unsupported or processing fails
        """
        # Determine file type and extract text
        filename_lower = filename.lower()
        
        if filename_lower.endswith('.pdf'):
            text = DocumentProcessor.extract_text_from_pdf(file_bytes)
        elif filename_lower.endswith(('.txt', '.text')):
            text = DocumentProcessor.extract_text_from_txt(file_bytes)
        elif filename_lower.endswith('.csv'):
            text = DocumentProcessor.extract_text_from_csv(file_bytes)
        else:
            raise ValueError(
                f"Unsupported file type: {filename}. "
                "Supported: PDF, TXT, CSV"
            )
        
        if not text.strip():
            raise ValueError("Extracted text is empty.")
        
        # Chunk the text
        chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
        chunks = chunker.chunk(text)
        
        if not chunks:
            raise ValueError("No chunks created from document.")
        
        # Create documents with metadata
        documents = []
        chunk_ids = []
        
        for chunk_idx, chunk_text in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            chunk_ids.append(chunk_id)
            
            doc = {
                "id": chunk_id,
                "text": chunk_text,
                "source": filename,
                "source_type": "user_upload",
                "company_ticker": company_ticker.upper(),
                "chunk_index": chunk_idx,
                "total_chunks": len(chunks),
                "metadata": {
                    "filename": filename,
                    "ticker": company_ticker.upper(),
                    "chunk_number": chunk_idx + 1,
                    "total_chunks": len(chunks),
                },
            }
            documents.append(doc)
        
        # Add to vector store
        self.add_documents(documents)
        
        return {
            "status": "success",
            "filename": filename,
            "ticker": company_ticker.upper(),
            "chunks_created": len(chunks),
            "documents_stored": len(documents),
            "chunk_ids": chunk_ids,
        }

    def search_by_ticker(
        self, query: str, ticker: str, limit: int = 8
    ) -> list[dict[str, Any]]:
        """Search documents filtered by company ticker.
        
        Args:
            query: Search query
            ticker: Company ticker filter
            limit: Maximum results
            
        Returns:
            List of matching documents
        """
        # First do hybrid search on all documents
        all_results = self.hybrid_search(query, limit=limit * 2)
        
        # Filter by ticker
        filtered = [
            doc for doc in all_results
            if doc.get("company_ticker", "").upper() == ticker.upper()
        ]
        
        return filtered[:limit]
