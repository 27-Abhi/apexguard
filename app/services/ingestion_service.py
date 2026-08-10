from typing import List
from pypdf import PdfReader
import docx2txt
from app.core.logging import get_logger

logger = get_logger(__name__)

class DocumentIngestionService:
    @staticmethod
    def extract_text_from_file(file_path: str, filename: str) -> str:
        ext = filename.split(".")[-1].lower() if "." in filename else ""
        logger.info(f"Extracting text from '{filename}' (type: .{ext}, path: {file_path})")
        
        if ext == "pdf":
            reader = PdfReader(file_path)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
            logger.info(f"Extracted {len(text)} chars from {len(reader.pages)} PDF pages")
            return text
        elif ext in ["docx", "doc"]:
            text = docx2txt.process(file_path)
            logger.info(f"Extracted {len(text)} chars from DOCX")
            return text
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            logger.info(f"Read {len(text)} chars from plain text file")
            return text

class DocumentChunkerService:
    @staticmethod
    def fixed_size_chunking(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        if not text:
            return []
        chunks = []
        start = 0
        text_length = len(text)
        step = chunk_size - overlap
        if step <= 0:
            step = chunk_size
            
        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= text_length:
                break
            start += step
        return chunks

    @staticmethod
    def recursive_character_chunking(
        text: str, 
        chunk_size: int = 500, 
        overlap: int = 50, 
        separators: List[str] = ["\n\n", "\n", ". ", " ", ""]
    ) -> List[str]:
        def _split_text(text_to_split: str, current_separators: List[str]) -> List[str]:
            if len(text_to_split) <= chunk_size or not current_separators:
                return [text_to_split]
            
            sep = current_separators[0]
            next_seps = current_separators[1:]
            
            if sep == "":
                return DocumentChunkerService.fixed_size_chunking(text_to_split, chunk_size, overlap)
                
            splits = text_to_split.split(sep)
            chunks = []
            current_chunk = ""
            
            for piece in splits:
                if len(current_chunk) + len(piece) + len(sep) <= chunk_size:
                    current_chunk += (sep if current_chunk else "") + piece
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    if len(piece) > chunk_size and next_seps:
                        chunks.extend(_split_text(piece, next_seps))
                        current_chunk = ""
                    else:
                        current_chunk = piece
            if current_chunk:
                chunks.append(current_chunk)
            return chunks

        raw_chunks = _split_text(text, separators)
        return [c.strip() for c in raw_chunks if c.strip()]

    @staticmethod
    def chunk_document(
        text: str, 
        strategy: str = "recursive", 
        chunk_size: int = 500, 
        overlap: int = 50
    ) -> List[str]:
        logger.info(f"Chunking document (strategy: {strategy}, chunk_size: {chunk_size}, overlap: {overlap}, text_length: {len(text)})")
        if strategy == "fixed":
            chunks = DocumentChunkerService.fixed_size_chunking(text, chunk_size, overlap)
        else:
            chunks = DocumentChunkerService.recursive_character_chunking(text, chunk_size, overlap)
        logger.info(f"Chunking complete → {len(chunks)} chunks produced")
        return chunks
