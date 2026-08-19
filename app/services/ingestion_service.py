import re
from enum import Enum
from typing import List, Callable, Optional
from pypdf import PdfReader
import docx2txt
import numpy as np
from app.core.logging import get_logger

logger = get_logger(__name__)


class ChunkingStrategy(str, Enum):
    FIXED = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    SENTENCE = "sentence"  # Added as an additional boundary strategy
    SECTION = "section"    # Section-aware chunking for structured documents


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
        elif ext in ["md", "markdown"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            logger.info(f"Read {len(text)} chars from Markdown file")
            return text
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            logger.info(f"Read {len(text)} chars from plain text file")
            return text


class DocumentChunkerService:
    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        """Splits raw text into sentences using sentence boundary detection."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

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
    def sentence_chunking(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Groups full sentences together up to chunk_size boundary."""
        sentences = DocumentChunkerService._split_into_sentences(text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                current_chunk += (" " if current_chunk else "") + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    @staticmethod
    def semantic_chunking(
        text: str,
        chunk_size: int = 500,
        overlap: int = 50,
        embedding_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
        similarity_threshold: float = 0.70
    ) -> List[str]:
        """
        Splits text into sentences and groups them based on semantic distance.
        A new chunk is created when similarity drops below threshold or max length is reached.
        """
        sentences = DocumentChunkerService._split_into_sentences(text)
        if not sentences:
            return []

        # If no embedding function is supplied or document is too short, fall back to recursive
        if not embedding_fn or len(sentences) <= 1:
            logger.warning("Embedding function omitted or single sentence; falling back to recursive chunking.")
            return DocumentChunkerService.recursive_character_chunking(text, chunk_size, overlap)

        try:
            # Generate embeddings for each sentence
            embeddings = embedding_fn(sentences)

            # Calculate cosine similarities between adjacent sentence vectors
            def cosine_similarity(v1, v2):
                vec1, vec2 = np.array(v1), np.array(v2)
                norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
                return float(np.dot(vec1, vec2) / norm) if norm > 0 else 0.0

            similarities = [
                cosine_similarity(embeddings[i], embeddings[i + 1])
                for i in range(len(embeddings) - 1)
            ]

            chunks = []
            current_sentences = [sentences[0]]
            current_len = len(sentences[0])

            for idx, sim in enumerate(similarities):
                next_sentence = sentences[idx + 1]
                next_len = len(next_sentence)

                # Split if sentence similarity is below threshold OR adding it exceeds max chunk size
                if sim < similarity_threshold or (current_len + next_len + 1 > chunk_size):
                    chunks.append(" ".join(current_sentences))
                    current_sentences = [next_sentence]
                    current_len = next_len
                else:
                    current_sentences.append(next_sentence)
                    current_len += next_len + 1

            if current_sentences:
                chunks.append(" ".join(current_sentences))

            return chunks
        except Exception as e:
            logger.error(f"Error during semantic chunking: {e}. Falling back to recursive chunking.")
            return DocumentChunkerService.recursive_character_chunking(text, chunk_size, overlap)

    @staticmethod
    def section_aware_chunking(
        text: str,
        chunk_size: int = 500,
        overlap: int = 50
    ) -> List[dict]:
        """
        Splits document by sections (markdown headers) while keeping headings,
        tables, code blocks, and procedures intact. Returns list of dicts with 'text' and 'section'.
        """
        if not text.strip():
            return []

        lines = text.split("\n")
        sections = []
        current_heading = "General"
        current_lines = []

        header_pattern = re.compile(r'^(#{1,6})\s+(.+)$')

        for line in lines:
            header_match = header_pattern.match(line.strip())
            if header_match:
                if current_lines:
                    sections.append((current_heading, "\n".join(current_lines)))
                    current_lines = []
                current_heading = header_match.group(2).strip()
                current_lines.append(line)
            else:
                current_lines.append(line)

        if current_lines:
            sections.append((current_heading, "\n".join(current_lines)))

        chunks_with_meta = []

        for heading, section_text in sections:
            # Skip Table of Contents section as it dilutes vector embeddings with links
            if heading.lower() in ["table of contents", "toc"]:
                continue

            blocks = []
            sec_lines = section_text.split("\n")
            i = 0
            n = len(sec_lines)

            while i < n:
                line = sec_lines[i]
                stripped = line.strip()

                if stripped.startswith("```"):
                    code_block = [line]
                    i += 1
                    while i < n:
                        code_block.append(sec_lines[i])
                        if sec_lines[i].strip().startswith("```"):
                            i += 1
                            break
                        i += 1
                    blocks.append("\n".join(code_block))
                    continue

                if "|" in stripped and (stripped.startswith("|") or stripped.endswith("|")):
                    table_block = [line]
                    i += 1
                    while i < n and "|" in sec_lines[i].strip():
                        table_block.append(sec_lines[i])
                        i += 1
                    blocks.append("\n".join(table_block))
                    continue

                if re.match(r'^(\d+\.|\-|\*)\s+', stripped):
                    list_block = [line]
                    i += 1
                    while i < n and (re.match(r'^(\d+\.|\-|\*)\s+', sec_lines[i].strip()) or (sec_lines[i].startswith(" ") or sec_lines[i].startswith("\t"))):
                        list_block.append(sec_lines[i])
                        i += 1
                    blocks.append("\n".join(list_block))
                    continue

                if not stripped:
                    i += 1
                    continue

                para_block = [line]
                i += 1
                while i < n:
                    nxt = sec_lines[i].strip()
                    if not nxt or nxt.startswith("```") or ("|" in nxt and (nxt.startswith("|") or nxt.endswith("|"))) or re.match(r'^(#{1,6}|\d+\.|\-|\*)\s+', nxt):
                        break
                    para_block.append(sec_lines[i])
                    i += 1
                blocks.append("\n".join(para_block))

            curr_chunk_blocks = []
            curr_len = 0

            for b in blocks:
                b_len = len(b)
                if b_len > chunk_size:
                    if curr_chunk_blocks:
                        chunk_content = "\n\n".join(curr_chunk_blocks).strip()
                        if heading != "General" and not chunk_content.startswith("#"):
                            chunk_content = f"[{heading}]\n{chunk_content}"
                        chunks_with_meta.append({"text": chunk_content, "section": heading, "parent_text": section_text[:2000]})
                        curr_chunk_blocks = []
                        curr_len = 0

                    sub_chunks = DocumentChunkerService.recursive_character_chunking(b, chunk_size, overlap)
                    for sc in sub_chunks:
                        sc_content = sc
                        if heading != "General" and not sc_content.startswith("#"):
                            sc_content = f"[{heading}]\n{sc_content}"
                        chunks_with_meta.append({"text": sc_content, "section": heading, "parent_text": section_text[:2000]})
                elif curr_len + b_len + 2 <= chunk_size:
                    curr_chunk_blocks.append(b)
                    curr_len += b_len + 2
                else:
                    chunk_content = "\n\n".join(curr_chunk_blocks).strip()
                    if heading != "General" and not chunk_content.startswith("#"):
                        chunk_content = f"[{heading}]\n{chunk_content}"
                    chunks_with_meta.append({"text": chunk_content, "section": heading, "parent_text": section_text[:2000]})
                    curr_chunk_blocks = [b]
                    curr_len = b_len

            if curr_chunk_blocks:
                chunk_content = "\n\n".join(curr_chunk_blocks).strip()
                if heading != "General" and not chunk_content.startswith("#"):
                    chunk_content = f"[{heading}]\n{chunk_content}"
                chunks_with_meta.append({"text": chunk_content, "section": heading, "parent_text": section_text[:2000]})

        if not chunks_with_meta:
            rec_chunks = DocumentChunkerService.recursive_character_chunking(text, chunk_size, overlap)
            chunks_with_meta = [{"text": c, "section": "General", "parent_text": text[:2000]} for c in rec_chunks]

        return chunks_with_meta

    @classmethod
    def chunk_document_with_metadata(
        cls,
        text: str,
        strategy: str = ChunkingStrategy.SECTION,
        chunk_size: int = 500,
        overlap: int = 50,
        embedding_fn: Optional[Callable[[List[str]], List[List[float]]]] = None
    ) -> List[dict]:
        """Returns structured list of dicts: [{"text": ..., "section": ...}, ...]"""
        strat_str = str(strategy).lower() if strategy else ChunkingStrategy.SECTION.value

        if strat_str == ChunkingStrategy.SECTION.value:
            return cls.section_aware_chunking(text, chunk_size, overlap)
        else:
            chunks = cls.chunk_document(text, strategy, chunk_size, overlap, embedding_fn)
            return [{"text": c, "section": "General"} for c in chunks]

    @classmethod
    def chunk_document(
        cls, 
        text: str, 
        strategy: str = ChunkingStrategy.RECURSIVE, 
        chunk_size: int = 500, 
        overlap: int = 50,
        embedding_fn: Optional[Callable[[List[str]], List[List[float]]]] = None
    ) -> List[str]:
        """
        Main entry point for chunking. Dispatches to the appropriate chunker based on strategy.
        """
        logger.info(f"Chunking document (strategy: {strategy}, chunk_size: {chunk_size}, overlap: {overlap}, text_length: {len(text)})")
        
        # Strategy Dispatcher Map
        strategy_map = {
            ChunkingStrategy.FIXED.value: lambda: cls.fixed_size_chunking(text, chunk_size, overlap),
            ChunkingStrategy.RECURSIVE.value: lambda: cls.recursive_character_chunking(text, chunk_size, overlap),
            ChunkingStrategy.SENTENCE.value: lambda: cls.sentence_chunking(text, chunk_size, overlap),
            ChunkingStrategy.SECTION.value: lambda: [c["text"] for c in cls.section_aware_chunking(text, chunk_size, overlap)],
            ChunkingStrategy.SEMANTIC.value: lambda: cls.semantic_chunking(
                text, chunk_size, overlap, embedding_fn=embedding_fn
            ),
        }

        # Select strategy or fallback to recursive
        chunker_func = strategy_map.get(str(strategy).lower(), strategy_map[ChunkingStrategy.RECURSIVE.value])
        chunks = chunker_func()

        logger.info(f"Chunking complete → {len(chunks)} chunks produced")
        return chunks