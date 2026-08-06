import uuid
from typing import List

from src.core.schema import Document, Chunk


class TextChunker:
    MIN_BREAK_RATIO = 0.5

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        chunks: List[Chunk] = []

        for doc in documents:
            doc_chunks = self._chunk_text(doc.content)

            for index, text_chunk in enumerate(doc_chunks):
                metadata = doc.metadata.copy()
                metadata["chunk_index"] = index

                chunks.append(
                    Chunk(
                        id=str(uuid.uuid4()),
                        doc_id=doc.id,
                        text=text_chunk,
                        metadata=metadata,
                    )
                )

        return chunks

    def _chunk_text(self, text: str) -> List[str]:
        if not text.strip():
            return []

        chunks: List[str] = []
        start = 0
        text_length = len(text)
        min_break = int(self.chunk_size * self.MIN_BREAK_RATIO)

        while start < text_length:
            end = min(start + self.chunk_size, text_length)

            if end < text_length:
                breakpoint = self._find_breakpoint(
                    text,
                    start,
                    end,
                    min_break,
                )

                if breakpoint != -1:
                    end = breakpoint

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end >= text_length:
                break

            start = max(0, end - self.chunk_overlap)

            while start > 0 and not text[start - 1].isspace():
                start -= 1

        return chunks

    def _find_breakpoint(
        self,
        text: str,
        start: int,
        end: int,
        min_break: int,
    ) -> int:
        separators = (
            "\n\n",
            "\n",
            ". ",
            "! ",
            "? ",
            ".\n",
            "!\n",
            "?\n",
        )

        for separator in separators:
            pos = text.rfind(separator, start, end)

            if pos != -1 and pos > start + min_break:
                return pos + len(separator)

        return -1