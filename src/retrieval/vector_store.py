import json
from pathlib import Path
from typing import List, Tuple
import faiss
import numpy as np

from configs.settings import settings
from src.core.loader import DocumentLoader
from src.core.chunker import TextChunker
from src.core.schema import Chunk
from src.retrieval.embeddings import EmbeddingModel
from src.utils.logger import get_logger

logger = get_logger(__name__)

class VectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks: List[Chunk] = []

    def add_chunks(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) == 0:
            return
        if embeddings.shape[0] != len(chunks):
            raise ValueError("Number of chunks and embeddings must match.")
        if embeddings.shape[1] != self.dimension:
            raise ValueError("Embedding dimension mismatch.")
        
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[Chunk, float]]:
        if self.index.ntotal == 0:
            return []
        
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        scores, indices = self.index.search(query_embedding, top_k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx != -1 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        return results

    def save(self, index_dir: str | Path) -> None:
        path = Path(index_dir)
        path.mkdir(parents=True, exist_ok=True)
        
        faiss_path = path / "index.faiss"
        chunks_path = path / "chunks.json"
        
        faiss.write_index(self.index, str(faiss_path))
        
        chunks_data = [chunk.model_dump() for chunk in self.chunks]
        chunks_path.write_text(json.dumps(chunks_data, indent=2), encoding="utf-8")
        logger.info(f"Vector store saved to {path}")

    @classmethod
    def load(cls, index_dir: str | Path) -> "VectorStore":
        path = Path(index_dir)
        faiss_path = path / "index.faiss"
        chunks_path = path / "chunks.json"
        
        if not faiss_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(f"Index files not found at {path}")
            
        faiss_index = faiss.read_index(str(faiss_path))
        dimension = faiss_index.d
        
        store = cls(dimension=dimension)
        store.index = faiss_index
        
        chunks_data = json.loads(chunks_path.read_text(encoding="utf-8"))
        store.chunks = [Chunk(**data) for data in chunks_data]
        logger.info(f"Vector store loaded from {path}")
        return store

    @classmethod
    def load_or_create(cls, embedding_model: EmbeddingModel, force_rebuild: bool = False) -> "VectorStore":
        faiss_file = settings.index_dir / "index.faiss"
        chunks_file = settings.index_dir / "chunks.json"

        if faiss_file.exists() and chunks_file.exists() and not force_rebuild:
            logger.info(f"Loading existing vector store from {settings.index_dir}")
            return cls.load(settings.index_dir)

        logger.info("Building new vector store index from raw data...")
        raw_files = list(settings.raw_data_dir.glob("*.*"))
        if not raw_files:
            raise FileNotFoundError(f"No raw data files found in {settings.raw_data_dir}")

        documents = []
        for file_path in raw_files:
            loader = DocumentLoader(file_path)
            documents.extend(loader.load())

        chunker = TextChunker(chunk_size=300, chunk_overlap=50)
        chunks = chunker.chunk_documents(documents)

        store = cls(dimension=embedding_model.embedding_dimension)
        if chunks:
            chunk_texts = [c.text for c in chunks]
            embeddings = embedding_model.get_document_embeddings(chunk_texts)
            store.add_chunks(chunks, embeddings)
            store.save(settings.index_dir)

        return store