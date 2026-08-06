from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

from configs.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

class EmbeddingModel:
    def __init__(
        self,
        model_name: str = settings.embedding_model_name,
        device: str = settings.embedding_device,
        batch_size: int = settings.embedding_batch_size,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def get_query_embedding(self, query: str) -> np.ndarray:
        embedding = self.model.encode(
            query,
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.array(embedding, dtype=np.float32)

    def get_document_embeddings(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.array(embeddings, dtype=np.float32)

    @property
    def embedding_dimension(self) -> int:
        dim = self.model.get_embedding_dimension()
        if dim is None:
            raise ValueError("Could not determine embedding dimension from model.")
        return int(dim)