from typing import List
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import VectorStore
from src.retrieval.schema import RetrievalResult
from src.utils.logger import get_logger
from src.retrieval.rrf import reciprocal_rank_fusion
from src.retrieval.bm25_store import BM25Store

logger = get_logger(__name__)

class Retriever:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        bm25_store :BM25Store, 
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.bm25_store = bm25_store

    def retrieve(self, query: str) -> List[RetrievalResult]:

        query_embedding = self.embedding_model.get_query_embedding(query)

        dense_result = self.vector_store.search(query_embedding)
        sparse_result = self.bm25_store.search(query)

        search_results = reciprocal_rank_fusion(dense_result , sparse_result)
        results = [
            RetrievalResult(chunk=chunk, score=score)
            for chunk, score in search_results
        ]
        return results