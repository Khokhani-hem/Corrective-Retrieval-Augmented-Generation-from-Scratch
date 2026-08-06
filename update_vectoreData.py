from src.retrieval.bm25_store import BM25Store
from src.retrieval.vector_store import VectorStore
from src.retrieval.embeddings import EmbeddingModel

model = EmbeddingModel()
VectorStore.load_or_create(model,force_rebuild=True)
BM25Store.save()