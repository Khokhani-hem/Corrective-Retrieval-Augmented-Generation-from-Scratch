from rank_bm25 import BM25Okapi
from typing import List,Tuple
import pickle
from src.core.schema import Chunk
from pathlib import Path
from configs.settings import settings
import json

from src.core.schema import Chunk

class BM25Store:

    def __init__(self):
        self.bm25 = None
        self.chunks = []
        self.tokenized_corpus = []

    def build(self, chunks: List[object]) -> None:
        self.chunks = chunks
        self.tokenized_corpus = [
            chunk.text.lower().split()
            for chunk in chunks
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:

        if self.bm25 is None:
            raise RuntimeError("BM25 index has not been built.")

        query_tokens = query.lower().split()

        scores = self.bm25.get_scores(query_tokens)

        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )

        results = []
        for idx,score in ranked[:top_k]:
            if idx != -1 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))


        return results

    @classmethod
    def save(cls) -> None:

        chunks_file = settings.index_dir / "chunks.json"
        bm25_file = settings.index_dir / "bm25_model.pkl"
        if not chunks_file.exists():
            raise FileNotFoundError("First Run The Normal Vector Store")

        chunks_data = json.loads(chunks_file.read_text(encoding="utf-8"))
        chunkes = [Chunk(**data) for data in chunks_data]

        store = cls()
        store.build(chunks=chunkes)

        with open(str(bm25_file), "wb") as f:
                    pickle.dump(store.bm25, f)


    @classmethod
    def load(cls) -> "BM25Store":
        chunks_path = settings.index_dir / "chunks.json"
        bm25_file = settings.index_dir / "bm25_model.pkl"

        store = BM25Store()
        chunks_data = json.loads(chunks_path.read_text(encoding="utf-8"))
        chunkes = [Chunk(**data) for data in chunks_data]

        with open(str(bm25_file), "rb") as f:
            bm25 = pickle.load(f)

        store.bm25 = bm25
        store.chunks = chunkes

        return store
