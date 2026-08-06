from collections import defaultdict
from typing import List, Tuple
from src.core.schema import Chunk

def reciprocal_rank_fusion(
    dense_results: List[Tuple[Chunk, float]],
    sparse_results: List[Tuple[Chunk, float]],
    k: int = 60,
) -> List[Tuple[Chunk, float]]:

    rrf_scores = defaultdict(float)
    chunk_lookup = {}

    dense_weight = 0.7
    sparse_weight = 0.3

    for rank, (chunk, _) in enumerate(dense_results, start=1):
        rrf_scores[chunk.id] += dense_weight * 1.0 / (k + rank)
        chunk_lookup[chunk.id] = chunk

    for rank, (chunk, _) in enumerate(sparse_results, start=1):
        rrf_scores[chunk.id] += sparse_weight * 1.0 / (k + rank)
        chunk_lookup[chunk.id] = chunk

    ranked = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    return [
        (chunk_lookup[chunk_id], score)
        for chunk_id, score in ranked
    ]