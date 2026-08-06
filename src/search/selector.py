from typing import List
from src.search.schema import SearchResult

class ResultSelector:
    def select(self, results: List[SearchResult], max_results: int = 3) -> List[SearchResult]:
        valid_results = [r for r in results if r.content.strip()]
        return valid_results[:max_results]