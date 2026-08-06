from typing import List, Dict, Any
from ddgs import DDGS

from configs.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

class SearchAPIWrapper:
    def __init__(self, top_k: int = settings.search_top_k):
        self.top_k = top_k

    def search(self, query: str) -> List[Dict[str, Any]]:
        results = []
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=self.top_k):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", "")
                    })
        except Exception as e:
            logger.error(f"Search API error for query '{query}': {e}")
        return results