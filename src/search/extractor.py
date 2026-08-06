from typing import List, Dict, Any
import requests
from bs4 import BeautifulSoup

from configs.settings import settings
from src.search.schema import SearchResult
from src.utils.logger import get_logger

logger = get_logger(__name__)

class PageExtractor:
    def __init__(self, timeout: int = settings.search_timeout):
        self.timeout = timeout
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

    def extract_all(self, raw_results: List[Dict[str, Any]]) -> List[SearchResult]:
        extracted = []
        for item in raw_results:
            url = item.get("url", "")
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            content = snippet
            
            if url:
                try:
                    resp = requests.get(
                        url,
                        headers=self.headers,
                        timeout=self.timeout
                    )
                    if resp.status_code == 200:
                        soup = BeautifulSoup(resp.text, "html.parser")
                        for tag in soup(["script", "style", "nav", "footer", "header"]):
                            tag.decompose()
                        text = soup.get_text(separator=" ")
                        cleaned_text = " ".join(text.split())
                        if cleaned_text:
                            content = cleaned_text
                except Exception as e:
                    logger.debug(f"Failed to fetch content from {url}: {e}")
            
            extracted.append(
                SearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    content=content
                )
            )
        return extracted