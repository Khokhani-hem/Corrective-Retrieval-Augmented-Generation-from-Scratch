from typing import List
from src.search.schema import SearchResult
from src.search.rewriter import QueryRewriter
from src.search.api_wrapper import SearchAPIWrapper
from src.search.extractor import PageExtractor
from src.search.selector import ResultSelector
from src.utils.logger import get_logger

logger = get_logger(__name__)

class WebSearchModule:
    def __init__(
        self,
        rewriter: QueryRewriter | None = None,
        api_wrapper: SearchAPIWrapper | None = None,
        extractor: PageExtractor | None = None,
        selector: ResultSelector | None = None,
    ):
        self.rewriter = rewriter or QueryRewriter()
        self.api_wrapper = api_wrapper or SearchAPIWrapper()
        self.extractor = extractor or PageExtractor()
        self.selector = selector or ResultSelector()

    def search(self, query: str) -> List[SearchResult]:
        queries = self.rewriter.rewrite(query)
        all_raw_results = []
        seen_urls = set()
        
        for q in queries:
            raw_res = self.api_wrapper.search(q)
            for item in raw_res:
                url = item.get("url")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_raw_results.append(item)
                    
        extracted_results = self.extractor.extract_all(all_raw_results)
        selected_results = self.selector.select(extracted_results)
        return selected_results