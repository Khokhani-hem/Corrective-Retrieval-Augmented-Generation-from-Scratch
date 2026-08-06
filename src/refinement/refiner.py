from typing import List
from src.evaluator.schema import EvaluatedDocument
from src.refinement.decomposer import KnowledgeDecomposer
from src.refinement.filter import StripFilter
from src.refinement.recomposer import KnowledgeRecomposer
from src.utils.logger import get_logger

logger = get_logger(__name__)

class KnowledgeRefiner:
    def __init__(
        self,
        decomposer: KnowledgeDecomposer | None = None,
        strip_filter: StripFilter | None = None,
        recomposer: KnowledgeRecomposer | None = None,
    ):
        self.decomposer = decomposer or KnowledgeDecomposer()
        self.strip_filter = strip_filter or StripFilter()
        self.recomposer = recomposer or KnowledgeRecomposer()

    def refine(self, query: str, evaluated_docs: List[EvaluatedDocument]) -> str:
        if not evaluated_docs:
            return ""
            
        strips = self.decomposer.decompose(evaluated_docs)
        filtered_strips = self.strip_filter.filter_strips(query, strips)
        refined_knowledge = self.recomposer.recompose(filtered_strips)
        return refined_knowledge