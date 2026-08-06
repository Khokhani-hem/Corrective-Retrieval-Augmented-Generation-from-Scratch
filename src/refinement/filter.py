from typing import List
from sentence_transformers import CrossEncoder

from configs.settings import settings
from src.refinement.schema import KnowledgeStrip
from src.utils.logger import get_logger

logger = get_logger(__name__)

class StripFilter:
    def __init__(
        self,
        model_name: str = settings.evaluator_model_name,
        device: str = settings.evaluator_device,
        threshold: float = settings.strip_relevance_threshold,
    ):
        self.model_name = model_name
        self.device = device
        self.threshold = threshold
        logger.info(f"Loading strip filter model: {self.model_name} on {self.device}")
        self.model = CrossEncoder(self.model_name, device=self.device)

    def filter_strips(self, query: str, strips: List[KnowledgeStrip]) -> List[KnowledgeStrip]:
        if not strips:
            return []

        pairs = [[query, strip.text] for strip in strips]
        scores = self.model.predict(pairs, show_progress_bar=False)
        
        filtered = []
        for strip, score in zip(strips, scores):
            prob = float(1.0 / (1.0 + float(2.718281828459045 ** (-score))))
            strip.score = prob
            strip.is_relevant = prob >= self.threshold
            if strip.is_relevant:
                filtered.append(strip)
                
        return filtered