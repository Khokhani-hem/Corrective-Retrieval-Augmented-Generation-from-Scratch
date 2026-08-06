from typing import List
import numpy as np
from sentence_transformers import CrossEncoder

from configs.settings import settings
from src.retrieval.schema import RetrievalResult
from src.evaluator.schema import EvaluatedDocument
from src.utils.logger import get_logger

logger = get_logger(__name__)

class RetrievalEvaluator:
    def __init__(
        self,
        model_name: str = settings.evaluator_model_name,
        device: str = settings.evaluator_device,
    ):
        self.model_name = model_name
        self.device = device
        logger.info(f"Loading retrieval evaluator model: {self.model_name} on {self.device}")
        self.model = CrossEncoder(self.model_name, device=self.device)

    def evaluate(
        self, query: str, results: List[RetrievalResult]
    ) -> List[EvaluatedDocument]:
        if not results:
            return []

        pairs = [[query, res.chunk.text] for res in results]
        raw_scores = self.model.predict(pairs, show_progress_bar=False)
        probabilities = self._compute_confidence(raw_scores)

        evaluated_docs = [
            EvaluatedDocument(
                retrieval_result=res,
                relevance_score=float(prob),
            )
            for res, prob in zip(results, probabilities)
        ]

        evaluated_docs = sorted(
        evaluated_docs,
        key=lambda x: x.relevance_score,
        reverse=True,)[:3]

        return evaluated_docs

    def _compute_confidence(self, scores: np.ndarray) -> np.ndarray:
        if isinstance(scores, (float, int)):
            scores = np.array([scores], dtype=np.float32)
        else:
            scores = np.array(scores, dtype=np.float32)
        return 1.0 / (1.0 + np.exp(-scores))