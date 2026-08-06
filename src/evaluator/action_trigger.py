from enum import Enum
from typing import List
import numpy as np

from configs.settings import settings
from src.evaluator.schema import EvaluatedDocument
from src.evaluator.schema import ConfidenceReport
from src.utils.logger import get_logger

logger = get_logger(__name__)

class RetrievalConfidenceEstimator:
    def compute(self, evaluated_docs):
        scores = [doc.relevance_score for doc in evaluated_docs]
        top = max(scores)
        avg = np.mean(scores)
        std = np.std(scores)
        high_docs = sum(score > 0.7 for score in scores)
        # confidence = (0.4 * top +0.3 * avg +0.2 * (high_docs / len(scores)) +0.1 * (1 - std))

        return ConfidenceReport(
            top_score=top,
            average_score=avg,
            std_score=std,
            supporting_docs=high_docs,
        )
            

class ActionDecision(str, Enum):

    CORRECT = "CORRECT"
    INCORRECT = "INCORRECT"
    AMBIGUOUS = "AMBIGUOUS"

class ActionTrigger:
    def __init__(
        self,
        upper_threshold: float = settings.crag_upper_threshold,
        lower_threshold: float = settings.crag_lower_threshold,
        avg_threshold: float = settings.crag_avg_threshold
    ):
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.avg_threshold = avg_threshold


    def decide(self, evaluated_docs: List[EvaluatedDocument]) -> ActionDecision:
        if not evaluated_docs:
            return ActionDecision.INCORRECT

        estimator = RetrievalConfidenceEstimator()
        report = estimator.compute(evaluated_docs)

        if report.top_score >= self.upper_threshold and report.average_score >= self.avg_threshold:
            return ActionDecision.CORRECT
        elif report.top_score < self.lower_threshold:
            return ActionDecision.INCORRECT
        else:
            return ActionDecision.AMBIGUOUS