from pydantic import BaseModel
from src.retrieval.schema import RetrievalResult

class EvaluatedDocument(BaseModel):
    retrieval_result: RetrievalResult
    relevance_score: float

class ConfidenceReport(BaseModel):
    top_score: float
    average_score: float
    std_score: float
    supporting_docs: int