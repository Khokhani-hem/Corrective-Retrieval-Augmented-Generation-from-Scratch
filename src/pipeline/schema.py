from typing import List
from pydantic import BaseModel
from src.evaluator.schema import EvaluatedDocument

class PipelineOutput(BaseModel):
    query: str
    response: str
    action: str
    context: str
    prompt: str
    retrieved_docs: List[str]
    filtered_docs: List[str]
    evaluated_doc: List[EvaluatedDocument]