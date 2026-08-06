from pydantic import BaseModel

class KnowledgeStrip(BaseModel):
    text: str
    score: float
    doc_id: str
    is_relevant: bool