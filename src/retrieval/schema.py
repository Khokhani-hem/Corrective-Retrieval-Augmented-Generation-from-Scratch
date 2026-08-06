from pydantic import BaseModel
from src.core.schema import Chunk

class RetrievalResult(BaseModel):
    chunk: Chunk
    score: float