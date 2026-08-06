from typing import Any, Dict
from pydantic import BaseModel, Field

class Document(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Chunk(BaseModel):
    id: str
    doc_id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)