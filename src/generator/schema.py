from pydantic import BaseModel

class GenerationResult(BaseModel):
    response: str
    prompt_used: str