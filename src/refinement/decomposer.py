from typing import List
from configs.settings import settings
from src.evaluator.schema import EvaluatedDocument
from src.refinement.schema import KnowledgeStrip

class KnowledgeDecomposer:
    def __init__(self, strip_length: int = settings.strip_length):
        self.strip_length = strip_length

    def decompose(self, evaluated_docs: List[EvaluatedDocument]) -> List[KnowledgeStrip]:
        strips = []
        for doc in evaluated_docs:
            doc_id = doc.retrieval_result.chunk.id
            text = doc.retrieval_result.chunk.text
            tokens = text.split()
            
            for i in range(0, len(tokens), self.strip_length):
                strip_tokens = tokens[i : i + self.strip_length]
                if not strip_tokens:
                    continue
                strip_text = " ".join(strip_tokens)
                strips.append(
                    KnowledgeStrip(
                        text=strip_text,
                        score=0.0,
                        doc_id=doc_id,
                        is_relevant=False,
                    )
                )
        return strips