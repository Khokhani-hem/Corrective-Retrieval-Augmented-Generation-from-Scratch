from typing import List
from src.refinement.schema import KnowledgeStrip

class KnowledgeRecomposer:
    def recompose(self, filtered_strips: List[KnowledgeStrip]) -> str:
        if not filtered_strips:
            return ""
            
        sorted_strips = sorted(
            filtered_strips,
            key=lambda x: x.score,
            reverse=True,
        )
        
        return " ".join(strip.text for strip in sorted_strips)