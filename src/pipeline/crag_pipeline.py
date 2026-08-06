from configs.settings import settings
from src.pipeline.schema import PipelineOutput
from src.retrieval.retriever import Retriever
from src.evaluator.retrieval_evaluator import RetrievalEvaluator
from src.evaluator.action_trigger import ActionTrigger, ActionDecision
from src.refinement.refiner import KnowledgeRefiner
from src.search.web_search import WebSearchModule
from src.generator.llm import Generator
from src.generator.formatter import ContextFormatter
from src.utils.logger import get_logger

logger = get_logger(__name__)

class CRAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        evaluator: RetrievalEvaluator,
        action_trigger: ActionTrigger,
        refiner: KnowledgeRefiner,
        web_search: WebSearchModule,
        generator: Generator,
        formatter: ContextFormatter
    ):
        self.retriever = retriever
        self.evaluator = evaluator
        self.action_trigger = action_trigger
        self.refiner = refiner
        self.web_search = web_search
        self.generator = generator
        self.formatter = formatter

    def run(self, query: str) -> PipelineOutput:
        logger.info(f"Processing query: {query}")
        
        retrieval_results = self.retriever.retrieve(query)
        evaluated_docs = self.evaluator.evaluate(query, retrieval_results)
        action = self.action_trigger.decide(evaluated_docs)
        
        logger.info(f"Action triggered: {action.value}")
        
        retrieved_docs_text = [res.retrieval_result.chunk.text for res in evaluated_docs]
        filtered_docs_text = [
            doc.retrieval_result.chunk.text 
            for doc in evaluated_docs 
            if doc.relevance_score >= settings.crag_lower_threshold
        ]
        
        context = ""
        
        if action == ActionDecision.CORRECT:
            context = self.refiner.refine(query, evaluated_docs)
        elif action == ActionDecision.INCORRECT:
            search_results = self.web_search.search(query)
            context = "\n".join([res.content for res in search_results])
        elif action == ActionDecision.AMBIGUOUS:
            refined_knowledge = self.refiner.refine(query, evaluated_docs)
            search_results = self.web_search.search(query)
            search_knowledge = "\n".join([res.content for res in search_results])
            context = f"{refined_knowledge}\n{search_knowledge}".strip()

        if len(context) > settings.max_context_length:
            context = context[:settings.max_context_length] + "..."

        prompt = self.formatter.format_prompt(query, context)
        generation_result = self.generator.generate(prompt)
        
        return PipelineOutput(
            query=query,
            response=generation_result.response,
            action=action.value,
            context=context,
            prompt=generation_result.prompt_used,
            retrieved_docs=retrieved_docs_text,
            filtered_docs=filtered_docs_text,
            evaluated_doc = evaluated_docs
        )