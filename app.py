import sys
from pathlib import Path
import streamlit as st

root_path = Path(__file__).resolve().parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from configs.settings import settings
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import VectorStore
from src.retrieval.retriever import Retriever
from src.evaluator.retrieval_evaluator import RetrievalEvaluator
from src.evaluator.action_trigger import ActionTrigger
from src.refinement.refiner import KnowledgeRefiner
from src.search.web_search import WebSearchModule
from src.generator.llm import Generator
from src.generator.formatter import ContextFormatter
from src.pipeline.crag_pipeline import CRAGPipeline
from src.retrieval.bm25_store import BM25Store

st.set_page_config(page_title="CRAG Pipeline", layout="wide")

@st.cache_resource
def load_pipeline():
    embedding_model = EmbeddingModel()
    vector_store = VectorStore.load_or_create(embedding_model)
    bm25_store = BM25Store.load()
    retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        bm25_store= bm25_store
    )
    evaluator = RetrievalEvaluator()
    action_trigger = ActionTrigger()
    refiner = KnowledgeRefiner()
    web_search = WebSearchModule()
    generator = Generator()
    formatter = ContextFormatter()
    
    pipeline = CRAGPipeline(
        retriever=retriever,
        evaluator=evaluator,
        action_trigger=action_trigger,
        refiner=refiner,
        web_search=web_search,
        generator=generator,
        formatter=formatter
    )
    return pipeline, embedding_model, evaluator

st.title("Corrective Retrieval Augmented Generation")

st.sidebar.header("Document Management")
uploaded_files = st.sidebar.file_uploader(
    "Upload Knowledge Documents",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True
)

if uploaded_files:
    settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
    for uploaded_file in uploaded_files:
        clean_name = uploaded_file.name
        for ext in [".txt", ".pdf", ".md"]:
            double_ext = f"{ext}{ext}"
            if clean_name.lower().endswith(double_ext):
                clean_name = clean_name[:-len(ext)]
        file_path = settings.raw_data_dir / clean_name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"Saved {len(uploaded_files)} file(s) to raw data directory.")

if st.sidebar.button("Rebuild Vector Store Index"):
    with st.spinner("Indexing documents..."):
        _, embedding_model, _ = load_pipeline()
        VectorStore.load_or_create(embedding_model, force_rebuild=True)
        BM25Store.save()
        st.cache_resource.clear()
        st.sidebar.success("Index rebuilt successfully!")

try:
    pipeline, _, evaluator = load_pipeline()
    pipeline_ready = True
except Exception as e:
    st.error(f"Error initializing pipeline: {e}")
    pipeline_ready = False

if pipeline_ready:
    query = st.text_input("Ask a question:")
    if st.button("Run CRAG Pipeline") and query:
        with st.spinner("Running CRAG evaluation and generation..."):
            output = pipeline.run(query)
            
            st.markdown(f"### Triggered Action: `{output.action}`")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Retrieved Documents")
                for i, doc in enumerate(output.retrieved_docs):
                    st.info(f"**Doc {i+1}:** {doc}")
                    
            with col2:
                st.markdown("#### Filtered / Refined Knowledge")
                for i, doc in enumerate(output.filtered_docs):
                    st.success(f"**Refined {i+1}:** {doc}")
                    
            st.markdown("### Generated Response")
            st.write(output.response)