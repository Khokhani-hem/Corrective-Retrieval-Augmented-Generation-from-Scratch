import torch
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    log_level: str = "INFO"

    base_dir: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = base_dir / "data"
    raw_data_dir: Path = data_dir / "raw"
    processed_data_dir: Path = data_dir / "processed"
    index_dir: Path = data_dir / "index"
    
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    embedding_model_name: str = "BAAI/bge-base-en-v1.5"
    embedding_device: str = device_str
    embedding_batch_size: int = 32

    evaluator_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    evaluator_device: str = device_str

    #Decides Action label like CORRECT,INCORRECT or AMBIGUOUS. Check src.evaluator.action_trigger.py
    crag_upper_threshold: float = 0.85
    crag_lower_threshold: float = 0.40
    crag_avg_threshold: float = 0.75
    
    strip_length: int = 64
    strip_relevance_threshold: float = 0.5
    
    search_top_k: int = 3
    search_timeout: int = 5
    
    generator_model_name: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    generator_device: str = device_str
    generator_max_tokens: int = 128
    max_context_length: int = 2048

    hf_token: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()