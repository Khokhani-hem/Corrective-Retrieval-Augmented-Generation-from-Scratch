import torch
from transformers import GenerationConfig, pipeline

from configs.settings import settings
from src.generator.schema import GenerationResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Generator:
    def __init__(
        self,
        model_name: str = settings.generator_model_name,
        device: str = settings.generator_device,
    ) -> None:
        self.model_name = model_name

        if device.lower() == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device.lower()

        logger.info(f"Loading generator model '{self.model_name}' on '{self.device}'")

        self.pipe = self._load_pipeline()
        self.generation_config = self._build_generation_config()

    def _load_pipeline(self):
        pipeline_kwargs = {
            "task": "text-generation",
            "model": self.model_name,
            "clean_up_tokenization_spaces": False,
        }

        if self.device == "cuda":
            pipeline_kwargs.update(
                {
                    "device_map": "auto",
                    "dtype": torch.bfloat16,
                }
            )
        else:
            pipeline_kwargs.update(
                {
                    "device": -1, 
                    "dtype": torch.float32,
                }
            )

        return pipeline(**pipeline_kwargs)

    def _build_generation_config(self) -> GenerationConfig:
        config = GenerationConfig.from_model_config(self.pipe.model.config)

        config.max_new_tokens = settings.generator_max_tokens
        config.do_sample = True
        config.temperature = 0.6
        config.top_p = 0.9
        config.repetition_penalty = 1.1
        config.max_length = None

        return config

    def generate(self, prompt: str) -> GenerationResult:
        outputs = self.pipe(
            prompt,
            generation_config=self.generation_config,
            return_full_text=False,
            truncation=True,
        )

        generated_text = outputs[0]["generated_text"].strip()

        return GenerationResult(
            response=generated_text,
            prompt_used=prompt,
        )