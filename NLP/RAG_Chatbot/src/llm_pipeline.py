"""Minimal QA wrapper using Hugging Face pipeline."""
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline


class LlamaRAG:
    def __init__(self, model_name: str, cache_dir: str = None, model_dir: str = None):
        model_path = model_dir or model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_path, cache_dir=cache_dir)
        self.qa_pipeline = pipeline(
            "question-answering",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
        )

    def generate_response(self, question: str, contexts: List[str]) -> str:
        if not contexts:
            return "Je n'ai rien trouvé dans la base de connaissances."
        merged_context = "\n\n".join(contexts)[:1500]
        result = self.qa_pipeline(question=question, context=merged_context)
        return result["answer"]
