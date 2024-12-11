import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Optional


class HuggingFaceTransformersEmbeddings:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 512,
        strip_new_lines: bool = True,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.strip_new_lines = strip_new_lines

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)

        # Set to evaluation mode
        self.model.eval()

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self.strip_new_lines:
            texts = [text.replace("\n", " ") for text in texts]

        # Tokenize sentences
        encoded_input = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling
        sentence_embeddings = self._mean_pooling(
            model_output, encoded_input["attention_mask"]
        )

        # Normalize embeddings
        sentence_embeddings = torch.nn.functional.normalize(
            sentence_embeddings, p=2, dim=1
        )

        return sentence_embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


if __name__ == "__main__":
    # Ensure you have torch installed: pip install torch
    embedding_model = HuggingFaceTransformersEmbeddings()

    texts = [
        "Hugging Face provides a powerful transformer-based library.",
        "The embeddings are generated using the transformer models.",
    ]

    embeddings = embedding_model.embed_documents(texts)
    for idx, embedding in enumerate(embeddings):
        print(
            f"Document {idx + 1} embedding: {embedding[:5]}..."
        )  # Show first 5 values for brevity

    query = "What is Hugging Face?"
    query_embedding = embedding_model.embed_query(query)
    print(f"Query embedding: {query_embedding[:5]}...")
