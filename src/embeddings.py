"""Embeddings using HuggingFace sentence-transformers."""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings



class HFEmbeddings:
    """HuggingFace embeddings wrapper."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs: dict = None,
        encode_kwargs: dict = None,
    ):
        self.model_name = model_name
        self.model_kwargs = model_kwargs or {"device": "cpu"}
        self.encode_kwargs = encode_kwargs or {"normalize_embeddings": True}

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs=self.model_kwargs,
            encode_kwargs=self.encode_kwargs,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embeddings.embed_query(text)
