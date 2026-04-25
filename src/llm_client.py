from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from loguru import logger


class LLMClient:
    """Handles LLM initialization"""

    def __init__(
        self,
        provider: str = "openai",
        model_name: str = "gpt-4",
        temperature: float = 0.7
    ):
        """
        Initialize LLM client

        Args:
            provider: "openai" or "ollama"
            model_name: model name
            temperature: generation temperature
        """
        self.provider = provider.lower()
        self.model_name = model_name
        self.temperature = temperature

        logger.info(f"Initializing LLM: {provider} | {model_name}")

        if self.provider == "openai":
            self.llm = ChatOpenAI(
                model_name=model_name,
                temperature=temperature
            )

        elif self.provider == "ollama":
            self.llm = ChatOllama(
                model=model_name,
                temperature=temperature
            )

        else:
            raise ValueError(f"Unsupported provider: {provider}")

        logger.success(f"LLM initialized: {provider} | {model_name}")

    def get_llm(self):
        """Get the LLM object"""
        return self.llm