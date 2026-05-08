from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from loguru import logger

class LLMClient:
    """Handles LLM initialization with support for explicit API keys."""

    def __init__(
        self,
        provider: str = "openai",
        model_name: str = "gpt-4",
        temperature: float = 0.7,
        api_key: Optional[str] = None  # Added api_key as an optional parameter
    ):
        """
        Initialize LLM client

        Args:
            provider: "openai" or "ollama"
            model_name: model name
            temperature: generation temperature
            api_key: OpenAI API key (required if provider is "openai")
        """
        self.provider = provider.lower()
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key

        logger.info(f"Initializing LLM: {provider} | {model_name}")

        if self.provider == "openai":
            # Ensure the API key is provided for OpenAI
            if not self.api_key:
                logger.error("Attempted to initialize OpenAI without an API key.")
                raise ValueError("An API key is required for the OpenAI provider.")

            self.llm = ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                openai_api_key=self.api_key  # Pass the key directly to LangChain
            )

        elif self.provider == "ollama":
            # Ollama is local, usually doesn't require an API key
            self.llm = ChatOllama(
                model=model_name,
                temperature=temperature
            )

        else:
            raise ValueError(f"Unsupported provider: {provider}")

        logger.success(f"LLM initialized: {provider} | {model_name}")

    def get_llm(self):
        """Get the LangChain LLM object"""
        return self.llm

# Example Usage:
# client = LLMClient(provider="openai", api_key="sk-...")
# llm = client.get_llm()