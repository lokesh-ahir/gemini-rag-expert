from typing import Any
from langchain_google_genai import ChatGoogleGenerativeAI
from src.interfaces.llm import LLMInterface


class GeminiLLM(LLMInterface):
    """Gemini LLM implementation"""

    def __init__(self, model: str, api_key: str, **kwargs):
        self.model = model
        self.api_key = api_key
        self._llm = None

    @property
    def llm(self):
        """Lazy load LLM"""
        if self._llm is None:
            self._llm = ChatGoogleGenerativeAI(
                model=self.model,
                google_api_key=self.api_key,
                temperature=0.7
            )
        return self._llm

    def invoke(self, prompt: str, **kwargs) -> str:
        """Invoke the LLM"""
        return self.llm.invoke(prompt, **kwargs)

    def stream(self, prompt: str, **kwargs):
        """Stream responses from the LLM"""
        return self.llm.stream(prompt, **kwargs)
