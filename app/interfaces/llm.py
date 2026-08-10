from abc import ABC, abstractmethod
from typing import AsyncGenerator

class BaseLLMService(ABC):
    @abstractmethod
    def construct_prompt(self, query: str, context: str) -> str:
        pass

    @abstractmethod
    async def generate_answer(self, query: str, context: str) -> str:
        pass

    @abstractmethod
    async def stream_answer(self, query: str, context: str) -> AsyncGenerator[str, None]:
        pass
