import httpx
import json
from typing import AsyncGenerator
from app.core.config import settings
from app.interfaces.llm import BaseLLMService

class OllamaLLMService(BaseLLMService):
    def __init__(self, base_url: str = settings.OLLAMA_BASE_URL, model: str = settings.OLLAMA_MODEL):
        self.base_url = base_url
        self.model = model

    def construct_prompt(self, query: str, context: str) -> str:
        return f"""You are ApexGuard RAG Assistant, an accurate AI answering system.
Answer the user query based ONLY on the provided context facts. If the context does not contain sufficient information, state that clearly.

--- CONTEXT START ---
{context}
--- CONTEXT END ---

User Question: {query}

Answer:"""

    async def generate_answer(self, query: str, context: str) -> str:
        prompt = self.construct_prompt(query, context)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(f"{self.base_url}/api/generate", json=payload)
                if response.status_code == 200:
                    data = response.json()
                    return data.get("response", "No answer generated.")
                else:
                    return f"[Ollama Error {response.status_code}]: Make sure model '{self.model}' is pulled in Ollama."
        except Exception as e:
            return (
                f"[ApexGuard Standalone Mode]: Ollama service at {self.base_url} was unreachable ({str(e)}). "
                f"Retrieved context successfully! (To enable live LLM generation, start Ollama with 'ollama run {self.model}')"
            )

    async def stream_answer(self, query: str, context: str) -> AsyncGenerator[str, None]:
        prompt = self.construct_prompt(query, context)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True
        }
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", f"{self.base_url}/api/generate", json=payload) as response:
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                yield data.get("response", "")
                            except Exception:
                                pass
        except Exception as e:
            yield f"[ApexGuard Standalone Mode]: Ollama connection error - {str(e)}"

llm_service = OllamaLLMService()
