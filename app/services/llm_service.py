import httpx
import json
from typing import AsyncGenerator
from app.core.config import settings
from app.core.logging import get_logger
from app.interfaces.llm import BaseLLMService

logger = get_logger(__name__)

class OllamaLLMService(BaseLLMService):
    def __init__(self, base_url: str = settings.OLLAMA_BASE_URL, model: str = settings.OLLAMA_MODEL):
        self.base_url = base_url
        self.model = model
        logger.info(f"LLM service initialized (provider: Ollama, model: {self.model}, url: {self.base_url})")

    def construct_prompt(self, query: str, context: str) -> str:
        logger.debug(f"Constructing prompt (context_length={len(context)} chars)")
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
        
        logger.info(f"Sending generation request to Ollama (model: {self.model}, prompt_length: {len(prompt)} chars)")
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(f"{self.base_url}/api/generate", json=payload)
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("response", "No answer generated.")
                    logger.info(f"LLM generation complete (response_length: {len(answer)} chars)")
                    return answer
                else:
                    logger.error(f"Ollama returned HTTP {response.status_code} — ensure model '{self.model}' is pulled")
                    return f"[Ollama Error {response.status_code}]: Make sure model '{self.model}' is pulled in Ollama."
        except Exception as e:
            logger.error(f"Ollama unreachable at {self.base_url} — {type(e).__name__}: {e}")
            logger.warning(f"To enable LLM generation, start Ollama with: ollama run {self.model}")
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
        logger.info(f"Starting streaming generation (model: {self.model})")
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=2.0)) as client:
                async with client.stream("POST", f"{self.base_url}/api/generate", json=payload) as response:
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                yield data.get("response", "")
                            except Exception:
                                pass
            logger.info("Streaming generation complete")
        except Exception as e:
            logger.error(f"Ollama streaming failed — {type(e).__name__}: {e}")
            yield f"[ApexGuard Standalone Mode]: Ollama connection error - {str(e)}"

llm_service = OllamaLLMService()
