import httpx
import json
from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
import logging
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/gateway", tags=["Gateway"])

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = settings.OLLAMA_MODEL
    messages: List[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

def is_retryable_error(exception):
    if isinstance(exception, httpx.HTTPStatusError):
        return exception.response.status_code in (429, 502, 503, 504)
    if isinstance(exception, httpx.RequestError):
        return True
    return False

@retry(
    stop=stop_after_attempt(settings.GATEWAY_MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.RequestError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True
)
async def call_llm_provider(payload: dict) -> httpx.Response:
    timeout = httpx.Timeout(settings.GATEWAY_TIMEOUT_SEC, connect=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        # Convert OpenAI format to Ollama format
        ollama_payload = {
            "model": payload.get("model", settings.OLLAMA_MODEL),
            "messages": [{"role": m["role"], "content": m["content"]} for m in payload.get("messages", [])],
            "stream": payload.get("stream", False),
            "options": {}
        }
        if payload.get("temperature") is not None:
            ollama_payload["options"]["temperature"] = payload["temperature"]
        if payload.get("max_tokens") is not None:
            ollama_payload["options"]["num_predict"] = payload["max_tokens"]
            
        url = f"{settings.OLLAMA_BASE_URL}/api/chat"
        logger.info(f"Forwarding chat request to {url}")
        
        response = await client.post(url, json=ollama_payload)
        response.raise_for_status()
        return response

async def call_llm_provider_stream(payload: dict):
    timeout = httpx.Timeout(settings.GATEWAY_TIMEOUT_SEC, connect=5.0)
    client = httpx.AsyncClient(timeout=timeout)
    
    ollama_payload = {
        "model": payload.get("model", settings.OLLAMA_MODEL),
        "messages": [{"role": m["role"], "content": m["content"]} for m in payload.get("messages", [])],
        "stream": True,
        "options": {}
    }
    if payload.get("temperature") is not None:
        ollama_payload["options"]["temperature"] = payload["temperature"]
    if payload.get("max_tokens") is not None:
        ollama_payload["options"]["num_predict"] = payload["max_tokens"]
        
    url = f"{settings.OLLAMA_BASE_URL}/api/chat"
    
    # Not using Tenacity for streaming yield since retrying a partial stream is complex
    # Retry logic here could wrap the initial connection phase
    try:
        async with client.stream("POST", url, json=ollama_payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        chunk = {
                            "id": "chatcmpl-stream",
                            "object": "chat.completion.chunk",
                            "model": ollama_payload["model"],
                            "choices": [{
                                "index": 0,
                                "delta": {"content": data.get("message", {}).get("content", "")},
                                "finish_reason": "stop" if data.get("done") else None
                            }]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                        if data.get("done"):
                            yield "data: [DONE]\n\n"
                    except json.JSONDecodeError:
                        pass
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    finally:
        await client.aclose()


from app.core.middleware import limiter
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

def verify_gateway_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme != "Bearer" or credentials.credentials != settings.GATEWAY_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

@router.post("/chat/completions", dependencies=[Depends(verify_gateway_token)])
@limiter.limit(settings.GATEWAY_RATE_LIMIT)
async def chat_completions(request: Request, body: ChatCompletionRequest):
    """
    OpenAI compatible chat completions endpoint.
    Protected by Bearer Token Authentication (middleware).
    Supports model='auto' or 'smart-router' for intelligent dynamic routing.
    """
    from app.services.router_service import router_service

    payload = body.dict()
    routing_metadata = None

    if body.model.lower() in ("auto", "smart-router"):
        user_message = next((m.content for m in reversed(body.messages) if m.role == "user"), "")
        decision = router_service.classify(user_message)
        payload["model"] = decision.model
        routing_metadata = decision.dict()
        logger.info(f"Auto-routed chat request to model '{decision.model}' (reason: {decision.reasoning})")

    if body.stream:
        return StreamingResponse(
            call_llm_provider_stream(payload),
            media_type="text/event-stream"
        )
    else:
        try:
            response = await call_llm_provider(payload)
            data = response.json()
            
            # Format to OpenAI
            import time
            resp_dict = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": data.get("model", payload.get("model")),
                "choices": [{
                    "index": 0,
                    "message": data.get("message", {}),
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                    "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
                }
            }
            if routing_metadata:
                resp_dict["routing"] = routing_metadata
            return resp_dict
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"Provider error: {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Gateway error: {str(e)}")
