import re
import time
from typing import List, Dict, Any, Optional
import httpx
from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.router import (
    RouteType,
    ComplexityLevel,
    RouteDecision,
    SmartQueryRequest,
    SmartQueryResponse,
)
from app.schemas.query import SourceResponse
from app.services.llm_service import llm_service
from app.services.rag_service import retriever_service

logger = get_logger(__name__)

# 1. Conversational greetings, persona, chitchat, and everyday queries (No RAG needed)
GENERAL_CONVERSATIONAL_PATTERNS = [
    re.compile(
        r"^(hi|hello|hey|greetings|good\s+(morning|afternoon|evening|day)|howdy|yo|sup|"
        r"how\s+are\s+you|who\s+are\s+you|what\s+is\s+your\s+name|thanks?|thank\s+you|"
        r"bye|goodbye|see\s+ya|nice\s+to\s+meet\s+you)[\s!.,?]*$",
        re.IGNORECASE
    ),
    re.compile(
        r"\b(what\s+is\s+the\s+time|what\s+time\s+is\s+it|what\s+is\s+the\s+date|what\s+day\s+is\s+today|"
        r"what\s+is\s+the\s+weather|tell\s+me\s+a\s+joke|how\s+old\s+are\s+you|who\s+created\s+you|"
        r"who\s+made\s+you|what\s+can\s+you\s+do|help\s+me|ping|test)\b",
        re.IGNORECASE
    ),
    re.compile(
        r"^(translate|summarize|paraphrase|spellcheck|format|convert\s+to\s+json|"
        r"capitalize|reverse|repeat|echo)\b",
        re.IGNORECASE
    )
]

# 2. Complex Reasoning, Coding, Mathematics, System Design
REASONING_PATTERNS = [
    re.compile(r"\b(def|class|lambda|return|import|async\s+def|function)\b"),
    re.compile(r"```"),
    re.compile(r"\b(python|javascript|typescript|c\+\+|rust|sql|bash|dockerfile|regex|html|css)\b", re.IGNORECASE),
    re.compile(r"\b(write\s+(a\s+)?(code|script|program|function|algorithm|query|regex))\b", re.IGNORECASE),
    re.compile(r"\b(debug|fix\s+the\s+bug|optimize|refactor|time\s+complexity|big\s+o)\b", re.IGNORECASE),
    re.compile(r"\b(calculate|solve|equation|integral|derivative|matrix|probability|proof)\b", re.IGNORECASE),
    re.compile(r"\b(step[- ]by[- ]step\s+(reasoning|analysis|proof)|deduce|evaluate\s+the\s+tradeoffs)\b", re.IGNORECASE),
    re.compile(r"\b(compare\s+and\s+contrast\s+in\s+detail|design\s+architecture\s+for|system\s+design)\b", re.IGNORECASE),
]

# 3. Domain-specific Knowledge, Indexed Documents & Architecture RAG
DOMAIN_KNOWLEDGE_PATTERNS = [
    re.compile(r"\b(apexguard|qdrant|fastembed|bm25|flashrank|rerank|rag|embedding|vector\s+store)\b", re.IGNORECASE),
    re.compile(r"\b(documentation|manual|guide|config|settings|env|upload|collection|payload|schema)\b", re.IGNORECASE),
    re.compile(r"\b(api\s+key|endpoint|port|localhost|host|url|jwt|secret|minio|gateway|slowapi|middleware)\b", re.IGNORECASE),
    re.compile(r"\b(in\s+the\s+document|from\s+the\s+pdf|according\s+to\s+the\s+file|in\s+the\s+upload|in\s+docs|indexed\s+data)\b", re.IGNORECASE),
    re.compile(r"\b(how\s+to\s+configure|how\s+does\s+.*work\s+in\s+apexguard|what\s+is\s+the\s+default\s+value)\b", re.IGNORECASE),
]


class QueryRouterService:
    """
    Intelligent Model Router for Phase 5.
    Evaluates incoming queries using generalized sub-millisecond heuristic intent
    and complexity analysis to route between:
      1. Direct Fast Model (Low latency, zero retrieval cost for general/conversational prompts)
      2. Hybrid RAG Pipeline (Domain documents, architectural knowledge, indexed enterprise data)
      3. Complex Reasoning Model (Coding, mathematics, multi-step algorithmic analysis)
    """

    def __init__(
        self,
        fast_model: str = settings.ROUTER_FAST_MODEL,
        reasoning_model: str = settings.ROUTER_REASONING_MODEL,
        rag_model: str = settings.OLLAMA_MODEL
    ):
        self.fast_model = fast_model
        self.reasoning_model = reasoning_model
        self.rag_model = rag_model
        logger.info(
            f"QueryRouterService initialized (Fast: {self.fast_model}, "
            f"Reasoning: {self.reasoning_model}, RAG: {self.rag_model})"
        )

    def classify(self, query: str, override_route: Optional[RouteType] = None) -> RouteDecision:
        t0 = time.perf_counter()
        clean_query = query.strip()

        # Handle manual override
        if override_route:
            if override_route == RouteType.DIRECT_FAST:
                model = self.fast_model
                complexity = ComplexityLevel.SIMPLE
                reasoning = "Manual override to DIRECT_FAST route."
                requires_rag = False
                latency_tier = "low"
                cost_tier = "free/local"
            elif override_route == RouteType.COMPLEX_REASONING:
                model = self.reasoning_model
                complexity = ComplexityLevel.COMPLEX
                reasoning = "Manual override to COMPLEX_REASONING route."
                requires_rag = False
                latency_tier = "high"
                cost_tier = "medium"
            else:
                model = self.rag_model
                complexity = ComplexityLevel.MODERATE
                reasoning = "Manual override to KNOWLEDGE_RAG route."
                requires_rag = True
                latency_tier = "medium"
                cost_tier = "low"

            dt_ms = (time.perf_counter() - t0) * 1000.0
            return RouteDecision(
                query=query,
                route=override_route,
                model=model,
                complexity=complexity,
                confidence=1.0,
                reasoning=reasoning,
                estimated_latency_tier=latency_tier,
                estimated_cost_tier=cost_tier,
                requires_rag=requires_rag,
                classification_time_ms=round(dt_ms, 3)
            )

        words = clean_query.split()
        word_count = len(words)

        # 1. Check general conversational, greetings, persona, and time/date/everyday questions
        if any(p.search(clean_query) for p in GENERAL_CONVERSATIONAL_PATTERNS):
            dt_ms = (time.perf_counter() - t0) * 1000.0
            return RouteDecision(
                query=query,
                route=RouteType.DIRECT_FAST,
                model=self.fast_model,
                complexity=ComplexityLevel.SIMPLE,
                confidence=0.92,
                reasoning="General conversational or everyday query; routed directly to fast local model without retrieval.",
                estimated_latency_tier="low",
                estimated_cost_tier="free/local",
                requires_rag=False,
                classification_time_ms=round(dt_ms, 3)
            )

        # 2. Check Reasoning / Code / Logic patterns
        reasoning_matches = sum(1 for p in REASONING_PATTERNS if p.search(clean_query))

        # 3. Check Domain / Document Knowledge patterns
        domain_matches = sum(1 for p in DOMAIN_KNOWLEDGE_PATTERNS if p.search(clean_query))

        if reasoning_matches >= 2 or (reasoning_matches >= 1 and word_count >= 6):
            confidence = min(0.95, 0.70 + (reasoning_matches * 0.10))
            dt_ms = (time.perf_counter() - t0) * 1000.0
            return RouteDecision(
                query=query,
                route=RouteType.COMPLEX_REASONING,
                model=self.reasoning_model,
                complexity=ComplexityLevel.COMPLEX,
                confidence=round(confidence, 2),
                reasoning=f"Detected complex reasoning or programming intent ({reasoning_matches} match(es)).",
                estimated_latency_tier="high",
                estimated_cost_tier="medium",
                requires_rag=False,
                classification_time_ms=round(dt_ms, 3)
            )

        if domain_matches >= 1:
            confidence = min(0.95, 0.75 + (domain_matches * 0.08))
            dt_ms = (time.perf_counter() - t0) * 1000.0
            return RouteDecision(
                query=query,
                route=RouteType.KNOWLEDGE_RAG,
                model=self.rag_model,
                complexity=ComplexityLevel.MODERATE,
                confidence=round(confidence, 2),
                reasoning=f"Detected domain knowledge / documentation retrieval intent ({domain_matches} match(es)).",
                estimated_latency_tier="medium",
                estimated_cost_tier="low",
                requires_rag=True,
                classification_time_ms=round(dt_ms, 3)
            )

        # 4. For general short or open-ended questions that lack domain entities:
        # Default to DIRECT_FAST to keep latency sub-150ms rather than invoking heavy vector retrieval.
        if word_count <= 8:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            return RouteDecision(
                query=query,
                route=RouteType.DIRECT_FAST,
                model=self.fast_model,
                complexity=ComplexityLevel.SIMPLE,
                confidence=0.85,
                reasoning="General question without domain entities; routed to fast direct LLM.",
                estimated_latency_tier="low",
                estimated_cost_tier="free/local",
                requires_rag=False,
                classification_time_ms=round(dt_ms, 3)
            )

        # 5. For longer queries with no explicit domain or code matches:
        # Route to fast model for general answering
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return RouteDecision(
            query=query,
            route=RouteType.DIRECT_FAST,
            model=self.fast_model,
            complexity=ComplexityLevel.SIMPLE,
            confidence=0.70,
            reasoning="Standard prompt without specific domain or heavy logic dependencies.",
            estimated_latency_tier="low",
            estimated_cost_tier="free/local",
            requires_rag=False,
            classification_time_ms=round(dt_ms, 3)
        )

    async def _call_ollama_direct(self, prompt: str, model: str, temperature: float = 0.7) -> str:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature}
        }
        try:
            async with httpx.AsyncClient(timeout=settings.GATEWAY_TIMEOUT_SEC) as client:
                resp = await client.post(f"{settings.OLLAMA_BASE_URL}/api/generate", json=payload)
                if resp.status_code == 200:
                    return resp.json().get("response", "")
                else:
                    return f"[Ollama Direct Error {resp.status_code}]: Model {model} generation failed."
        except Exception as e:
            logger.warning(f"Direct LLM call to {model} failed: {e}")
            return f"[ApexGuard Standalone Direct Mode]: Generated response placeholder for '{prompt[:50]}...' ({str(e)})"

    async def execute_route(self, request: SmartQueryRequest) -> SmartQueryResponse:
        t_start = time.perf_counter()
        decision = self.classify(request.query, override_route=request.override_route)
        logger.info(f"Routing query '{request.query[:60]}' -> {decision.route.value} (model: {decision.model})")

        sources: List[SourceResponse] = []
        answer = ""

        if decision.route == RouteType.DIRECT_FAST:
            prompt = f"You are a helpful and concise AI assistant.\nUser: {request.query}\nAnswer:"
            answer = await self._call_ollama_direct(prompt, decision.model, temperature=request.temperature or 0.7)

        elif decision.route == RouteType.COMPLEX_REASONING:
            prompt = (
                f"You are an advanced reasoning and engineering AI assistant. "
                f"Provide a rigorous, well-structured, step-by-step solution.\n\n"
                f"Task:\n{request.query}\n\n"
                f"Detailed Solution:"
            )
            answer = await self._call_ollama_direct(prompt, decision.model, temperature=request.temperature or 0.7)

        elif decision.route == RouteType.KNOWLEDGE_RAG:
            # Execute Hybrid RAG
            search_results = retriever_service.retrieve_hybrid(
                query=request.query,
                top_k=request.top_k or settings.HYBRID_TOP_K
            )
            sources = [
                SourceResponse(
                    text=res["text"],
                    score=float(res.get("score", 0.0)),
                    metadata=res.get("metadata", {})
                )
                for res in search_results
            ]
            context = retriever_service.format_context(search_results)
            answer = await llm_service.generate_answer(query=request.query, context=context)

        total_latency = (time.perf_counter() - t_start) * 1000.0
        return SmartQueryResponse(
            query=request.query,
            answer=answer,
            routing=decision,
            sources=sources,
            latency_ms=round(total_latency, 2)
        )


router_service = QueryRouterService()
