from typing import Optional
from fastapi import APIRouter, Query
from tests.run_eval import RAGEvaluator

router = APIRouter(prefix="/eval", tags=["Phase 3 Evaluation Engine"])

@router.post("/run")
async def run_evaluation_benchmark(
    k: int = Query(3, ge=1, le=20, description="Retrieved chunks per question."),
    max_questions: Optional[int] = Query(
        5,
        ge=1,
        description="Maximum benchmark questions to evaluate. Use 45 for the full dataset."
    ),
    strategy: str = Query(
        "dense",
        pattern="^(dense|hybrid|hybrid_reranked|all)$",
        description="Retrieval strategy to evaluate."
    ),
    include_generation: bool = Query(
        False,
        description="Run Ollama answer generation and semantic answer scoring."
    )
):
    """Triggers the Phase 3 RAG & LLM evaluation harness and outputs benchmark metrics."""
    evaluator = RAGEvaluator()
    strategies = None if strategy == "all" else [strategy]
    report = await evaluator.run_evaluation(
        k=k,
        max_questions=max_questions,
        strategies=strategies,
        include_generation=include_generation
    )
    return {
        "status": "success",
        "benchmark_questions": len(evaluator.dataset),
        "questions_evaluated": min(max_questions or len(evaluator.dataset), len(evaluator.dataset)),
        "strategies_evaluated": [result["strategy"] for result in report],
        "include_generation": include_generation,
        "results": report
    }
