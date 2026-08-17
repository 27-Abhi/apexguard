from fastapi import APIRouter
from app.schemas.query import RAGResponse
from tests.run_eval import RAGEvaluator

router = APIRouter(prefix="/eval", tags=["Phase 3 Evaluation Engine"])

@router.post("/run")
async def run_evaluation_benchmark(k: int = 3):
    """Triggers the Phase 3 RAG & LLM evaluation harness and outputs benchmark metrics."""
    evaluator = RAGEvaluator()
    report = await evaluator.run_evaluation(k=k)
    return {
        "status": "success",
        "benchmark_questions": len(evaluator.dataset),
        "results": report
    }
