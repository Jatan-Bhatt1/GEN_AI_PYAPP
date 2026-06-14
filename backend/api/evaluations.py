"""
Evaluations API — Endpoints to run and view evaluations.

Endpoints:
  POST   /api/evaluations/hallucination    — Check if answer is grounded
  POST   /api/evaluations/score            — Rate answer quality
  POST   /api/evaluations/evaluate-rag     — Run full evaluation on a RAG query
  GET    /api/evaluations/runs             — View recent LLM call metrics
  GET    /api/evaluations/stats            — Aggregate usage statistics
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from loguru import logger

from backend.evaluations.hallucination import check_hallucination
from backend.evaluations.scoring import score_answer
from backend.evaluations.tracing import get_run_log, get_aggregate_stats
from backend.chains.rag_chain import invoke_rag
from backend.rag.retriever import search_documents

router = APIRouter(prefix="/api/evaluations", tags=["Evaluations"])


# ─── Models ───────────────────────────────────────────────────────────────────

class HallucinationRequest(BaseModel):
    answer: str = Field(..., description="The AI answer to check")
    context: str = Field(..., description="The source context used to generate the answer")


class ScoreRequest(BaseModel):
    question: str = Field(..., description="The original question")
    answer: str = Field(..., description="The AI answer to score")
    context: str = Field(default="", description="Source context (optional, for RAG answers)")


class EvaluateRAGRequest(BaseModel):
    question: str = Field(..., description="Question to ask the RAG system")
    session_id: str = Field(default="eval_session", description="Session ID")


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/hallucination", summary="Check if an answer is grounded in context")
async def check_hallucination_endpoint(request: HallucinationRequest):
    """
    Run the LLM-as-Judge hallucination check.

    Determines if every factual claim in `answer` is supported by `context`.

    **Use case:** After a RAG query, pass the answer + retrieved chunks as context.
    """
    result = check_hallucination(
        answer=request.answer,
        context=request.context,
    )
    return result


@router.post("/score", summary="Rate an answer on quality dimensions")
async def score_answer_endpoint(request: ScoreRequest):
    """
    Rate an answer on:
    - **Relevance** (1-5): Does it answer the question?
    - **Completeness** (1-5): Is it thorough?
    - **Accuracy** (1-5): Is it factually correct?

    Returns a `quality_score` (1.0-5.0) plus strengths, weaknesses, and suggestions.
    """
    result = score_answer(
        question=request.question,
        answer=request.answer,
        context=request.context,
    )
    return result


@router.post("/evaluate-rag", summary="Run a RAG query and automatically evaluate the result")
async def evaluate_rag_endpoint(request: EvaluateRAGRequest):
    """
    **End-to-end RAG evaluation in one call:**

    1. Runs the RAG query → gets answer + sources
    2. Retrieves the context chunks used
    3. Checks for hallucinations (grounded check)
    4. Scores the answer quality (relevance, completeness, accuracy)
    5. Returns everything in a single response

    This is the most useful endpoint for quality monitoring.
    """
    logger.info(f"Full RAG evaluation | question='{request.question[:60]}'")

    # Step 1: Get the RAG answer
    try:
        rag_result = invoke_rag(
            session_id=request.session_id,
            question=request.question,
        )
        answer = rag_result["answer"]
        sources = rag_result["sources"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")

    # Step 2: Get context text from the search (for evaluation)
    context_chunks = search_documents(request.question, k=5)
    context_text = "\n\n".join(
        f"[{c['source']} p.{c['page']}]: {c['content']}"
        for c in context_chunks
    )

    # Step 3: Hallucination check
    hallucination_result = check_hallucination(answer=answer, context=context_text)

    # Step 4: Quality scoring
    quality_result = score_answer(
        question=request.question,
        answer=answer,
        context=context_text,
    )

    return {
        "question": request.question,
        "answer": answer,
        "sources": sources,
        "hallucination_check": hallucination_result,
        "quality_score": quality_result,
        "summary": {
            "is_grounded": hallucination_result.get("grounded"),
            "quality": quality_result.get("quality_score"),
            "confidence": hallucination_result.get("confidence"),
            "verdict": hallucination_result.get("verdict"),
        },
    }


@router.get("/runs", summary="View recent LLM call metrics")
async def get_runs(limit: int = 50):
    """
    View the most recent LLM calls with their latency and token usage.

    Requires that you use `LatencyAndTokenTracker` callbacks in your LLM calls.
    Returns up to `limit` most recent runs (default 50).
    """
    runs = get_run_log(limit=limit)
    return {
        "runs": runs,
        "count": len(runs),
    }


@router.get("/stats", summary="Aggregate usage statistics")
async def get_stats():
    """
    Get aggregate statistics across all tracked LLM calls:
    - Total calls, average latency, total tokens, estimated cost
    - Breakdown by run type (chat, rag, agent, etc.)
    """
    return get_aggregate_stats()
