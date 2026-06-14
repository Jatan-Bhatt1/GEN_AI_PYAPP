"""
Tracing — LangSmith integration + local latency and token tracking.

LangSmith is Langchain's observability platform. When enabled:
- Every LLM call is recorded with inputs, outputs, latency, token usage
- You can trace exactly what happened in multi-step chains
- View a waterfall diagram of chain execution at smith.langchain.com

Local tracking:
- Even without LangSmith, we track latency and token counts per request
- Stored in a simple in-memory log (swap to DB in production)

Setup:
  In .env set:
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_API_KEY=your_langsmith_key
    LANGCHAIN_PROJECT=enterprise-ai-assistant

  Then visit: https://smith.langchain.com → see all your traces
"""

import time
import os
from datetime import datetime, timezone
from typing import Any
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from loguru import logger

from backend.config import get_settings

settings = get_settings()


# ─── In-Memory Run Log ────────────────────────────────────────────────────────
# Stores the last 1000 runs for the /api/evaluations/runs endpoint
# In production: persist to PostgreSQL

_run_log: list[dict] = []
MAX_LOG_SIZE = 1000


def _append_run(run: dict) -> None:
    """Add a run to the log, maintaining max size."""
    global _run_log
    _run_log.append(run)
    if len(_run_log) > MAX_LOG_SIZE:
        _run_log = _run_log[-MAX_LOG_SIZE:]


def get_run_log(limit: int = 50) -> list[dict]:
    """Get recent runs (most recent first)."""
    return list(reversed(_run_log[-limit:]))


# ─── LangChain Callback Handler ───────────────────────────────────────────────

class LatencyAndTokenTracker(BaseCallbackHandler):
    """
    Custom LangChain callback handler that tracks:
    - Request latency (time from LLM call start to finish)
    - Token usage (prompt tokens + completion tokens)
    - Estimated cost (based on model pricing)

    This is attached to LLM calls as a callback:
        llm.invoke(prompt, callbacks=[LatencyAndTokenTracker()])

    OR passed to a chain:
        chain.invoke({...}, config={"callbacks": [LatencyAndTokenTracker()]})
    """

    def __init__(self, run_name: str = "llm_call", session_id: str = "unknown"):
        super().__init__()
        self.run_name = run_name
        self.session_id = session_id
        self._start_time: float | None = None
        self._prompt_tokens: int = 0
        self._completion_tokens: int = 0

    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs) -> None:
        """Called when LLM starts generating."""
        self._start_time = time.perf_counter()
        logger.debug(f"[Tracker] LLM started: {self.run_name}")

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """
        Called when LLM finishes. Extracts token usage and computes latency.

        LLMResult.llm_output contains token usage from the API response.
        """
        latency_ms = 0.0
        if self._start_time:
            latency_ms = (time.perf_counter() - self._start_time) * 1000

        # Extract token usage from LLM response
        token_usage = {}
        if response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})
            self._prompt_tokens = token_usage.get("prompt_tokens", 0)
            self._completion_tokens = token_usage.get("completion_tokens", 0)

        total_tokens = self._prompt_tokens + self._completion_tokens

        # Estimate cost (GPT-4o pricing: ~$5/1M input + $15/1M output)
        cost_usd = (
            (self._prompt_tokens / 1_000_000) * 5.0
            + (self._completion_tokens / 1_000_000) * 15.0
        )

        run = {
            "run_name": self.run_name,
            "session_id": self.session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latency_ms": round(latency_ms, 2),
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": round(cost_usd, 6),
            "model": settings.default_llm_provider,
        }

        _append_run(run)

        logger.info(
            f"[Tracker] {self.run_name} | "
            f"latency={latency_ms:.0f}ms | "
            f"tokens={total_tokens} | "
            f"cost=${cost_usd:.4f}"
        )

    def on_llm_error(self, error: Exception, **kwargs) -> None:
        """Called if the LLM call fails."""
        logger.error(f"[Tracker] LLM error in {self.run_name}: {error}")


# ─── Convenience Function ─────────────────────────────────────────────────────

def get_tracker(run_name: str = "llm_call", session_id: str = "unknown") -> LatencyAndTokenTracker:
    """
    Get a configured tracker callback.

    Usage:
        tracker = get_tracker("rag_query", session_id=session_id)
        result = llm.invoke(prompt, config={"callbacks": [tracker]})

    Args:
        run_name: Label for this run (e.g., "rag_query", "chat", "agent_run").
        session_id: The user's session ID for grouping runs.

    Returns:
        LatencyAndTokenTracker instance ready to use as a callback.
    """
    return LatencyAndTokenTracker(run_name=run_name, session_id=session_id)


def get_aggregate_stats() -> dict:
    """
    Compute aggregate statistics from the run log.

    Returns:
        Dict with:
        - total_runs: Total number of LLM calls logged
        - avg_latency_ms: Average response time
        - total_tokens: Total tokens consumed
        - total_cost_usd: Estimated total cost
        - runs_by_name: Breakdown by run_name (e.g., chat, rag, agent)
    """
    if not _run_log:
        return {
            "total_runs": 0,
            "avg_latency_ms": 0,
            "total_tokens": 0,
            "total_cost_usd": 0,
            "runs_by_name": {},
        }

    total_runs = len(_run_log)
    avg_latency = sum(r["latency_ms"] for r in _run_log) / total_runs
    total_tokens = sum(r["total_tokens"] for r in _run_log)
    total_cost = sum(r["estimated_cost_usd"] for r in _run_log)

    # Group by run_name
    runs_by_name: dict[str, dict] = {}
    for r in _run_log:
        name = r["run_name"]
        if name not in runs_by_name:
            runs_by_name[name] = {"count": 0, "total_tokens": 0, "avg_latency_ms": 0}
        runs_by_name[name]["count"] += 1
        runs_by_name[name]["total_tokens"] += r["total_tokens"]

    # Compute per-name averages
    for name, stats in runs_by_name.items():
        name_runs = [r for r in _run_log if r["run_name"] == name]
        stats["avg_latency_ms"] = round(
            sum(r["latency_ms"] for r in name_runs) / len(name_runs), 2
        )

    return {
        "total_runs": total_runs,
        "avg_latency_ms": round(avg_latency, 2),
        "total_tokens": total_tokens,
        "total_cost_usd": round(total_cost, 6),
        "runs_by_name": runs_by_name,
    }
