"""
Workflows API — REST endpoints for multi-agent workflows.

Endpoints:
  POST   /api/workflows/run     — Run a multi-agent workflow (sync, returns full result)
  POST   /api/workflows/stream  — Run a workflow with streaming progress events
  GET    /api/workflows/workers — List all available worker agents
"""

import asyncio
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from loguru import logger

from backend.workflows.worker_graph import run_workflow, astream_workflow, WORKERS

router = APIRouter(prefix="/api/workflows", tags=["Workflows"])


# ─── Request/Response Models ──────────────────────────────────────────────────

class WorkflowRequest(BaseModel):
    task: str = Field(
        ...,
        min_length=10,
        description=(
            "The complex task for the multi-agent system. "
            "Be specific about what you want researched, analyzed, or written."
        ),
        examples=[
            "Research the top 5 AI companies by market cap, analyze their growth trends, "
            "and write a structured investment report with recommendations.",
            "Create a detailed action plan for launching a SaaS product in 90 days, "
            "including market research, development milestones, and go-to-market strategy.",
        ],
    )


class WorkflowResponse(BaseModel):
    task: str
    final_output: str
    completed_work: dict
    iteration_count: int
    worker_sequence: list[str]


# ─── Endpoints ───────────────────────────────────────────────────────────────

@router.post("/run", response_model=WorkflowResponse)
async def run_workflow_endpoint(request: WorkflowRequest):
    """
    Run a multi-agent workflow and return the complete result.

    The supervisor will automatically delegate your task to the right combination
    of worker agents (researcher, analyst, writer, planner) and synthesize
    a final answer.

    **This can take 30-120 seconds** depending on task complexity.
    For real-time progress updates, use `POST /api/workflows/stream` instead.
    """
    logger.info(f"Workflow run request: '{request.task[:80]}'")

    try:
        result = await asyncio.to_thread(run_workflow, request.task)
    except Exception as e:
        logger.error(f"Workflow endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Workflow failed: {str(e)}")

    return WorkflowResponse(
        task=request.task,
        final_output=result["final_output"],
        completed_work=result["completed_work"],
        iteration_count=result["iteration_count"],
        worker_sequence=result.get("worker_sequence", []),
    )


@router.post("/stream")
async def stream_workflow_endpoint(request: WorkflowRequest):
    """
    Run a multi-agent workflow with **streaming progress updates**.

    Each worker's output is streamed as JSON lines as soon as it completes.
    The final event is `workflow_complete`.

    **Event format (newline-delimited JSON):**
    ```json
    {"event": "supervisor_routing", "next_worker": "researcher", "instructions": "..."}
    {"event": "worker_output", "worker": "researcher", "output": "..."}
    {"event": "worker_output", "worker": "writer", "output": "..."}
    {"event": "workflow_complete", "final_output": {...}}
    ```

    **JavaScript client:**
    ```javascript
    const response = await fetch('/api/workflows/stream', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({task: 'Research AI trends...'})
    });
    for await (const line of response.body) {
      const event = JSON.parse(new TextDecoder().decode(line));
      console.log(event);
    }
    ```
    """
    logger.info(f"Workflow stream request: '{request.task[:80]}'")

    async def generate():
        try:
            async for event in astream_workflow(request.task):
                yield json.dumps(event) + "\n"
        except Exception as e:
            yield json.dumps({"event": "error", "message": str(e)}) + "\n"

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",  # Newline-delimited JSON
        headers={"Cache-Control": "no-cache"},
    )


@router.get("/workers")
async def list_workers():
    """
    List all available worker agents in the workflow system.
    """
    return {
        "workers": [
            {
                "name": "researcher",
                "description": "Searches web, gathers facts and current information",
                "use_when": "Need up-to-date data, facts, statistics, or background research",
            },
            {
                "name": "analyst",
                "description": "Analyzes data, extracts insights, finds patterns",
                "use_when": "Have data that needs interpretation, comparison, or analysis",
            },
            {
                "name": "writer",
                "description": "Writes structured Markdown reports, memos, articles",
                "use_when": "Have enough information and need polished written output",
            },
            {
                "name": "planner",
                "description": "Creates action plans, timelines, checklists",
                "use_when": "Need step-by-step plans, roadmaps, or implementation guidance",
            },
        ],
        "max_iterations": 8,
        "entry_point": "supervisor",
    }
