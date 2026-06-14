"""
Agents API — Endpoints to invoke all specialized AI agents.
Supports: research, coding, code_review, report, summary, sql, csv
"""

import os
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from loguru import logger
from typing import Optional

from backend.config import get_settings

settings = get_settings()

router = APIRouter(prefix="/api/agents", tags=["Agents"])


# ─────────────────────────────────────────────
# REQUEST / RESPONSE SCHEMAS
# ─────────────────────────────────────────────

class AgentRequest(BaseModel):
    agent: str = Field(
        ...,
        description=(
            "Which agent to invoke. Options: "
            "'research', 'coding', 'code_review', 'report', 'summary', 'sql', 'csv'"
        ),
    )
    input: str = Field(..., description="The task or question to send to the agent")
    # Extra fields used by specific agents
    language: str = Field(default="python", description="Programming language (for code_review)")
    context: str = Field(default="", description="Additional context (for report/summary agents)")
    topic: str = Field(default="", description="Report topic (for report agent)")
    audience: str = Field(default="General", description="Target audience (for report agent)")
    format_instructions: str = Field(default="None", description="Special formatting instructions")
    max_words: int = Field(default=150, description="Max words for summaries")
    file_id: str = Field(default="", description="Uploaded CSV file ID (for csv agent)")
    session_id: str = Field(default="default", description="Session ID for stateful agents")


class AgentResponse(BaseModel):
    agent: str
    output: str
    intermediate_steps: Optional[list] = None


# ─────────────────────────────────────────────
# AGENT INVOCATION ENDPOINT
# ─────────────────────────────────────────────

@router.post("/invoke", response_model=AgentResponse)
async def invoke_agent(request: AgentRequest):
    """
    Invoke a specialized AI agent by name.

    Available agents:
    - **research**: Web search + calculator + weather + email (ReAct loop)
    - **coding**: Code generation with best practices
    - **code_review**: Analyze and improve existing code
    - **report**: Generate structured Markdown reports
    - **summary**: Summarize long text concisely
    - **sql**: Natural language → PostgreSQL queries
    - **csv**: Analyze CSV/Excel files with natural language

    Example:
    ```json
    {"agent": "research", "input": "What are the top AI trends in 2025?"}
    ```
    """
    agent_name = request.agent.lower().strip()
    logger.info(f"Invoking agent: {agent_name} | input: {request.input[:100]}")

    try:
        # ── RESEARCH AGENT ──
        if agent_name == "research":
            from backend.agents.research_agent import invoke_research_agent
            result = invoke_research_agent(request.input)
            return AgentResponse(agent=agent_name, output=result["output"])

        # ── CODING AGENT ──
        elif agent_name == "coding":
            from backend.agents.coding_agent import coding_chain
            output = coding_chain.invoke({
                "input": request.input,
                "chat_history": [],
            })
            return AgentResponse(agent=agent_name, output=output)

        # ── CODE REVIEW AGENT ──
        elif agent_name == "code_review":
            from backend.agents.coding_agent import code_review_chain
            output = code_review_chain.invoke({
                "code": request.input,
                "language": request.language,
                "context": request.context or "No additional context provided.",
            })
            return AgentResponse(agent=agent_name, output=output)

        # ── REPORT AGENT ──
        elif agent_name == "report":
            from backend.agents.report_agent import report_chain
            output = report_chain.invoke({
                "topic": request.topic or request.input,
                "context": request.context or request.input,
                "audience": request.audience,
                "format_instructions": request.format_instructions,
            })
            return AgentResponse(agent=agent_name, output=output)

        # ── SUMMARY AGENT ──
        elif agent_name == "summary":
            from backend.agents.report_agent import summary_chain
            output = summary_chain.invoke({
                "content": request.context or request.input,
                "max_words": request.max_words,
            })
            return AgentResponse(agent=agent_name, output=output)

        # ── SQL AGENT ──
        elif agent_name == "sql":
            from backend.agents.sql_agent import invoke_sql_agent
            result = invoke_sql_agent(request.input)
            return AgentResponse(agent=agent_name, output=result["output"])

        # ── CSV AGENT ──
        elif agent_name == "csv":
            if not request.file_id:
                raise HTTPException(
                    status_code=400,
                    detail="CSV agent requires a 'file_id'. "
                           "First upload a file via POST /api/analytics/upload, then use its file_id here.",
                )
            file_path = os.path.join(settings.upload_dir, request.file_id)
            if not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail=f"File not found: {request.file_id}")

            from backend.agents.csv_agent import create_csv_agent
            agent = create_csv_agent(file_path)
            result = agent.invoke({"input": request.input})
            steps = _format_steps(result.get("intermediate_steps", []))
            return AgentResponse(agent=agent_name, output=result["output"], intermediate_steps=steps)

        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unknown agent: '{agent_name}'. "
                    "Valid options: research, coding, code_review, report, summary, sql, csv"
                ),
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agent '{agent_name}' error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")


# ─────────────────────────────────────────────
# LIST AVAILABLE AGENTS
# ─────────────────────────────────────────────

@router.get("/list")
async def list_agents():
    """List all available agents with their descriptions and required fields."""
    return {
        "agents": [
            {
                "name": "research",
                "description": "Web search + calculator + weather + email tools. Best for current events, fact-finding, and multi-step research.",
                "required_fields": ["input"],
                "optional_fields": [],
            },
            {
                "name": "coding",
                "description": "Code generation with best practices, tests, and documentation.",
                "required_fields": ["input"],
                "optional_fields": [],
            },
            {
                "name": "code_review",
                "description": "Analyze code for bugs, security issues, performance, and style.",
                "required_fields": ["input (the code to review)"],
                "optional_fields": ["language", "context"],
            },
            {
                "name": "report",
                "description": "Generate structured executive Markdown reports from data/context.",
                "required_fields": ["topic", "context"],
                "optional_fields": ["audience", "format_instructions"],
            },
            {
                "name": "summary",
                "description": "Summarize long text, documents, or research into concise paragraphs.",
                "required_fields": ["context (the content to summarize)"],
                "optional_fields": ["max_words"],
            },
            {
                "name": "sql",
                "description": "Natural language to PostgreSQL. Lists tables, writes and runs SELECT queries.",
                "required_fields": ["input"],
                "optional_fields": [],
            },
            {
                "name": "csv",
                "description": "Analyze CSV/Excel files with natural language questions.",
                "required_fields": ["input", "file_id"],
                "optional_fields": [],
            },
        ]
    }


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _format_steps(intermediate_steps: list) -> list[dict]:
    """
    Convert raw intermediate steps into a JSON-serializable format.
    Each step is (AgentAction, observation_string).
    """
    formatted = []
    for action, observation in intermediate_steps:
        formatted.append({
            "tool": getattr(action, "tool", str(action)),
            "tool_input": getattr(action, "tool_input", ""),
            "observation": str(observation)[:500],  # truncate long outputs
        })
    return formatted
