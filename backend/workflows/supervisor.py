"""
Supervisor Agent — Routes tasks to specialized worker agents.

The supervisor is an LLM that:
1. Reads the user's task and all previous results
2. Decides which worker to call next (or FINISH)
3. Returns: {"next": "researcher" | "analyst" | "writer" | "planner" | "FINISH"}

This is the classic LangGraph Supervisor pattern:
  https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from loguru import logger

from backend.config import get_settings

settings = get_settings()

# ─── Available workers ────────────────────────────────────────────────────────
WORKERS = ["researcher", "analyst", "writer", "planner"]
# researcher — searches web, gathers facts and sources
# analyst    — analyzes data, extracts insights, creates summaries
# writer     — writes structured reports, memos, content in Markdown
# planner    — creates action plans, timelines, task breakdowns

# ─── Supervisor output schema ─────────────────────────────────────────────────
class SupervisorDecision(BaseModel):
    """Structured output from the supervisor LLM."""
    next: Literal["researcher", "analyst", "writer", "planner", "FINISH"] = Field(
        description="Which worker to call next, or FINISH if the task is complete."
    )
    instructions: str = Field(
        description="Specific instructions for the chosen worker. Be precise and detailed."
    )
    reasoning: str = Field(
        description="Brief explanation of why you chose this worker."
    )


# ─── Supervisor prompt ────────────────────────────────────────────────────────
SUPERVISOR_SYSTEM = """You are a task supervisor managing a team of specialized AI workers.
Your job is to break down complex tasks and route them to the right specialist.

## Your Team:
- **researcher**: Searches the web, gathers current information, verifies facts. 
  Use when: needing up-to-date data, facts, statistics, or background information.
  
- **analyst**: Analyzes information, finds patterns, extracts insights, creates summaries.
  Use when: you have data that needs interpretation, comparison, or insight extraction.
  
- **writer**: Writes structured documents — reports, memos, summaries, articles in Markdown.
  Use when: you have enough information and need it formatted into readable output.
  
- **planner**: Creates action plans, timelines, project breakdowns, checklists.
  Use when: the task requires planning, steps, or structured execution guidance.

## Decision Rules:
1. Analyze the original task and all completed work so far
2. Identify what's MISSING to complete the task
3. Choose the worker that fills that gap
4. Give that worker CLEAR, SPECIFIC instructions (not vague)
5. Choose FINISH only when the task is fully complete with a final answer

## Output Format (JSON):
{{
  "next": "researcher" | "analyst" | "writer" | "planner" | "FINISH",
  "instructions": "Specific instructions for the chosen worker...",
  "reasoning": "Why I chose this worker..."
}}

Current task: {task}
Work completed so far:
{completed_work}
"""

def get_supervisor_llm():
    """Return LLM for supervisor — use a capable model for good routing decisions."""
    if settings.default_llm_provider == "openai":
        return ChatOpenAI(
            model="gpt-4o",
            temperature=0,           # deterministic routing decisions
            openai_api_key=settings.openai_api_key,
        )
    elif settings.default_llm_provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            groq_api_key=settings.groq_api_key,
        )
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        google_api_key=settings.google_api_key,
    )


def supervisor_node(state: dict) -> dict:
    """
    The supervisor node function — called by LangGraph at each routing step.

    Reads the current workflow state, asks the LLM what to do next,
    and returns the routing decision.

    Args:
        state: Current WorkflowState (task, messages, completed_work, etc.)

    Returns:
        Dict with "next" (worker name or FINISH) and "supervisor_instructions"
    """
    llm = get_supervisor_llm()

    task = state.get("task", "")
    completed_work = state.get("completed_work", {})

    # Format completed work for the supervisor to read
    completed_str = "None yet."
    if completed_work:
        parts = []
        for worker, result in completed_work.items():
            parts.append(f"### {worker.capitalize()} Output:\n{result}")
        completed_str = "\n\n".join(parts)

    # Ask the supervisor LLM for a routing decision
    prompt = SUPERVISOR_SYSTEM.format(
        task=task,
        completed_work=completed_str,
    )

    try:
        response = llm.invoke(prompt)
        raw = response.content.strip()

        # Try to parse JSON from the response
        import json, re
        # Extract JSON block if wrapped in markdown code fences
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            decision_data = json.loads(json_match.group())
        else:
            decision_data = json.loads(raw)

        decision = SupervisorDecision(**decision_data)
        logger.info(
            f"Supervisor decision: next={decision.next} | "
            f"reasoning={decision.reasoning[:80]}"
        )

    except Exception as e:
        logger.error(f"Supervisor LLM parsing failed: {e}. Defaulting to FINISH.")
        decision = SupervisorDecision(
            next="FINISH",
            instructions="Task complete.",
            reasoning=f"Error in supervisor: {e}",
        )

    return {
        "next": decision.next,
        "supervisor_instructions": decision.instructions,
        "iteration_count": state.get("iteration_count", 0) + 1,
    }
