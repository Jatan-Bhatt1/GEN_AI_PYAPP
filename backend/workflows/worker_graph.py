"""
Worker Graph — LangGraph multi-agent workflow with supervisor routing.

Graph topology:
    START → supervisor → [researcher | analyst | writer | planner] → supervisor → ...
                                                                            └→ END

The supervisor is called after every worker. It decides:
- Which worker to call next
- OR route to END when the task is complete

State is passed through the entire graph:
- task: the original user request
- messages: conversation history
- completed_work: dict of {worker_name: result} for all completed workers
- next: routing decision from supervisor
- supervisor_instructions: specific instructions for the next worker
- final_output: the synthesized final answer
- iteration_count: safety counter to prevent infinite loops
"""

from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from loguru import logger

from backend.config import get_settings
from backend.workflows.supervisor import supervisor_node, WORKERS
from backend.tools.web_search import web_search_tool
from backend.tools.calculator import calculator_tool

settings = get_settings()

# ─── Maximum iterations before forcing FINISH ────────────────────────────────
MAX_ITERATIONS = 8


# ─── Workflow State Schema ────────────────────────────────────────────────────

class WorkflowState(TypedDict):
    """
    The complete state passed through the multi-agent workflow.

    Every node reads from this state and returns updates to it.
    LangGraph merges the returned dict into the state automatically.
    """
    task: str                                          # Original user task (immutable)
    messages: Annotated[list[BaseMessage], add_messages]  # Full message log
    completed_work: dict                               # {worker_name: result_string}
    next: str                                          # Supervisor's routing decision
    supervisor_instructions: str                       # Instructions for the next worker
    final_output: str                                  # Synthesized final answer
    iteration_count: int                               # Safety counter
    error: str                                         # Error message if something failed


# ─── LLM Factory ─────────────────────────────────────────────────────────────

def _get_worker_llm():
    """Return LLM for workers — same model, slightly more creative than supervisor."""
    if settings.default_llm_provider == "openai":
        return ChatOpenAI(
            model="gpt-4o",
            temperature=0.3,
            openai_api_key=settings.openai_api_key,
        )
    elif settings.default_llm_provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            groq_api_key=settings.groq_api_key,
        )
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
        google_api_key=settings.google_api_key,
    )


# ─── Worker Node Implementations ─────────────────────────────────────────────

def researcher_node(state: WorkflowState) -> dict:
    """
    Researcher worker — searches web and gathers information.

    Uses web_search tool to fetch current data.
    Returns gathered information in a structured format.
    """
    logger.info("🔍 Researcher worker activated")
    instructions = state.get("supervisor_instructions", "Research the task thoroughly.")
    task = state.get("task", "")
    llm = _get_worker_llm()

    system_msg = SystemMessage(content=(
        "You are a research specialist. Your job is to gather comprehensive, "
        "accurate information on the given topic.\n\n"
        "Guidelines:\n"
        "- Search for current, relevant information\n"
        "- Cite sources when possible\n"
        "- Organize findings clearly with headings\n"
        "- Focus on facts, statistics, and verifiable claims\n"
        "- Be thorough but concise (aim for 300-600 words)"
    ))

    # Try to use web search if available
    search_results = ""
    try:
        # Extract search query from instructions
        search_query = instructions if len(instructions) < 200 else task[:200]
        raw_results = web_search_tool.invoke({"query": search_query, "max_results": 5})
        search_results = f"\n\nWeb Search Results:\n{raw_results}"
    except Exception as e:
        logger.warning(f"Web search failed in researcher: {e}")
        search_results = "\n\n[Web search unavailable — using knowledge from training data]"

    human_msg = HumanMessage(content=(
        f"Original task: {task}\n\n"
        f"Your specific instructions: {instructions}\n"
        f"{search_results}\n\n"
        "Please provide a well-organized research report based on this information."
    ))

    try:
        response = llm.invoke([system_msg, human_msg])
        result = response.content
        logger.info(f"Researcher completed: {len(result)} chars")
    except Exception as e:
        result = f"Research failed: {e}"
        logger.error(f"Researcher error: {e}")

    completed_work = dict(state.get("completed_work", {}))
    completed_work["researcher"] = result

    return {
        "completed_work": completed_work,
        "messages": [AIMessage(content=f"[Researcher]\n{result}")],
    }


def analyst_node(state: WorkflowState) -> dict:
    """
    Analyst worker — analyzes information and extracts insights.

    Takes research output or raw data and produces:
    - Key insights and patterns
    - Comparative analysis
    - Data-driven conclusions
    """
    logger.info("📊 Analyst worker activated")
    instructions = state.get("supervisor_instructions", "Analyze the available information.")
    task = state.get("task", "")
    completed_work = state.get("completed_work", {})
    llm = _get_worker_llm()

    # Build context from previous worker outputs
    previous_work = ""
    if completed_work:
        parts = [f"### {k.capitalize()} Output:\n{v}" for k, v in completed_work.items()]
        previous_work = "\n\n".join(parts)

    system_msg = SystemMessage(content=(
        "You are a data and information analyst. Your job is to analyze information "
        "and extract meaningful insights.\n\n"
        "Guidelines:\n"
        "- Identify patterns, trends, and key insights\n"
        "- Make data-driven comparisons\n"
        "- Highlight the most important findings\n"
        "- Use bullet points and structured formatting\n"
        "- Include specific numbers and percentages when available\n"
        "- Identify gaps or limitations in the available data"
    ))

    human_msg = HumanMessage(content=(
        f"Original task: {task}\n\n"
        f"Your specific instructions: {instructions}\n\n"
        f"Available information to analyze:\n{previous_work or 'No previous work yet.'}"
    ))

    try:
        response = llm.invoke([system_msg, human_msg])
        result = response.content
        logger.info(f"Analyst completed: {len(result)} chars")
    except Exception as e:
        result = f"Analysis failed: {e}"
        logger.error(f"Analyst error: {e}")

    completed = dict(completed_work)
    completed["analyst"] = result

    return {
        "completed_work": completed,
        "messages": [AIMessage(content=f"[Analyst]\n{result}")],
    }


def writer_node(state: WorkflowState) -> dict:
    """
    Writer worker — produces polished, structured Markdown documents.

    Takes research + analysis and creates:
    - Executive summaries
    - Full reports with sections
    - Articles and blog posts
    - Technical documentation
    """
    logger.info("✍️ Writer worker activated")
    instructions = state.get("supervisor_instructions", "Write a comprehensive report.")
    task = state.get("task", "")
    completed_work = state.get("completed_work", {})
    llm = _get_worker_llm()

    previous_work = ""
    if completed_work:
        parts = [f"### {k.capitalize()} Output:\n{v}" for k, v in completed_work.items()]
        previous_work = "\n\n".join(parts)

    system_msg = SystemMessage(content=(
        "You are a professional writer specializing in business and technical content.\n\n"
        "Guidelines:\n"
        "- Write in clear, professional language\n"
        "- Use proper Markdown formatting (# headings, **bold**, bullet points, tables)\n"
        "- Structure: Executive Summary → Key Findings → Details → Conclusions\n"
        "- Include all relevant information from research and analysis\n"
        "- Make it ready to present to stakeholders\n"
        "- Do NOT add placeholder sections — only write what you have data for"
    ))

    human_msg = HumanMessage(content=(
        f"Original task: {task}\n\n"
        f"Your specific instructions: {instructions}\n\n"
        f"Source material (research + analysis):\n{previous_work or 'Write based on your knowledge.'}"
    ))

    try:
        response = llm.invoke([system_msg, human_msg])
        result = response.content
        logger.info(f"Writer completed: {len(result)} chars")
    except Exception as e:
        result = f"Writing failed: {e}"
        logger.error(f"Writer error: {e}")

    completed = dict(completed_work)
    completed["writer"] = result

    return {
        "completed_work": completed,
        "messages": [AIMessage(content=f"[Writer]\n{result}")],
        "final_output": result,  # Writer's output is often the final deliverable
    }


def planner_node(state: WorkflowState) -> dict:
    """
    Planner worker — creates action plans, roadmaps, and structured steps.

    Produces:
    - Step-by-step action plans
    - Project timelines
    - Task checklists
    - Implementation roadmaps
    """
    logger.info("📋 Planner worker activated")
    instructions = state.get("supervisor_instructions", "Create a detailed action plan.")
    task = state.get("task", "")
    completed_work = state.get("completed_work", {})
    llm = _get_worker_llm()

    previous_work = ""
    if completed_work:
        parts = [f"### {k.capitalize()} Output:\n{v}" for k, v in completed_work.items()]
        previous_work = "\n\n".join(parts)

    system_msg = SystemMessage(content=(
        "You are a strategic planner and project manager.\n\n"
        "Guidelines:\n"
        "- Break complex goals into concrete, actionable steps\n"
        "- Number steps sequentially with time estimates\n"
        "- Identify dependencies between steps\n"
        "- Include success criteria for each step\n"
        "- Use Markdown with numbered lists and checkboxes [ ]\n"
        "- Be realistic about timelines and resources needed"
    ))

    human_msg = HumanMessage(content=(
        f"Original task: {task}\n\n"
        f"Your specific instructions: {instructions}\n\n"
        f"Available context:\n{previous_work or 'No previous work yet.'}"
    ))

    try:
        response = llm.invoke([system_msg, human_msg])
        result = response.content
        logger.info(f"Planner completed: {len(result)} chars")
    except Exception as e:
        result = f"Planning failed: {e}"
        logger.error(f"Planner error: {e}")

    completed = dict(completed_work)
    completed["planner"] = result

    return {
        "completed_work": completed,
        "messages": [AIMessage(content=f"[Planner]\n{result}")],
    }


# ─── Routing Function ─────────────────────────────────────────────────────────

def route_after_supervisor(state: WorkflowState) -> Literal[
    "researcher", "analyst", "writer", "planner", "__end__"
]:
    """
    Conditional edge function — reads supervisor's decision and routes to
    the correct worker node OR ends the workflow.

    This function is called by LangGraph after every supervisor invocation.
    It must return the NAME of the next node, or END.

    Args:
        state: Current workflow state (contains "next" field set by supervisor)

    Returns:
        Name of the next node to execute
    """
    next_worker = state.get("next", "FINISH")
    iteration = state.get("iteration_count", 0)

    # Safety: force FINISH if we've run too many iterations
    if iteration >= MAX_ITERATIONS:
        logger.warning(f"Max iterations ({MAX_ITERATIONS}) reached. Forcing FINISH.")
        return "__end__"

    if next_worker == "FINISH":
        logger.info("Supervisor decided: FINISH")
        return "__end__"

    if next_worker in WORKERS:
        logger.info(f"Supervisor routing to: {next_worker}")
        return next_worker

    # Unknown routing decision — end safely
    logger.error(f"Unknown routing decision: '{next_worker}'. Ending workflow.")
    return "__end__"


# ─── Build the Graph ──────────────────────────────────────────────────────────

def build_workflow_graph():
    """
    Compile and return the multi-agent workflow graph.

    Graph topology:
        START → supervisor
        supervisor → researcher (conditional)
        supervisor → analyst (conditional)
        supervisor → writer (conditional)
        supervisor → planner (conditional)
        supervisor → END (conditional, when next="FINISH")
        researcher → supervisor
        analyst → supervisor
        writer → supervisor
        planner → supervisor

    Returns:
        Compiled CompiledStateGraph ready for .invoke() and .stream()
    """
    graph = StateGraph(WorkflowState)

    # Register all nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("writer", writer_node)
    graph.add_node("planner", planner_node)

    # Entry point: always start at supervisor
    graph.add_edge(START, "supervisor")

    # After supervisor: conditional routing to any worker or END
    graph.add_conditional_edges(
        "supervisor",            # Source node
        route_after_supervisor,  # Routing function
        {
            "researcher": "researcher",
            "analyst": "analyst",
            "writer": "writer",
            "planner": "planner",
            "__end__": END,
        },
    )

    # After any worker: always return to supervisor
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("analyst", "supervisor")
    graph.add_edge("writer", "supervisor")
    graph.add_edge("planner", "supervisor")

    return graph.compile()


# Global compiled workflow — shared across requests
_workflow_app = build_workflow_graph()


# ─── Public Invoke API ────────────────────────────────────────────────────────

def run_workflow(task: str) -> dict:
    """
    Run the multi-agent workflow for a given task.

    Args:
        task: The user's complex task description.

    Returns:
        dict with:
          - final_output: The synthesized final answer
          - completed_work: Dict of {worker: output} for each worker that ran
          - iteration_count: How many supervisor iterations ran
          - worker_sequence: Order in which workers were called
    """
    logger.info(f"Starting workflow for task: '{task[:80]}...'")

    initial_state: WorkflowState = {
        "task": task,
        "messages": [HumanMessage(content=task)],
        "completed_work": {},
        "next": "",
        "supervisor_instructions": "",
        "final_output": "",
        "iteration_count": 0,
        "error": "",
    }

    try:
        result = _workflow_app.invoke(initial_state)
    except Exception as e:
        logger.error(f"Workflow failed: {e}", exc_info=True)
        return {
            "final_output": f"Workflow failed: {str(e)}",
            "completed_work": {},
            "iteration_count": 0,
            "error": str(e),
        }

    # Build worker sequence from messages
    worker_sequence = []
    for msg in result.get("messages", []):
        if isinstance(msg, AIMessage) and msg.content.startswith("["):
            for worker in WORKERS:
                if msg.content.startswith(f"[{worker.capitalize()}]"):
                    worker_sequence.append(worker)

    # If no writer output, combine all completed work as final output
    final = result.get("final_output", "")
    if not final and result.get("completed_work"):
        parts = [f"## {k.capitalize()} Output\n\n{v}"
                 for k, v in result["completed_work"].items()]
        final = "\n\n---\n\n".join(parts)

    logger.info(
        f"Workflow complete | iterations={result.get('iteration_count', 0)} | "
        f"workers={worker_sequence}"
    )

    return {
        "final_output": final or "No output generated.",
        "completed_work": result.get("completed_work", {}),
        "iteration_count": result.get("iteration_count", 0),
        "worker_sequence": worker_sequence,
    }


async def astream_workflow(task: str):
    """
    Async generator that streams workflow events as they happen.

    Yields dicts with event information:
    - {"event": "worker_start", "worker": "researcher", ...}
    - {"event": "worker_output", "worker": "researcher", "output": "..."}
    - {"event": "workflow_complete", "final_output": "...", ...}

    Use this in streaming API endpoints.
    """
    initial_state: WorkflowState = {
        "task": task,
        "messages": [HumanMessage(content=task)],
        "completed_work": {},
        "next": "",
        "supervisor_instructions": "",
        "final_output": "",
        "iteration_count": 0,
        "error": "",
    }

    final_result = None

    try:
        async for event in _workflow_app.astream(initial_state):
            # event is a dict like {"supervisor": {...state update...}}
            for node_name, state_update in event.items():
                if node_name == "__end__":
                    continue

                if node_name == "supervisor":
                    decision = state_update.get("next", "")
                    instructions = state_update.get("supervisor_instructions", "")
                    if decision and decision != "FINISH":
                        yield {
                            "event": "supervisor_routing",
                            "next_worker": decision,
                            "instructions": instructions[:100],
                        }

                elif node_name in WORKERS:
                    completed = state_update.get("completed_work", {})
                    worker_output = completed.get(node_name, "")
                    if worker_output:
                        yield {
                            "event": "worker_output",
                            "worker": node_name,
                            "output": worker_output,
                            "output_length": len(worker_output),
                        }
                        final_result = state_update

    except Exception as e:
        yield {"event": "error", "message": str(e)}
        return

    yield {"event": "workflow_complete", "final_output": final_result}
