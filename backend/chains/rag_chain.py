"""
Conversational RAG Chain — The heart of Phase 8.

This chain does two things:
1. REFORMULATE: Turn vague follow-up questions into standalone searchable queries
2. ANSWER: Search documents, combine with history, generate cited answers

Key LangChain 1.x functions used:
  - create_history_aware_retriever: wraps retriever with question reformulation
  - create_stuff_documents_chain: stuffs retrieved docs into LLM prompt
  - create_retrieval_chain: combines retriever + QA chain

Memory (per session):
  - Uses LangGraph MemorySaver — same pattern as Phase 5 chat_chain.py
  - Each session (thread_id) has its own conversation history
  - History is automatically loaded before each call and saved after

Source Citations:
  - Every answer includes the source documents used
  - Format: "Source: policy.pdf (Page 3)"
"""

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated
from loguru import logger

from backend.config import get_settings
from backend.rag.retriever import get_retriever

settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# LLM FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def _get_llm():
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


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT 1 — QUESTION REFORMULATION
# (History-Aware Retriever needs this prompt)
# ─────────────────────────────────────────────────────────────────────────────

_REFORMULATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Given the conversation history below and the user's latest question, "
     "reformulate the question so it is STANDALONE and can be understood "
     "WITHOUT needing the conversation history.\n\n"
     "Rules:\n"
     "- If the question is already standalone, return it AS-IS\n"
     "- If it uses pronouns like 'it', 'that', 'they', replace with specific nouns\n"
     "- Do NOT answer the question — only rewrite it\n"
     "- Output ONLY the reformulated question, nothing else\n\n"
     "Examples:\n"
     "  History: 'What is the refund policy?' / 'Refunds take 5-7 days.'\n"
     "  Follow-up: 'What about international orders?'\n"
     "  Reformulated: 'What is the refund policy for international orders?'\n\n"
     "  History: 'Tell me about Python lists.'\n"
     "  Follow-up: 'How do I sort them?'\n"
     "  Reformulated: 'How do I sort Python lists?'"
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT 2 — ANSWER GENERATION WITH SOURCE CITATIONS
# ─────────────────────────────────────────────────────────────────────────────

_QA_SYSTEM_PROMPT = (
    "You are an expert document analyst for an enterprise AI assistant. "
    "Answer questions based ONLY on the provided document context.\n\n"
    "Guidelines:\n"
    "- Answer precisely and thoroughly using ONLY the provided context\n"
    "- If the answer is not in the context, say: 'I could not find information about this "
    "in the uploaded documents. Please check the source material directly.'\n"
    "- NEVER fabricate information not present in the context\n"
    "- Always cite your sources using this format: **[Source: {{filename}}, Page {{page}}]**\n"
    "- Use bullet points and structured formatting for clarity\n"
    "- If multiple documents are relevant, synthesize information from all of them\n\n"
    "Document context:\n"
    "{context}"
    # ↑ {context} is automatically filled by create_stuff_documents_chain
    #   with the retrieved document chunks, formatted as text
)

_QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _QA_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])


# ─────────────────────────────────────────────────────────────────────────────
# LANGGRAPH STATE FOR CONVERSATIONAL RAG
# ─────────────────────────────────────────────────────────────────────────────

class RAGState(TypedDict):
    """
    State for the conversational RAG graph.

    messages: Full conversation history (auto-appended by add_messages reducer)
    sources: List of source citations from the last retrieval
    """
    messages: Annotated[list[BaseMessage], add_messages]
    sources: list[dict]  # [{source, page, content_preview}] from last answer


# ─────────────────────────────────────────────────────────────────────────────
# THE RAG NODE — Core logic
# ─────────────────────────────────────────────────────────────────────────────

def rag_node(state: RAGState) -> dict:
    """
    The single LangGraph node for conversational RAG.

    On every invocation:
    1. Gets history from state (LangGraph loads it automatically from checkpointer)
    2. Gets the latest human message
    3. Runs history-aware retriever (reformulate question → search ChromaDB)
    4. Generates answer with source citations
    5. Returns AI message + source info to save to state

    Args:
        state: RAGState with messages (including the new human message)

    Returns:
        Dict with new AI message and source citations
    """
    llm = _get_llm()
    retriever = get_retriever(k=5)

    # Extract conversation history (all messages EXCEPT the last human one)
    all_messages = state["messages"]
    # The last message is the new user question
    current_question = all_messages[-1].content
    # Everything before it is the history
    history = all_messages[:-1]

    # Build the history-aware retriever
    # This wraps our retriever with the reformulation step
    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=_REFORMULATION_PROMPT,
    )
    # When invoked: takes {input, chat_history} → reformulates → retrieves

    # Build the QA chain (stuffs retrieved docs into prompt, generates answer)
    qa_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=_QA_PROMPT,
    )
    # When invoked: takes {input, context, chat_history} → generates answer

    # Combine: retriever → QA chain
    rag_chain = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=qa_chain,
    )

    try:
        result = rag_chain.invoke({
            "input": current_question,
            "chat_history": history,
        })
        # result = {
        #   "input": "...",
        #   "chat_history": [...],
        #   "context": [Document, Document, ...],  ← retrieved chunks
        #   "answer": "The refund policy states..."  ← AI answer
        # }

        answer = result["answer"]
        context_docs = result.get("context", [])

        # Build source citations from retrieved documents
        sources = []
        seen_sources = set()
        for doc in context_docs:
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "N/A")
            key = f"{source}:{page}"
            if key not in seen_sources:
                seen_sources.add(key)
                sources.append({
                    "source": source,
                    "page": page,
                    "content_preview": doc.page_content[:200],
                })

        logger.info(
            f"RAG answer generated | "
            f"sources={[s['source'] for s in sources]} | "
            f"answer_length={len(answer)}"
        )

    except Exception as e:
        logger.error("RAG chain failed: {}", str(e), exc_info=True)
        answer = (
            "I encountered an error while searching the documents. "
            "Please ensure documents are uploaded and try again."
        )
        sources = []

    return {
        "messages": [AIMessage(content=answer)],
        "sources": sources,
    }


# ─────────────────────────────────────────────────────────────────────────────
# BUILD THE RAG GRAPH
# ─────────────────────────────────────────────────────────────────────────────

def build_rag_graph():
    """
    Compile the conversational RAG state graph.

    Graph: START → rag_node → END
    Checkpointer: MemorySaver (keyed by thread_id = session_id)

    This gives us per-session conversation history automatically.
    Same pattern as chat_chain.py (Phase 5).

    Returns:
        Compiled LangGraph app.
    """
    graph = StateGraph(RAGState)
    graph.add_node("rag", rag_node)
    graph.add_edge(START, "rag")
    graph.add_edge("rag", END)

    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)
    logger.info("Conversational RAG graph compiled")
    return compiled


# Global compiled RAG graph
_rag_app = build_rag_graph()


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def invoke_rag(session_id: str, question: str) -> dict:
    """
    Invoke the conversational RAG chain for a session.

    The graph checkpointer automatically:
    - Loads existing conversation history for this session
    - Saves the new human + AI messages after each invocation

    Args:
        session_id: Unique session identifier.
        question: The user's question about uploaded documents.

    Returns:
        Dict with:
          - answer: The AI's answer string
          - sources: List of source citations [{source, page, content_preview}]
          - session_id: Echo of the session ID
    """
    result = _rag_app.invoke(
        {
            "messages": [HumanMessage(content=question)],
            "sources": [],
        },
        config={"configurable": {"thread_id": session_id}},
    )

    return {
        "answer": result["messages"][-1].content,
        "sources": result.get("sources", []),
        "session_id": session_id,
    }


async def astream_rag(session_id: str, question: str):
    """
    Async generator: streams the RAG answer token-by-token.

    Yields string chunks as they arrive from the LLM.
    Source citations are emitted as a special final chunk:
      '[[SOURCES]]{"sources": [...]}'

    Args:
        session_id: Unique session identifier.
        question: User's question.

    Yields:
        String chunks of the answer, then a sources JSON chunk.
    """
    sources_emitted = False

    async for event in _rag_app.astream_events(
        {
            "messages": [HumanMessage(content=question)],
            "sources": [],
        },
        config={"configurable": {"thread_id": session_id}},
        version="v2",
    ):
        event_name = event.get("event")
        node_name = event.get("name", "")

        # Stream LLM tokens
        if event_name == "on_chat_model_stream" and node_name in (
            "ChatOpenAI", "ChatGoogleGenerativeAI", "ChatGroq"
        ):
            chunk = event["data"].get("chunk")
            if chunk and hasattr(chunk, "content") and chunk.content:
                yield chunk.content

        # After rag_node completes, emit sources
        elif event_name == "on_chain_end" and node_name == "rag" and not sources_emitted:
            output = event.get("data", {}).get("output", {})
            sources = output.get("sources", [])
            if sources:
                import json
                yield f"\n\n[[SOURCES]]{json.dumps(sources)}"
            sources_emitted = True


def get_rag_history(session_id: str) -> list[dict]:
    """
    Get the full RAG conversation history for a session.

    Args:
        session_id: Session to inspect.

    Returns:
        List of {role, content} dicts.
    """
    try:
        state = _rag_app.get_state(
            config={"configurable": {"thread_id": session_id}}
        )
        messages = state.values.get("messages", [])
        return [
            {
                "role": "human" if msg.type == "human" else "ai",
                "content": msg.content,
            }
            for msg in messages
        ]
    except Exception as e:
        logger.warning(f"Could not get RAG history for session '{session_id}': {e}")
        return []
