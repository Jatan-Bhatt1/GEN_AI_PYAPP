"""
Enterprise AI Assistant — FastAPI Application Entry Point.
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from loguru import logger
from backend.api.chat import router as chat_router
from backend.api.agents import router as agents_router
from backend.api.documents import router as documents_router
from backend.api.workflows import router as workflows_router

from backend.config import get_settings

# Load environment variables BEFORE importing anything that reads them
load_dotenv()

settings = get_settings()

# Set LangSmith tracing env vars at process level so they are visible to langchain internals
if settings.langchain_tracing_v2:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    # --- Startup ---
    logger.info("🚀 Starting Enterprise AI Assistant...")
    logger.info(f"  Environment  : {settings.app_env}")
    logger.info(f"  LLM Provider : {settings.default_llm_provider}")
    logger.info(f"  LangSmith    : {'enabled' if settings.langchain_tracing_v2 else 'disabled'}")

    # Ensure critical directories exist
    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs(settings.chroma_persist_dir, exist_ok=True)

    # Warm up: verify Redis connectivity (non-fatal if Redis is down)
    try:
        import redis as redis_client
        r = redis_client.from_url(settings.redis_url, socket_connect_timeout=2)
        r.ping()
        logger.info("  Redis        : ✅ connected")
    except Exception as e:
        logger.warning(f"  Redis        : ⚠️  not reachable ({e}) — memory features degraded")

    logger.info("✅ Application started successfully")
    yield

    # --- Shutdown ---
    logger.info("👋 Shutting down Enterprise AI Assistant...")


# ─────────────────────────────────────────────────────────────
# CREATE APP
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Enterprise AI Assistant",
    description=(
        "AI Workspace for Businesses — ChatGPT + Perplexity + Notion AI combined.\n\n"
        "**Phases implemented:**\n"
        "- Phase 1: Basic Chat (streaming)\n"
        "- Phase 2: RAG Pipeline (documents)\n"
        "- Phase 3: AI Agents (research, coding, SQL, CSV, report)\n"
        "- Phase 4: Tool Calling (web search, calculator, weather, email, SQL)\n"
        "- Phase 5: Memory System (buffer, summary, semantic)\n"
    ),
    version="0.5.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─────────────────────────────────────────────────────────────
# MIDDLEWARE
# ─────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# ROUTERS — register all API modules here
# ─────────────────────────────────────────────────────────────



app.include_router(chat_router)     # /api/chat/*   — Phase 1 + Phase 5 (memory)
app.include_router(agents_router)   # /api/agents/* — Phase 3 + Phase 4 (tools)
app.include_router(documents_router)
app.include_router(workflows_router)
# /api/documents/* — Phase 2 (RAG)
# ─────────────────────────────────────────────────────────────
# ROOT ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    """Root health check."""
    return {
        "status": "running",
        "app": "Enterprise AI Assistant",
        "version": "0.5.0",
        "environment": settings.app_env,
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check — shows status of all integrated services."""
    # Redis check
    redis_ok = False
    try:
        import redis as redis_client
        r = redis_client.from_url(settings.redis_url, socket_connect_timeout=1)
        r.ping()
        redis_ok = True
    except Exception:
        pass

    return {
        "status": "healthy",
        "version": "0.5.0",
        "llm_provider": settings.default_llm_provider,
        "services": {
            "openai_key": bool(settings.openai_api_key),
            "google_key": bool(settings.google_api_key),
            "langsmith": settings.langchain_tracing_v2,
            "tavily": bool(settings.tavily_api_key),
            "redis": redis_ok,
            "openweather": bool(settings.openweather_api_key),
            "smtp": bool(settings.smtp_user),
        },
        "memory": {
            "buffer_window": settings.memory_window_size,
            "summary_threshold": settings.memory_summary_threshold,
            "vector_collection": settings.memory_collection_name,
            "ttl_seconds": settings.memory_ttl_seconds,
        },
    }
