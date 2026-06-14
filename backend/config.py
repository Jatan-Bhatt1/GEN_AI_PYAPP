"""
Centralized configuration for the Enterprise AI Assistant.
Loads environment variables and provides typed settings.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- LLM ---
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    default_llm_provider: str = Field(default="openai", alias="DEFAULT_LLM_PROVIDER")

    # --- Database ---
    database_url: str = Field(
        default="postgresql://postgres:password@localhost:5432/enterprise_ai",
        alias="DATABASE_URL",
    )
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

    # --- Vector Store ---
    chroma_persist_dir: str = Field(default="./vectorstore", alias="CHROMA_PERSIST_DIR")
    chroma_collection_name: str = Field(
        default="enterprise_docs", alias="CHROMA_COLLECTION_NAME"
    )

    # --- Memory ---
    memory_window_size: int = Field(
        default=10,
        alias="MEMORY_WINDOW_SIZE",
        description="Number of recent message pairs to keep in buffer (per side)",
    )
    memory_summary_threshold: int = Field(
        default=20,
        alias="MEMORY_SUMMARY_THRESHOLD",
        description="Total message count that triggers auto-summarization",
    )
    memory_collection_name: str = Field(
        default="conversation_memory",
        alias="MEMORY_COLLECTION_NAME",
        description="ChromaDB collection name for semantic memory (separate from RAG docs)",
    )
    memory_ttl_seconds: int = Field(
        default=86400,
        alias="MEMORY_TTL_SECONDS",
        description="How long Redis buffer memory lives (seconds). Default: 24h",
    )

    # --- LangSmith ---
    langchain_tracing_v2: bool = Field(default=False, alias="LANGCHAIN_TRACING_V2")
    langchain_api_key: str = Field(default="", alias="LANGCHAIN_API_KEY")
    langchain_project: str = Field(
        default="enterprise-ai-assistant", alias="LANGCHAIN_PROJECT"
    )

    # --- Web Search ---
    tavily_api_key: str = Field(default="", alias="TAVILY_API_KEY")

    # --- Weather ---
    openweather_api_key: str = Field(default="", alias="OPENWEATHER_API_KEY")

    # --- Email / SMTP ---
    smtp_host: str = Field(default="smtp.gmail.com", alias="SMTP_HOST")
    smtp_port: int = Field(default=587, alias="SMTP_PORT")
    smtp_user: str = Field(default="", alias="SMTP_USER")
    smtp_password: str = Field(default="", alias="SMTP_PASSWORD")
    smtp_from: str = Field(default="", alias="SMTP_FROM")

    # --- Auth ---
    jwt_secret_key: str = Field(default="change-me", alias="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    jwt_expiration_minutes: int = Field(default=60, alias="JWT_EXPIRATION_MINUTES")

    # --- App ---
    app_env: str = Field(default="development", alias="APP_ENV")
    app_debug: bool = Field(default=True, alias="APP_DEBUG")
    upload_dir: str = Field(default="./uploads", alias="UPLOAD_DIR")
    cors_origins: str = Field(default="http://localhost:3000", alias="CORS_ORIGINS")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra env vars


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
