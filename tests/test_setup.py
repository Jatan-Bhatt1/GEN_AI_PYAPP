"""
Phase 0 — Verify project setup is correct.
Run with: python -m pytest tests/test_setup.py -v
"""

import importlib


def test_backend_config_loads():
    """Config module loads without errors."""
    from backend.config import get_settings
    settings = get_settings()
    assert settings.app_env in ("development", "production", "testing")
    assert settings.default_llm_provider in ("openai", "google")


def test_fastapi_app_creates():
    """FastAPI app initializes correctly."""
    from backend.main import app
    assert app.title == "Enterprise AI Assistant"


def test_all_packages_importable():
    """All backend subpackages are importable."""
    packages = [
        "backend",
        "backend.api",
        "backend.agents",
        "backend.chains",
        "backend.rag",
        "backend.memory",
        "backend.tools",
        "backend.prompts",
        "backend.workflows",
        "backend.evaluations",
    ]
    for pkg in packages:
        mod = importlib.import_module(pkg)
        assert mod is not None, f"Failed to import {pkg}"


def test_core_dependencies_installed():
    """All critical dependencies are importable."""
    deps = [
        "langchain",
        "langchain_openai",
        "langchain_core",
        "langchain_community",
        "fastapi",
        "uvicorn",
        "chromadb",
        "pydantic",
        "dotenv",
        "loguru",
        "pandas",
    ]
    for dep in deps:
        mod = importlib.import_module(dep)
        assert mod is not None, f"Failed to import {dep}"
