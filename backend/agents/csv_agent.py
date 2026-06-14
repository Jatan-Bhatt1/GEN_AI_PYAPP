"""
CSV Agent — Pandas-powered agent for analyzing CSV and Excel files.
Allows natural language queries against tabular data.
"""

import os
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from loguru import logger

from backend.config import get_settings

settings = get_settings()


def _get_llm():
    if settings.default_llm_provider == "openai":
        return ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            openai_api_key=settings.openai_api_key,
        )
    elif settings.default_llm_provider == "groq":
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            groq_api_key=settings.groq_api_key,
        )
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,
        google_api_key=settings.google_api_key,
    )


def create_csv_agent(file_path: str):
    """
    Create a CSV/Excel analysis agent for a specific file.

    Supports: .csv, .xlsx, .xls, .tsv

    Args:
        file_path: Absolute or relative path to the CSV/Excel file

    Returns:
        AgentExecutor that can answer questions about the data

    Example:
        agent = create_csv_agent("uploads/sales_data.csv")
        result = agent.invoke({"input": "What is the average revenue by region?"})
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    # Load data based on file type
    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(file_path, engine="openpyxl")
    elif ext == ".tsv":
        df = pd.read_csv(file_path, sep="\t")
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: .csv, .xlsx, .xls, .tsv")

    logger.info(f"CSV agent loaded: {file_path} — shape: {df.shape}, columns: {list(df.columns)}")

    llm = _get_llm()

    agent = create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        agent_type="tool-calling",   # works for both OpenAI and Google in LangChain 1.x
        verbose=True,
        allow_dangerous_code=True,   # required for Pandas agent (runs Python)
        handle_parsing_errors=True,
        max_iterations=8,
        return_intermediate_steps=True,
        prefix=(
            "You are a data analyst expert. Analyze the dataframe and answer questions accurately. "
            "Always show your work. For numerical results, include units if relevant. "
            "When asked to plot or visualize, describe what the chart would show instead."
        ),
    )

    return agent


def get_dataframe_summary(file_path: str) -> dict:
    """
    Generate a quick statistical summary of a CSV/Excel file.

    Returns a dict with shape, column types, null counts, and describe() stats.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(file_path, engine="openpyxl")
    else:
        df = pd.read_csv(file_path, sep="\t")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    summary = {
        "file": os.path.basename(file_path),
        "shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "columns": list(df.columns),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "null_counts": df.isnull().sum().to_dict(),
        "numeric_stats": df[numeric_cols].describe().to_dict() if numeric_cols else {},
        "sample_rows": df.head(3).to_dict(orient="records"),
    }

    return summary
