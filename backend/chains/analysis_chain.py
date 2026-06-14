"""
Analysis Chain — LLM-powered data analysis chain for CSV/Excel files.

Two main capabilities:
1. Auto-summarize: Given a Pandas DataFrame, generate a human-readable summary
2. Query: Answer natural language questions about the data using the CSV agent
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd
from loguru import logger

from backend.config import get_settings

settings = get_settings()


def _get_llm(temperature: float = 0):
    if settings.default_llm_provider == "openai":
        return ChatOpenAI(
            model="gpt-4o",
            temperature=temperature,
            openai_api_key=settings.openai_api_key,
        )
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=temperature,
        google_api_key=settings.google_api_key,
    )


# ─── Auto-Summary Chain ───────────────────────────────────────────────────────

_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a data analyst. Given metadata about a dataset, provide a clear, "
     "insightful summary that a business user would find immediately useful.\n\n"
     "Include:\n"
     "- What the dataset is about (inferred from column names)\n"
     "- Size (rows × columns)\n"
     "- Key columns and their types\n"
     "- Notable statistics (min, max, mean for numeric columns)\n"
     "- Any obvious data quality issues (nulls, outliers)\n"
     "- 3-5 suggested questions someone could ask about this data\n"
     "Format in clean Markdown."
    ),
    ("human", "Dataset metadata:\n{metadata}"),
])

_summary_chain = _SUMMARY_PROMPT | _get_llm(temperature=0) | StrOutputParser()


def generate_dataframe_summary(df: pd.DataFrame, filename: str) -> str:
    """
    Auto-generate a business-friendly summary of a DataFrame.

    Args:
        df: The loaded Pandas DataFrame
        filename: Original filename (for context)

    Returns:
        Markdown-formatted summary string
    """
    # Build metadata string from DataFrame
    metadata_parts = [
        f"Filename: {filename}",
        f"Shape: {df.shape[0]} rows × {df.shape[1]} columns",
        f"Columns: {', '.join(df.columns.tolist())}",
        f"Data types:\n{df.dtypes.to_string()}",
        f"Null counts:\n{df.isnull().sum().to_string()}",
    ]

    # Add numeric statistics
    numeric_cols = df.select_dtypes(include="number")
    if not numeric_cols.empty:
        metadata_parts.append(f"Numeric statistics:\n{numeric_cols.describe().to_string()}")

    # Add sample rows
    metadata_parts.append(f"First 5 rows:\n{df.head(5).to_string()}")

    # Add value counts for categorical columns (limit to top 3)
    cat_cols = df.select_dtypes(include=["object", "category"]).columns[:3]
    for col in cat_cols:
        vc = df[col].value_counts().head(5)
        metadata_parts.append(f"Top values in '{col}':\n{vc.to_string()}")

    metadata = "\n\n".join(metadata_parts)

    try:
        summary = _summary_chain.invoke({"metadata": metadata})
        logger.info(f"Generated summary for '{filename}': {len(summary)} chars")
        return summary
    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        return (
            f"## {filename}\n\n"
            f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns\n\n"
            f"**Columns:** {', '.join(df.columns.tolist())}\n\n"
            f"*Auto-summary failed: {e}*"
        )


def query_dataframe(df: pd.DataFrame, question: str) -> str:
    """
    Answer a natural language question about a DataFrame using the CSV agent.

    Args:
        df: The Pandas DataFrame to query
        question: Natural language question

    Returns:
        Answer string with the agent's analysis
    """
    llm = _get_llm(temperature=0)

    try:
        agent = create_pandas_dataframe_agent(
            llm=llm,
            df=df,
            agent_type="tool-calling",
            verbose=False,
            allow_dangerous_code=True,
            handle_parsing_errors=True,
            max_iterations=10,
        )
        result = agent.invoke({"input": question})
        # result is a dict with "output" key
        return result.get("output", str(result))

    except Exception as e:
        logger.error(f"DataFrame query failed: {e}")
        return f"Query failed: {str(e)}"
