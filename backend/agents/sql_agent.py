"""
SQL Agent — Natural language to PostgreSQL using LangGraph ReAct agent.
Uses SQLDatabaseToolkit and langgraph.prebuilt.create_react_agent (LangChain 1.x compatible).
"""

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from loguru import logger

from backend.config import get_settings

settings = get_settings()

_SQL_SYSTEM_PROMPT = SystemMessage(content=(
    "You are an expert SQL analyst connected to a PostgreSQL database.\n"
    "Rules you MUST follow:\n"
    "1. ALWAYS start by listing tables with the sql_db_list_tables tool\n"
    "2. Then inspect relevant table schemas with sql_db_schema\n"
    "3. Only write safe SELECT queries — never INSERT, UPDATE, DELETE, DROP\n"
    "4. Always add a LIMIT clause to avoid huge result sets\n"
    "5. Explain your findings in plain English after running the query\n"
    "Format SQL queries in ```sql code blocks in your final answer."
))


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
        model="gemini-2.0-flash",
        temperature=0,
        google_api_key=settings.google_api_key,
    )


def create_sql_agent():
    """
    Build a SQL agent connected to the configured PostgreSQL database.

    Uses SQLDatabaseToolkit which provides:
    - sql_db_list_tables: Lists all tables
    - sql_db_schema: Gets schema + sample rows for tables
    - sql_db_query: Executes SELECT queries
    - sql_db_query_checker: Validates SQL syntax before execution

    Returns:
        Compiled LangGraph agent, or None if DB connection fails
    """
    llm = _get_llm()

    try:
        db = SQLDatabase.from_uri(
            settings.database_url,
            sample_rows_in_table_info=3,
        )
        logger.info(f"SQL agent connected. Tables: {db.get_table_names()}")
    except Exception as e:
        logger.error(f"SQL agent: cannot connect to database: {e}")
        logger.warning("SQL agent unavailable — start PostgreSQL and check DATABASE_URL")
        return None

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=_SQL_SYSTEM_PROMPT,
    )

    logger.info("SQL agent created successfully")
    return agent


def invoke_sql_agent(query: str) -> dict:
    """
    Helper to invoke the SQL agent and return clean output.

    Args:
        query: Natural language database question

    Returns:
        dict with 'output' (str) and 'messages' (list)
    """
    agent = create_sql_agent()
    if agent is None:
        return {
            "output": "SQL agent unavailable: cannot connect to database. "
                      "Ensure PostgreSQL is running and DATABASE_URL is correct.",
            "messages": [],
        }

    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    messages = result.get("messages", [])
    final_output = messages[-1].content if messages else "No response generated."

    return {
        "output": final_output,
        "messages": [{"role": m.type, "content": m.content} for m in messages],
    }
