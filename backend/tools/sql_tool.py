"""
SQL Tool — Natural language to PostgreSQL query executor.
Wraps SQLAlchemy to let agents safely query the database.
"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from sqlalchemy import text
from backend.database import SessionLocal
from loguru import logger
import re


class SQLQueryInput(BaseModel):
    query: str = Field(
        description=(
            "A raw SQL SELECT query to execute against the PostgreSQL database. "
            "Only SELECT statements are allowed. No INSERT, UPDATE, DELETE, DROP, etc. "
            "Example: 'SELECT table_name FROM information_schema.tables WHERE table_schema = \\'public\\''"
        )
    )
    limit: int = Field(
        default=20,
        description="Maximum number of rows to return (default: 20, max: 100)",
    )


# Patterns that indicate a dangerous (write) SQL statement
_WRITE_PATTERNS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|GRANT|REVOKE|EXEC|EXECUTE)\b",
    re.IGNORECASE,
)


@tool("execute_sql", args_schema=SQLQueryInput)
def sql_tool(query: str, limit: int = 20) -> str:
    """
    Execute a SELECT SQL query against the PostgreSQL database and return results.
    Use this tool to look up data, inspect tables, count records, or join tables.
    Only read-only SELECT queries are permitted.

    Args:
        query: A SELECT SQL statement
        limit: Maximum rows to return (capped at 100)

    Returns:
        Query results as a formatted table string, or an error message
    """
    # Safety: block all write operations
    if _WRITE_PATTERNS.search(query):
        return (
            "Error: Only SELECT statements are allowed. "
            "Write operations (INSERT, UPDATE, DELETE, DROP, etc.) are blocked."
        )

    # Enforce limit cap
    limit = min(limit, 100)

    # Inject LIMIT if not present
    query_stripped = query.strip().rstrip(";")
    if "LIMIT" not in query.upper():
        query_stripped = f"{query_stripped} LIMIT {limit}"

    db = SessionLocal()
    try:
        logger.info(f"Executing SQL: {query_stripped[:200]}")
        result = db.execute(text(query_stripped))
        rows = result.fetchall()
        columns = list(result.keys())

        if not rows:
            return "Query returned 0 rows."

        # Format as a simple table
        col_widths = [max(len(str(c)), max((len(str(r[i])) for r in rows), default=0)) for i, c in enumerate(columns)]
        header = " | ".join(str(c).ljust(col_widths[i]) for i, c in enumerate(columns))
        separator = "-+-".join("-" * w for w in col_widths)
        data_rows = [
            " | ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(columns)))
            for row in rows
        ]

        table = "\n".join([header, separator, *data_rows])
        return f"Results ({len(rows)} rows):\n\n{table}"

    except Exception as e:
        logger.error(f"SQL execution error: {e}")
        return f"SQL execution failed: {str(e)}"
    finally:
        db.close()


class ListTablesInput(BaseModel):
    schema_name: str = Field(default="public", description="PostgreSQL schema to list tables from")


@tool("list_database_tables", args_schema=ListTablesInput)
def list_tables_tool(schema_name: str = "public") -> str:
    """
    List all tables in the PostgreSQL database.
    Use this first to understand the database schema before writing SQL queries.

    Args:
        schema_name: The schema to inspect (default: 'public')

    Returns:
        List of table names and their column names
    """
    db = SessionLocal()
    try:
        result = db.execute(
            text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = :schema ORDER BY table_name"
            ),
            {"schema": schema_name},
        )
        tables = [row[0] for row in result.fetchall()]

        if not tables:
            return f"No tables found in schema '{schema_name}'."

        details = []
        for table in tables:
            col_result = db.execute(
                text(
                    "SELECT column_name, data_type FROM information_schema.columns "
                    "WHERE table_name = :table AND table_schema = :schema "
                    "ORDER BY ordinal_position"
                ),
                {"table": table, "schema": schema_name},
            )
            columns = [f"{r[0]} ({r[1]})" for r in col_result.fetchall()]
            details.append(f"Table: {table}\n  Columns: {', '.join(columns)}")

        return "\n\n".join(details)

    except Exception as e:
        return f"Failed to list tables: {str(e)}"
    finally:
        db.close()
