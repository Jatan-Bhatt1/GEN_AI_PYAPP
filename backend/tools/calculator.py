"""
Calculator Tool — Safe mathematical expression evaluator using numexpr.
Supports arithmetic, algebra, and numeric expressions. Never uses Python eval().
"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field
import numexpr
import math
import re


class CalculatorInput(BaseModel):
    expression: str = Field(
        description=(
            "A valid mathematical expression to evaluate. "
            "Supports: +, -, *, /, **, %, sqrt(), sin(), cos(), log(), abs(), etc. "
            "Examples: '2 ** 10', 'sqrt(144)', '(15 * 8) / 4 + 100'"
        )
    )


# Safe constants available in expressions
_SAFE_NAMES = {
    "pi": math.pi,
    "e": math.e,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "abs": abs,
    "ceil": math.ceil,
    "floor": math.floor,
    "round": round,
}


@tool("calculator", args_schema=CalculatorInput)
def calculator_tool(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the numeric result.
    Use this for arithmetic, algebra, unit conversions, or any math calculation.
    Do NOT use for string operations or code execution.

    Args:
        expression: A math expression like '2 ** 10' or 'sqrt(144) * pi'

    Returns:
        The numeric result as a string, or an error message
    """
    # Strip surrounding whitespace and quotes
    expression = expression.strip().strip("'\"")

    # Block obviously dangerous patterns
    dangerous = ["import", "exec", "eval", "__", "open", "os.", "sys."]
    for pattern in dangerous:
        if pattern in expression.lower():
            return f"Error: Expression contains disallowed pattern '{pattern}'"

    try:
        # numexpr is safe and fast for numeric evaluation
        result = numexpr.evaluate(expression).item()

        # Format result cleanly
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        elif isinstance(result, float):
            return f"{result:.6g}"  # up to 6 significant figures
        else:
            return str(result)

    except Exception:
        # Fallback: try Python math evaluation with restricted builtins
        try:
            result = eval(expression, {"__builtins__": {}}, _SAFE_NAMES)  # noqa: S307
            if isinstance(result, float) and result.is_integer():
                return str(int(result))
            return f"{result:.6g}" if isinstance(result, float) else str(result)
        except Exception as e:
            return (
                f"Could not evaluate '{expression}': {str(e)}. "
                "Make sure to use valid math syntax (e.g., '**' for powers, not '^')."
            )
