"""
Analytics API — CSV/Excel upload and analysis endpoints.

Endpoints:
  POST   /api/analytics/upload               — Upload CSV or Excel file
  POST   /api/analytics/query                — Ask questions about uploaded file
  GET    /api/analytics/summary/{file_id}    — Get auto-generated summary
  GET    /api/analytics/files                — List all uploaded files
  DELETE /api/analytics/files/{file_id}      — Delete an uploaded file
"""

import os
import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import pandas as pd
from loguru import logger

from backend.config import get_settings
from backend.chains.analysis_chain import generate_dataframe_summary, query_dataframe

router = APIRouter(prefix="/api/analytics", tags=["Analytics"])
settings = get_settings()

# Directory where uploaded files are saved
UPLOAD_DIR = Path(settings.upload_dir)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# In-memory registry of uploaded files (file_id → metadata)
# In production, store this in PostgreSQL
_file_registry: dict[str, dict] = {}

# Supported file types
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


# ─── Models ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    file_id: str = Field(..., description="File ID returned by /upload")
    question: str = Field(..., min_length=5, description="Natural language question about the data")


# ─── Helper ───────────────────────────────────────────────────────────────────

def _load_dataframe(file_id: str) -> pd.DataFrame:
    """Load a DataFrame from the uploaded file."""
    if file_id not in _file_registry:
        raise HTTPException(status_code=404, detail=f"File '{file_id}' not found. Upload it first.")

    file_info = _file_registry[file_id]
    file_path = Path(file_info["path"])

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File '{file_id}' missing from disk.")

    ext = file_path.suffix.lower()
    try:
        if ext == ".csv":
            df = pd.read_csv(file_path)
        elif ext in {".xlsx", ".xls"}:
            df = pd.read_excel(file_path)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read file: {e}")

    return df


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/upload", summary="Upload CSV or Excel file for analysis")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a CSV or Excel file. Returns a `file_id` to use in subsequent requests.

    **Supported formats:** `.csv`, `.xlsx`, `.xls`

    **Response:**
    ```json
    {
      "file_id": "abc123",
      "filename": "sales_data.csv",
      "rows": 5000,
      "columns": 12,
      "column_names": ["date", "revenue", "region", ...],
      "summary": "## sales_data.csv\\n\\n**Shape:** 5000 rows × 12 columns\\n..."
    }
    ```
    """
    # Validate file extension
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    # Generate unique file ID and save to disk
    file_id = str(uuid.uuid4())[:8]
    safe_name = f"{file_id}_{file.filename}"
    file_path = UPLOAD_DIR / safe_name

    content = await file.read()
    file_path.write_bytes(content)
    logger.info(f"Saved upload: {file_path} ({len(content)} bytes)")

    # Load and inspect the DataFrame
    try:
        if ext == ".csv":
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
    except Exception as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Could not parse file: {e}")

    # Register file
    _file_registry[file_id] = {
        "file_id": file_id,
        "filename": file.filename,
        "path": str(file_path),
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
    }

    # Auto-generate summary (runs LLM)
    logger.info(f"Generating auto-summary for '{file.filename}'...")
    summary = generate_dataframe_summary(df, file.filename)
    _file_registry[file_id]["summary"] = summary

    return {
        "file_id": file_id,
        "filename": file.filename,
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "summary": summary,
    }


@router.post("/query", summary="Ask a natural language question about uploaded data")
async def query_data(request: QueryRequest):
    """
    Ask a natural language question about a previously uploaded CSV/Excel file.

    The AI uses Python/Pandas to analyze the data and answer your question.

    **Example questions:**
    - "What is the total revenue by region?"
    - "Which product has the highest return rate?"
    - "Show me the top 5 customers by order value"
    - "What is the month-over-month growth rate?"
    - "Are there any missing values? Which columns have the most?"

    **Note:** Complex queries may take 15-30 seconds.
    """
    logger.info(f"Analytics query | file_id={request.file_id} | question='{request.question}'")

    df = _load_dataframe(request.file_id)
    file_info = _file_registry[request.file_id]

    try:
        answer = query_dataframe(df, request.question)
    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    return {
        "file_id": request.file_id,
        "filename": file_info["filename"],
        "question": request.question,
        "answer": answer,
        "data_shape": {"rows": file_info["rows"], "columns": file_info["columns"]},
    }


@router.get("/summary/{file_id}", summary="Get auto-generated summary for uploaded file")
async def get_summary(file_id: str):
    """
    Get the auto-generated data summary for an uploaded file.

    The summary is generated automatically on upload and includes:
    - Dataset overview and column descriptions
    - Key statistics
    - Data quality information
    - Suggested questions to ask
    """
    if file_id not in _file_registry:
        raise HTTPException(status_code=404, detail=f"File '{file_id}' not found.")

    file_info = _file_registry[file_id]

    # If summary doesn't exist yet (unlikely but defensive), regenerate it
    if "summary" not in file_info:
        df = _load_dataframe(file_id)
        file_info["summary"] = generate_dataframe_summary(df, file_info["filename"])

    return {
        "file_id": file_id,
        "filename": file_info["filename"],
        "rows": file_info["rows"],
        "columns": file_info["columns"],
        "column_names": file_info["column_names"],
        "summary": file_info["summary"],
    }


@router.get("/files", summary="List all uploaded files")
async def list_files():
    """List all files currently uploaded and available for analysis."""
    return {
        "files": [
            {
                "file_id": v["file_id"],
                "filename": v["filename"],
                "rows": v["rows"],
                "columns": v["columns"],
            }
            for v in _file_registry.values()
        ],
        "total": len(_file_registry),
    }


@router.delete("/files/{file_id}", summary="Delete an uploaded file")
async def delete_file(file_id: str):
    """Delete a previously uploaded file from disk and registry."""
    if file_id not in _file_registry:
        raise HTTPException(status_code=404, detail=f"File '{file_id}' not found.")

    file_info = _file_registry.pop(file_id)
    file_path = Path(file_info["path"])
    if file_path.exists():
        file_path.unlink()
        logger.info(f"Deleted file: {file_path}")

    return {"message": f"File '{file_info['filename']}' deleted.", "file_id": file_id}
