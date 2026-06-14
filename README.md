# Fullstack Generative AI Application

This project is a comprehensive Fullstack Generative AI application featuring a robust **Python/FastAPI** backend and a modern **Next.js/React** frontend. It is designed to handle complex LLM (Large Language Model) operations, including RAG (Retrieval-Augmented Generation), intelligent agents, and memory management.

## 🚀 Project Structure

The repository is divided into two main parts:

### Backend (`/backend`)
A powerful, modular Python backend built with **FastAPI**.
- **`agents/`**: Logic for AI agents (entities that can use tools and make decisions).
- **`api/`**: REST API endpoints and routes.
- **`chains/`**: Sequential operations and LLM chains.
- **`evaluations/`**: Scripts for testing and evaluating the AI's responses.
- **`memory/`**: Chat history and context retention.
- **`prompts/`**: System prompts and templates for instructing LLMs.
- **`rag/`**: Retrieval-Augmented Generation logic for fetching custom data context.
- **`tools/`**: Custom tools accessible by the agents.
- **`workflows/`**: Complex orchestration logic and multi-step processes.
- **`main.py`**: The main entry point for the FastAPI server.
- **`database.py` & `config.py`**: Database connections and application configurations.

### Frontend (`/frontend`)
A modern, responsive web application built with **Next.js**, **TypeScript**, and **Tailwind CSS**.
- **`app/`**: Next.js App Router for pages and layouts.
- **`components/`**: Reusable UI components (buttons, chat interfaces, etc.).
- **`lib/`**: Utility functions and frontend helpers.

### Data & Environment Storage
- **`uploads/`**: Directory for storing user-uploaded files.
- **`vectorstore/`**: Local vector database files powering the RAG system.
- *(Note: `uploads/`, `vectorstore/`, and `.env` files are ignored by git to keep your data secure and repository clean)*

## 🛠️ Getting Started

### Prerequisites
- Node.js (v18+)
- Python (3.9+)

### 1. Backend Setup
Navigate to the root directory and activate your virtual environment:
```bash
# Activate virtual environment (Windows)
.\venv\Scripts\activate

# Activate virtual environment (macOS/Linux)
source venv/bin/activate

# Install dependencies (if not already installed)
pip install -r backend/requirements.txt

# Run the FastAPI server
uvicorn backend.main:app --reload --port 8000
```

### 2. Frontend Setup
Open a new terminal, navigate to the frontend directory:
```bash
cd frontend

# Install dependencies
npm install

# Run the Next.js development server
npm run dev
```

The frontend will be available at `http://localhost:3000` and the backend API at `http://localhost:8000`.

## ⚙️ Environment Variables
Make sure to create a `.env` file in the root directory (and `.env.local` in `/frontend` if needed) to store your API keys (e.g., OpenAI, Gemini) and database URIs.
