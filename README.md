# Fullstack Generative AI Application

Welcome to the **Fullstack Generative AI Application**! This project combines a powerful, modular **Python/FastAPI** backend with a modern, responsive **Next.js/React** frontend. It is designed to handle complex Large Language Model (LLM) operations, including Retrieval-Augmented Generation (RAG), intelligent agents with tools, and long-term conversational memory.

## 🌟 Key Features
- **Intelligent Agents:** AI entities that can make decisions and use tools (web search, weather API, etc.).
- **RAG Capabilities:** Seamlessly query your custom documents using a local vector store.
- **Conversational Memory:** Retain context and summarize long chat histories.
- **Modern UI:** Built with Next.js, Tailwind CSS, and TypeScript for a clean, responsive user experience.
- **Multi-LLM Support:** Easily switch between OpenAI, Google (Gemini), and Groq.

---

## 🛠️ Prerequisites

Before you begin, ensure you have the following installed on your machine:
- [Git](https://git-scm.com/downloads)
- [Node.js](https://nodejs.org/) (v18 or higher)
- [Python](https://www.python.org/downloads/) (3.9 or higher)
- [PostgreSQL](https://www.postgresql.org/) (for database)
- [Redis](https://redis.io/) (for memory buffer)

---

## 🚀 Installation Guide

Follow these steps to get a working copy of the project on your local machine.

### 1. Clone the Repository
First, clone the forked repository to your local machine and navigate into the project directory:
```bash
git clone https://github.com/YOUR_USERNAME/GEN_AI_PYAPP.git
cd GEN_AI_PYAPP
```

### 2. Environment Variables Setup
You need to configure your environment variables for the backend to function correctly (API keys, database URLs, etc.).

1. In the root directory, create a copy of the `.env.example` file and name it `.env`:
   ```bash
   cp .env.example .env
   ```
2. Open the `.env` file and fill in your specific API keys and configuration details (e.g., `OPENAI_API_KEY`, `DATABASE_URL`, `REDIS_URL`).

### 3. Backend Setup (FastAPI)
The backend requires a Python virtual environment to manage dependencies securely.

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - **Windows:**
     ```bash
     .\venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```

3. Install the required Python packages:
   ```bash
   pip install -r backend/requirements.txt
   ```

4. Start the backend FastAPI server:
   ```bash
   uvicorn backend.main:app --reload --port 8000
   ```
   *The backend API will now be running at `http://localhost:8000`.*

### 4. Frontend Setup (Next.js)
With the backend running, open a **new terminal window/tab**, and set up the frontend web app.

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install the Node.js dependencies:
   ```bash
   npm install
   ```

3. Start the Next.js development server:
   ```bash
   npm run dev
   ```
   *The frontend will now be accessible in your browser at `http://localhost:3000`.*

---

## 📂 Project Structure Overview

- **`backend/`**: Contains the FastAPI application.
  - `agents/`, `chains/`, `tools/`: Core LLM logic and LangChain integrations.
  - `rag/`, `memory/`: Logic for vector search and conversation history.
  - `api/`, `main.py`: REST API routes and server entry point.
- **`frontend/`**: Contains the Next.js frontend application.
  - `app/`, `components/`: React components and page routing.
- **`uploads/`**: Directory for user-uploaded files (git-ignored).
- **`vectorstore/`**: Local vector database files for RAG (git-ignored).

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

---
*Happy Coding!* 🚀
