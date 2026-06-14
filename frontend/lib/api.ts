/**
 * API Client — All calls to the FastAPI backend.
 * 
 * Base URL: http://localhost:8000
 * All streaming endpoints use the native fetch() ReadableStream API.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface Message {
    id: string;
    role: "user" | "assistant";
    content: string;
    sources?: Source[];
    timestamp: Date;
    isStreaming?: boolean;
}

export interface Source {
    source: string;
    page: string | number;
    content_preview: string;
}

export interface ChatRequest {
    message: string;
    session_id: string;
    use_memory?: boolean;
}


// ─── Chat API ──────────────────────────────────────────────────────────────

/**
 * Stream a chat response from the backend.
 * 
 * Uses Server-Sent Events pattern: reads chunks from the streaming response
 * and calls onChunk() for each token received.
 * 
 * @param message - User's message
 * @param sessionId - Session ID for conversation memory
 * @param onChunk - Called with each text token as it arrives
 * @returns Full response text when streaming completes
 */
export async function streamChat(
    message: string,
    sessionId: string,
    onChunk: (chunk: string) => void
): Promise<string> {
    const response = await fetch(`${API_BASE}/api/chat/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            message,
            session_id: sessionId,
            use_memory: true,
        }),
    });

    if (!response.ok) {
        throw new Error(`Chat API error: ${response.status}`);
    }

    const reader = response.body!.getReader();
    const decoder = new TextDecoder();
    let fullText = "";

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        fullText += chunk;
        onChunk(chunk);
    }

    return fullText;
}


// ─── RAG / Documents API ──────────────────────────────────────────────────

/**
 * Upload a document to the RAG index.
 */
export async function uploadDocument(file: File): Promise<{
    filename: string;
    chunks_indexed: number;
    message: string;
}> {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${API_BASE}/api/documents/upload`, {
        method: "POST",
        body: formData,
    });

    if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Upload failed");
    }

    return response.json();
}

/**
 * Stream a RAG query response.
 */
export async function streamRAGQuery(
    question: string,
    sessionId: string,
    onChunk: (chunk: string) => void
): Promise<{ fullText: string; sources: Source[] }> {
    const response = await fetch(`${API_BASE}/api/documents/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, session_id: sessionId }),
    });

    if (!response.ok) {
        throw new Error(`RAG API error: ${response.status}`);
    }

    const reader = response.body!.getReader();
    const decoder = new TextDecoder();
    let fullText = "";
    let sources: Source[] = [];

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });

        // Check for sources marker at end of stream
        if (chunk.includes("[[SOURCES]]")) {
            const parts = chunk.split("[[SOURCES]]");
            const textPart = parts[0];
            const sourcesPart = parts[1];

            if (textPart) {
                fullText += textPart;
                onChunk(textPart);
            }

            if (sourcesPart) {
                try {
                    sources = JSON.parse(sourcesPart);
                } catch (e) {
                    console.error("Failed to parse sources:", e);
                }
            }
        } else {
            fullText += chunk;
            onChunk(chunk);
        }
    }

    return { fullText, sources };
}

/**
 * Get list of indexed documents.
 */
export async function getIndexedSources(): Promise<{
    indexed_sources: Array<{ source: string; chunk_count: number }>;
    total_sources: number;
}> {
    const response = await fetch(`${API_BASE}/api/documents/sources`);
    return response.json();
}


// ─── Agent API ────────────────────────────────────────────────────────────

/**
 * Run a multi-agent workflow (non-streaming).
 */
export async function runWorkflow(task: string): Promise<{
    final_output: string;
    worker_sequence: string[];
    iteration_count: number;
}> {
    const response = await fetch(`${API_BASE}/api/workflows/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ task }),
    });

    if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Workflow failed");
    }

    return response.json();
}


// ─── Health Check ─────────────────────────────────────────────────────────

export async function checkHealth(): Promise<{
    status: string;
    llm_provider: string;
    services: Record<string, boolean | string>;
}> {
    const response = await fetch(`${API_BASE}/health`);
    return response.json();
}
