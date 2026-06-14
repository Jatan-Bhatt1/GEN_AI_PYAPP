"use client";
import { useState, useCallback, useRef, useEffect } from "react";
import { useDropzone } from "react-dropzone";
import Sidebar from "@/components/Sidebar";
import MessageBubble from "@/components/MessageBubble";
import ChatInput from "@/components/ChatInput";
import { uploadDocument, streamRAGQuery, getIndexedSources, Message, Source } from "@/lib/api";

interface IndexedSource {
    source: string;
    chunk_count: number;
}

export default function DocsPage() {
    const uid = () => Math.random().toString(36).slice(2, 11);
    const [messages, setMessages] = useState<Message[]>([
        {
            id: "rag-welcome",
            role: "assistant",
            content:
                "📄 **Document RAG Mode**\n\n" +
                "1. **Upload** a PDF, DOCX, or TXT file using the panel on the right\n" +
                "2. **Ask questions** about the document content\n" +
                "3. Answers will cite the source document and page number\n\n" +
                "_Follow-up questions work — I remember the conversation context._",
            timestamp: new Date(),
        },
    ]);
    const [isLoading, setIsLoading] = useState(false);
    const [isUploading, setIsUploading] = useState(false);
    const [uploadStatus, setUploadStatus] = useState<string>("");
    const [indexedSources, setIndexedSources] = useState<IndexedSource[]>([]);
    const [sessionId] = useState(() => `rag_${Math.random().toString(36).slice(2, 10)}`);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages]);

    // Load indexed sources on mount
    useEffect(() => {
        getIndexedSources()
            .then((data) => setIndexedSources(data.indexed_sources || []))
            .catch(console.error);
    }, []);

    // Dropzone setup
    const onDrop = useCallback(async (acceptedFiles: File[]) => {
        const file = acceptedFiles[0];
        if (!file) return;

        setIsUploading(true);
        setUploadStatus(`Uploading ${file.name}...`);

        try {
            const result = await uploadDocument(file);
            setUploadStatus(
                `✅ ${result.filename} indexed (${result.chunks_indexed} chunks)`
            );
            // Refresh sources list
            const sources = await getIndexedSources();
            setIndexedSources(sources.indexed_sources || []);

            // Add confirmation message to chat
            setMessages((prev) => [
                ...prev,
                {
                    id: uid(),
                    role: "assistant",
                    content: `✅ **${result.filename}** has been uploaded and indexed into ChromaDB.\n\n${result.chunks_indexed} text chunks are now searchable. You can start asking questions about this document!`,
                    timestamp: new Date(),
                },
            ]);
        } catch (err: any) {
            setUploadStatus(`❌ Upload failed: ${err.message}`);
        } finally {
            setIsUploading(false);
        }
    }, []);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: {
            "application/pdf": [".pdf"],
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
            "text/plain": [".txt"],
            "text/markdown": [".md"],
        },
        maxFiles: 1,
        disabled: isUploading,
    });

    const handleSend = async (question: string) => {
        if (isLoading) return;

        const userMsg: Message = {
            id: uid(),
            role: "user",
            content: question,
            timestamp: new Date(),
        };

        const aiMsgId = uid();
        const aiMsg: Message = {
            id: aiMsgId,
            role: "assistant",
            content: "",
            timestamp: new Date(),
            isStreaming: true,
        };

        setMessages((prev) => [...prev, userMsg, aiMsg]);
        setIsLoading(true);

        try {
            const { fullText, sources } = await streamRAGQuery(
                question,
                sessionId,
                (chunk) => {
                    setMessages((prev) =>
                        prev.map((m) =>
                            m.id === aiMsgId ? { ...m, content: m.content + chunk } : m
                        )
                    );
                }
            );

            // Update with sources
            setMessages((prev) =>
                prev.map((m) =>
                    m.id === aiMsgId
                        ? { ...m, content: fullText, sources, isStreaming: false }
                        : m
                )
            );
        } catch (err: any) {
            setMessages((prev) =>
                prev.map((m) =>
                    m.id === aiMsgId
                        ? {
                            ...m,
                            content: `❌ Error: ${err.message}`,
                            isStreaming: false,
                        }
                        : m
                )
            );
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="flex h-screen overflow-hidden">
            <Sidebar />

            {/* Chat Area */}
            <div className="flex-1 flex flex-col h-screen">
                <header
                    className="px-6 py-4 border-b flex items-center justify-between"
                    style={{
                        background: "rgba(255,255,255,0.02)",
                        borderColor: "rgba(255,255,255,0.06)",
                    }}
                >
                    <div>
                        <h2 className="text-base font-semibold text-white">📄 Document Q&A</h2>
                        <p className="text-xs text-zinc-500">
                            {indexedSources.length} document(s) indexed
                        </p>
                    </div>
                </header>

                <div className="flex flex-1 overflow-hidden">
                    {/* Messages */}
                    <div className="flex-1 flex flex-col overflow-hidden">
                        <div className="flex-1 overflow-y-auto px-6 py-6 space-y-6">
                            {messages.map((msg) => (
                                <MessageBubble key={msg.id} message={msg} />
                            ))}
                            <div ref={messagesEndRef} />
                        </div>

                        <div className="px-6 pb-6">
                            <ChatInput
                                onSend={handleSend}
                                disabled={isLoading}
                                placeholder="Ask a question about your documents..."
                            />
                        </div>
                    </div>

                    {/* Upload Panel */}
                    <div
                        className="w-72 border-l p-4 overflow-y-auto"
                        style={{
                            background: "rgba(255,255,255,0.02)",
                            borderColor: "rgba(255,255,255,0.06)",
                        }}
                    >
                        <h3 className="text-sm font-semibold text-zinc-300 mb-3">
                            Upload Document
                        </h3>

                        {/* Dropzone */}
                        <div
                            {...getRootProps()}
                            className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer
                         transition-all duration-200 ${isDragActive
                                    ? "border-white bg-white/5"
                                    : "border-white/10 hover:border-white/20 bg-transparent hover:bg-white/[0.02]"
                                } ${isUploading ? "opacity-50 cursor-not-allowed" : ""}`}
                        >
                            <input {...getInputProps()} />
                            <div className="text-3xl mb-2">
                                {isUploading ? "⏳" : isDragActive ? "📂" : "📁"}
                            </div>
                            <p className="text-xs text-zinc-400">
                                {isUploading
                                    ? "Uploading..."
                                    : isDragActive
                                        ? "Drop it here!"
                                        : "Drag & drop or click to upload"}
                            </p>
                            <p className="text-xs text-zinc-600 mt-1">PDF, DOCX, TXT, MD</p>
                        </div>

                        {uploadStatus && (
                            <p className="text-xs text-zinc-400 mt-3 p-2 glass rounded-lg">
                                {uploadStatus}
                            </p>
                        )}

                        {/* Indexed sources */}
                        {indexedSources.length > 0 && (
                            <div className="mt-4">
                                <h4 className="text-xs font-semibold text-zinc-400 mb-2">
                                    Indexed Documents
                                </h4>
                                <div className="space-y-2">
                                    {indexedSources.map((src) => (
                                        <div
                                            key={src.source}
                                            className="glass rounded-lg p-2"
                                        >
                                            <p className="text-xs text-zinc-300 font-medium truncate">
                                                {src.source}
                                            </p>
                                            <p className="text-xs text-zinc-500">
                                                {src.chunk_count} chunks
                                            </p>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
