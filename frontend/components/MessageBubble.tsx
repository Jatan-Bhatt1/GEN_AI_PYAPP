"use client";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Message } from "@/lib/api";

interface Props {
    message: Message;
}

export default function MessageBubble({ message }: Props) {
    const isUser = message.role === "user";

    return (
        <div
            className={`flex gap-3 message-enter ${isUser ? "flex-row-reverse" : "flex-row"
                }`}
        >
            {/* Avatar */}
            <div
                className={`w-9 h-9 rounded-xl flex items-center justify-center text-sm font-bold flex-shrink-0 shadow-sm ${isUser
                    ? "bg-white text-black"
                    : "bg-zinc-900 text-white border border-white/10"
                    }`}
            >
                {isUser ? "U" : "AI"}
            </div>

            {/* Message content */}
            <div
                className={`max-w-[75%] rounded-2xl px-4 py-3 shadow-sm ${isUser
                    ? "bg-white border border-white/10 text-black rounded-tr-sm"
                    : "glass text-zinc-100 rounded-tl-sm"
                    } ${message.isStreaming ? "typing-cursor" : ""}`}
            >
                {isUser ? (
                    <p className="text-sm leading-relaxed">{message.content}</p>
                ) : (
                    <div className="prose-dark text-sm">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                            {message.content || ""}
                        </ReactMarkdown>
                    </div>
                )}

                {/* Source citations */}
                {message.sources && message.sources.length > 0 && (
                    <div className="mt-3 pt-3 border-t border-white/10">
                        <p className="text-xs text-slate-400 font-medium mb-2">
                            📚 Sources
                        </p>
                        <div className="flex flex-wrap gap-2">
                            {message.sources.map((src, i) => (
                                <div
                                    key={i}
                                    className="text-xs bg-white/5 border border-white/10 
                             rounded-lg px-2 py-1 text-zinc-300 hover:bg-white/10 transition-colors cursor-default"
                                >
                                    {src.source}
                                    {src.page !== "N/A" && ` · p.${src.page}`}
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {/* Timestamp */}
                <p className="text-xs text-slate-500 mt-2">
                    {message.timestamp.toLocaleTimeString([], {
                        hour: "2-digit",
                        minute: "2-digit",
                    })}
                </p>
            </div>
        </div>
    );
}
