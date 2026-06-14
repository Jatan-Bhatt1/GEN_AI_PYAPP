"use client";
import { useState, useRef, KeyboardEvent } from "react";

interface Props {
    onSend: (message: string) => void;
    disabled?: boolean;
    placeholder?: string;
}

export default function ChatInput({ onSend, disabled, placeholder }: Props) {
    const [input, setInput] = useState("");
    const textareaRef = useRef<HTMLTextAreaElement>(null);

    const handleSend = () => {
        const trimmed = input.trim();
        if (!trimmed || disabled) return;
        onSend(trimmed);
        setInput("");
        if (textareaRef.current) {
            textareaRef.current.style.height = "auto";
        }
    };

    const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    const handleInput = () => {
        if (textareaRef.current) {
            textareaRef.current.style.height = "auto";
            textareaRef.current.style.height = `${Math.min(
                textareaRef.current.scrollHeight,
                200
            )}px`;
        }
    };

    return (
        <div className="glass p-2 flex items-end gap-2">
            <textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => {
                    setInput(e.target.value);
                    handleInput();
                }}
                onKeyDown={handleKeyDown}
                placeholder={
                    placeholder || "Type a message... (Enter to send, Shift+Enter for newline)"
                }
                disabled={disabled}
                rows={1}
                style={{ resize: "none", minHeight: "44px" }}
                className="flex-1 bg-transparent text-white placeholder-zinc-500 
                   text-sm outline-none px-3 py-2.5 leading-relaxed rounded-lg
                   focus:ring-1 focus:ring-white/20 transition-all duration-200
                   disabled:opacity-50"
            />
            <button
                onClick={handleSend}
                disabled={disabled || !input.trim()}
                className="w-10 h-10 rounded-xl bg-white hover:bg-zinc-200 
                   disabled:opacity-30 disabled:bg-zinc-800 disabled:cursor-not-allowed
                   flex items-center justify-center transition-all duration-200
                   hover:shadow-lg hover:shadow-white/10 flex-shrink-0"
            >
                {disabled ? (
                    <span className="w-4 h-4 border-2 border-zinc-500/30 border-t-zinc-400 
                          rounded-full animate-spin" />
                ) : (
                    <svg
                        className="w-4 h-4 text-black"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                    >
                        <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2.5}
                            d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                        />
                    </svg>
                )}
            </button>
        </div>
    );
}
