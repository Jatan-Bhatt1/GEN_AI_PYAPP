"use client";
import { useState, useRef, useEffect } from "react";
import Sidebar from "@/components/Sidebar";
import MessageBubble from "@/components/MessageBubble";
import ChatInput from "@/components/ChatInput";
import { streamChat, Message } from "@/lib/api";

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome-msg",
      role: "assistant",
      content:
        "👋 Hello! I'm your **Enterprise AI Assistant**.\n\n" +
        "I can help you with:\n" +
        "- 💬 **Chat** — ask me anything\n" +
        "- 📄 **Documents** → go to the Documents page to upload files and ask questions\n" +
        "- 🤖 **Agent** → go to Agent page for complex multi-step research\n\n" +
        "What would you like to do?",
      timestamp: new Date(),
    },
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId] = useState(() => `chat_${Math.random().toString(36).slice(2, 10)}`);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async (userText: string) => {
    if (isLoading) return;

    const uid = () => Math.random().toString(36).slice(2, 11);

    const userMsg: Message = {
      id: uid(),
      role: "user",
      content: userText,
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
      await streamChat(userText, sessionId, (chunk) => {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === aiMsgId ? { ...m, content: m.content + chunk } : m
          )
        );
      });
    } catch (err: any) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === aiMsgId
            ? {
                ...m,
                content: `❌ Error: ${err.message}. Make sure the backend is running at http://localhost:8000`,
                isStreaming: false,
              }
            : m
        )
      );
    } finally {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === aiMsgId ? { ...m, isStreaming: false } : m
        )
      );
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />

      <div className="flex-1 flex flex-col h-screen">
        {/* Header */}
        <header
          className="px-6 py-4 border-b flex items-center justify-between"
          style={{
            background: "rgba(255,255,255,0.02)",
            borderColor: "rgba(255,255,255,0.06)",
          }}
        >
          <div>
            <h2 className="text-base font-semibold text-white">💬 Chat</h2>
            <p className="text-xs text-zinc-500">Session: {sessionId}</p>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-xs text-zinc-400">Connected</span>
          </div>
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-6 py-6 space-y-6">
          {messages.map((msg) => (
            <MessageBubble key={msg.id} message={msg} />
          ))}
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="px-6 pb-6">
          <ChatInput
            onSend={handleSend}
            disabled={isLoading}
            placeholder="Ask me anything... (memory enabled)"
          />
          <p className="text-xs text-zinc-600 mt-2 text-center">
            Shift+Enter for new line · Responses stream in real-time
          </p>
        </div>
      </div>
    </div>
  );
}
