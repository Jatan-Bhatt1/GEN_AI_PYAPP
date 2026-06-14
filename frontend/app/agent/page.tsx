"use client";
import { useState } from "react";
import Sidebar from "@/components/Sidebar";
import { runWorkflow } from "@/lib/api";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface WorkflowResult {
    final_output: string;
    worker_sequence: string[];
    iteration_count: number;
}

const EXAMPLE_TASKS = [
    "Research the top 3 benefits of renewable energy and write a brief executive summary.",
    "Create a 5-step action plan for launching a mobile app startup in 30 days.",
    "Analyze the pros and cons of Python vs JavaScript for AI development and write a comparison report.",
];

export default function AgentPage() {
    const [task, setTask] = useState("");
    const [isRunning, setIsRunning] = useState(false);
    const [result, setResult] = useState<WorkflowResult | null>(null);
    const [error, setError] = useState("");

    const handleRun = async () => {
        if (!task.trim() || isRunning) return;
        setIsRunning(true);
        setResult(null);
        setError("");

        try {
            const res = await runWorkflow(task);
            setResult(res);
        } catch (err: any) {
            setError(err.message);
        } finally {
            setIsRunning(false);
        }
    };

    return (
        <div className="flex h-screen overflow-hidden">
            <Sidebar />

            <div className="flex-1 overflow-y-auto">
                <header
                    className="px-8 py-5 border-b sticky top-0 z-10"
                    style={{
                        background: "rgba(10,10,15,0.9)",
                        backdropFilter: "blur(20px)",
                        borderColor: "rgba(255,255,255,0.06)",
                    }}
                >
                    <h2 className="text-base font-semibold text-white">
                        🤖 Multi-Agent Workflow
                    </h2>
                    <p className="text-xs text-zinc-500 mt-0.5">
                        Supervisor → Researcher → Analyst → Writer → Planner
                    </p>
                </header>

                <div className="max-w-3xl mx-auto px-8 py-8">
                    {/* Task input */}
                    <div className="glass rounded-2xl p-6 mb-6">
                        <label className="block text-sm font-medium text-zinc-300 mb-3">
                            Describe your complex task
                        </label>
                        <textarea
                            value={task}
                            onChange={(e) => setTask(e.target.value)}
                            placeholder="e.g., Research the top 5 AI companies by market cap, analyze their growth trends, and write a structured investment report..."
                            rows={4}
                            className="w-full bg-transparent text-white placeholder-zinc-500 
                        text-sm outline-none resize-none leading-relaxed"
                            disabled={isRunning}
                        />
                        <div className="flex items-center justify-between mt-4">
                            <p className="text-xs text-zinc-500">
                                ⏱ Complex tasks may take 1-3 minutes
                            </p>
                            <button
                                onClick={handleRun}
                                disabled={isRunning || !task.trim()}
                                className="px-6 py-2.5 rounded-xl bg-white hover:bg-zinc-200 
                          text-black text-sm font-medium transition-all duration-200
                          disabled:opacity-30 disabled:bg-zinc-800 disabled:text-white disabled:cursor-not-allowed
                          hover:shadow-lg hover:shadow-white/10"
                            >
                                {isRunning ? (
                                    <span className="flex items-center gap-2">
                                        <span className="w-3 h-3 border-2 border-zinc-500/30 border-t-zinc-400 rounded-full animate-spin" />
                                        Running agents...
                                    </span>
                                ) : (
                                    "🚀 Run Workflow"
                                )}
                            </button>
                        </div>
                    </div>

                    {/* Example tasks */}
                    {!result && !isRunning && (
                        <div className="mb-6">
                            <p className="text-xs text-zinc-500 mb-3 font-medium uppercase tracking-wider">
                                Example Tasks
                            </p>
                            <div className="space-y-2">
                                {EXAMPLE_TASKS.map((t, i) => (
                                    <button
                                        key={i}
                                        onClick={() => setTask(t)}
                                        className="w-full text-left text-sm text-zinc-400 hover:text-zinc-200
                              glass rounded-xl px-4 py-3 transition-all duration-200 hover:bg-white/5"
                                    >
                                        {t}
                                    </button>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Loading state */}
                    {isRunning && (
                        <div className="glass rounded-2xl p-8 text-center">
                            <div className="w-12 h-12 border-2 border-zinc-500/30 border-t-zinc-400 
                             rounded-full animate-spin mx-auto mb-4" />
                            <p className="text-zinc-300 font-medium">Agents working...</p>
                            <p className="text-zinc-500 text-sm mt-1">
                                Supervisor → routing to workers
                            </p>
                        </div>
                    )}

                    {/* Error */}
                    {error && (
                        <div className="glass border border-red-500/20 rounded-2xl p-4 mb-6">
                            <p className="text-red-400 text-sm">❌ {error}</p>
                        </div>
                    )}

                    {/* Result */}
                    {result && (
                        <div className="space-y-4">
                            {/* Worker sequence */}
                            <div className="glass rounded-xl p-4">
                                <p className="text-xs text-zinc-400 font-medium mb-2">
                                    Worker Sequence ({result.iteration_count} iterations)
                                </p>
                                <div className="flex items-center gap-2 flex-wrap">
                                    {result.worker_sequence.map((w, i) => (
                                        <>
                                            <span
                                                key={i}
                                                className="text-xs px-3 py-1 rounded-full bg-white/5 
                                  border border-white/10 text-zinc-300 capitalize shadow-sm"
                                            >
                                                {w === "researcher" ? "🔍" : w === "analyst" ? "📊" : w === "writer" ? "✍️" : "📋"} {w}
                                            </span>
                                            {i < result.worker_sequence.length - 1 && (
                                                <span key={`arrow-${i}`} className="text-zinc-600 text-xs">→</span>
                                            )}
                                        </>
                                    ))}
                                </div>
                            </div>

                            {/* Final output */}
                            <div className="glass rounded-2xl p-6">
                                <p className="text-xs text-zinc-400 font-medium mb-4 uppercase tracking-wider">
                                    Final Output
                                </p>
                                <div className="prose-dark text-sm">
                                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                        {result.final_output}
                                    </ReactMarkdown>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}