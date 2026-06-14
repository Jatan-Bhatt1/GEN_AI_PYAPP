"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV_ITEMS = [
    { href: "/", label: "Chat", icon: "💬", desc: "Streaming AI chat" },
    { href: "/docs", label: "Documents", icon: "📄", desc: "Upload & ask questions" },
    { href: "/agent", label: "Agent", icon: "🤖", desc: "Multi-agent workflows" },
];

export default function Sidebar() {
    const pathname = usePathname();

    return (
        <aside
            className="w-64 h-screen flex flex-col py-6 px-4 border-r"
            style={{
                background: "rgba(255,255,255,0.02)",
                borderColor: "rgba(255,255,255,0.06)",
            }}
        >
            {/* Logo */}
            <div className="mb-8 px-2">
                <h1 className="font-semibold text-zinc-100 tracking-tight">Enterprise AI</h1>
                <p className="text-xs text-zinc-500 font-medium tracking-wide">WORKSPACE</p>
            </div>

            {/* Navigation */}
            <nav className="flex-1 space-y-1">
                {NAV_ITEMS.map((item) => {
                    const active = pathname === item.href;
                    return (
                        <Link
                            key={item.href}
                            href={item.href}
                            className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all duration-200 text-sm font-medium ${
                                active
                                    ? "bg-white/10 text-white shadow-sm"
                                    : "text-zinc-400 hover:text-zinc-100 hover:bg-white/5"
                            }`}
                        >
                            <span className="text-lg">{item.icon}</span>
                            <div>
                                <p className={`text-sm font-medium ${active ? "text-white" : ""}`}>
                                    {item.label}
                                </p>
                                <p className="text-xs text-zinc-600">{item.desc}</p>
                            </div>
                        </Link>
                    );
                })}
            </nav>

            {/* Footer */}
            <div className="pt-4 border-t border-white/5">
                <p className="text-xs text-zinc-600 font-medium px-2 tracking-wide">
                    FastAPI ⚡ LangGraph
                </p>
            </div>
        </aside>
    );
}
