import { useState, useEffect } from "react";

import Sidebar from "./components/Sidebar";
import Starter from "./components/Starter";
import PaperUpload from "./components/PaperUpload";
import Chat from "./components/Chat";

import type { Session, Message, PaperSummary } from "./types";

type View = "home" | "upload" | "chat";

function App() {
    const [view, setView] = useState<View>("home");
    const [sessions, setSessions] = useState<Session[]>(() => {
        try {
            const stored = localStorage.getItem("sessions");
            return stored ? (JSON.parse(stored) as Session[]) : [];
        } catch {
            return [];
        }
    });
    const [activeSessionId, setActiveSessionId] = useState<string | null>(() =>
        localStorage.getItem("activeSessionId"),
    );

    useEffect(() => {
        localStorage.setItem("sessions", JSON.stringify(sessions));
    }, [sessions]);

    useEffect(() => {
        if (activeSessionId)
            localStorage.setItem("activeSessionId", activeSessionId);
        else localStorage.removeItem("activeSessionId");
    }, [activeSessionId]);

    function handleUploadSuccess(
        sessionId: string,
        title: string,
        created_at: number,
        session_ttl: number,
        paper_summary: PaperSummary,
    ) {
        const formattedSummary = [
            `**Research Problem**\n${paper_summary.research_problem}`,
            paper_summary.key_contributions?.length
                ? `**Key Contributions**\n${paper_summary.key_contributions.map((c: string) => `• ${c}`).join("\n")}`
                : null,
            `**Method Overview**\n${paper_summary.method_overview}`,
            paper_summary.experimental_findings?.length
                ? `**Experimental Findings**\n${paper_summary.experimental_findings.map((f: string) => `• ${f}`).join("\n")}`
                : null,
            paper_summary.limitations?.length
                ? `**Limitations**\n${paper_summary.limitations.map((l: string) => `• ${l}`).join("\n")}`
                : null,
        ]
            .filter(Boolean)
            .join("\n\n");

        const newSession: Session = {
            id: sessionId,
            title: title,
            created_at: created_at,
            session_ttl: session_ttl,
            messages: [{ role: "assistant", content: formattedSummary }],
        };

        setSessions((prev) => [newSession, ...prev]);
        setActiveSessionId(sessionId);
        setView("chat");
    }

    function handleSelectSession(id: string) {
        setActiveSessionId(id);
        setView("chat");
    }

    function handleAddMessage(sessionId: string, message: Message) {
        setSessions((prev) =>
            prev.map((s) =>
                s.id === sessionId
                    ? { ...s, messages: [...s.messages, message] }
                    : s,
            ),
        );
    }

    const activeSession =
        sessions.find((s) => s.id === activeSessionId) ?? null;

    function renderMain() {
        if (view === "upload")
            return (
                <PaperUpload
                    onUploadSuccess={handleUploadSuccess}
                    onCancel={() => setView("home")}
                />
            );
        if (view === "chat" && activeSession)
            return (
                <Chat
                    session={activeSession}
                    onAddMessage={handleAddMessage}
                    onGoHome={() => setView("home")}
                />
            );
        return <Starter />;
    }

    return (
        <div className="d-flex vh-100">
            <Sidebar
                sessions={sessions}
                activeSessionId={activeSessionId}
                onSelectSession={handleSelectSession}
                onNewSession={() => setView("upload")}
            />
            {renderMain()}
        </div>
    );
}

export default App;
