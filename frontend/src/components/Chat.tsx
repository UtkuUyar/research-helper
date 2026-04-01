import { useState } from "react";
import ReactMarkdown from "react-markdown";
import type { Session, Message } from "../types";

const API_URL = import.meta.env.VITE_BACKEND_API_ENDPOINT;

type Props = {
    session: Session;
    onAddMessage: (sessionId: string, message: Message) => void;
    onGoHome: () => void;
};

function Chat({ session, onAddMessage, onGoHome }: Props) {
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);

    const isExpired =
        Date.now() / 1000 - session.created_at > session.session_ttl;

    async function handleSend(e: React.SubmitEvent<HTMLFormElement>) {
        e.preventDefault();
        if (!input.trim()) return;

        const userMsg: Message = { role: "user", content: input };
        onAddMessage(session.id, userMsg);
        setInput("");
        setLoading(true);

        try {
            const response = await fetch(`${API_URL}/api/chat`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    session_id: session.id,
                    message: input,
                }),
            });
            if (!response.ok) throw new Error("Chat request failed.");
            const data = await response.json();
            onAddMessage(session.id, {
                role: "assistant",
                content: data.answer,
            });
        } catch (err) {
            onAddMessage(session.id, {
                role: "assistant",
                content: "Error: could not get a response.",
            });
        } finally {
            setLoading(false);
        }
    }

    return (
        <div className="flex-grow-1 d-flex flex-column bg-light">
            {/* Header */}
            <div className="p-3 border-bottom d-flex align-items-center gap-3 bg-white">
                <button
                    className="btn btn-outline-secondary btn-sm"
                    onClick={onGoHome}
                >
                    ← Home
                </button>
                <span className="fw-semibold">{session.title}</span>
            </div>

            {/* Messages */}
            <div className="flex-grow-1 p-4 overflow-auto">
                {session.messages.map((msg, i) => (
                    <div
                        key={i}
                        className={`mb-3 d-flex ${msg.role === "user" ? "justify-content-end" : "justify-content-start"}`}
                    >
                        <div
                            className={`p-3 rounded ${msg.role === "user" ? "bg-primary text-white" : "bg-white border"}`}
                            style={{ maxWidth: "70%" }}
                        >
                            {msg.role === "assistant" ? (
                                <div className="markdown-body">
                                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                                </div>
                            ) : (
                                msg.content
                            )}
                        </div>
                    </div>
                ))}
                {loading && (
                    <div className="text-muted fst-italic">Thinking...</div>
                )}
            </div>

            {/* Input */}
            {isExpired ? (
                <div className="p-3 border-top text-center text-muted bg-white">
                    This session has expired. You can read the chat but can no
                    longer send messages.
                </div>
            ) : (
                <form
                    className="p-3 border-top d-flex gap-2"
                    onSubmit={handleSend}
                >
                    <input
                        type="text"
                        className="form-control"
                        placeholder="Ask a question about the paper..."
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        disabled={loading}
                    />
                    <button
                        type="submit"
                        className="btn btn-primary"
                        disabled={loading || !input.trim()}
                    >
                        Send
                    </button>
                </form>
            )}
        </div>
    );
}

export default Chat;
