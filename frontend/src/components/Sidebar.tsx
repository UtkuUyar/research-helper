import type { Session } from "../types";

type Props = {
    sessions: Session[];
    activeSessionId: string | null;
    onSelectSession: (id: string) => void;
    onNewSession: () => void;
};

function Sidebar({
    sessions,
    activeSessionId,
    onSelectSession,
    onNewSession,
}: Props) {
    return (
        <div
            className="d-flex flex-column bg-dark text-white p-3"
            style={{ width: "240px", minWidth: "240px" }}
        >
            <h5 className="mb-4">Sessions</h5>
            <button
                className="btn btn-outline-light btn-sm mb-4"
                onClick={onNewSession}
            >
                + New Session
            </button>
            <ul className="list-unstyled">
                {sessions.map((session) => (
                    <li
                        key={session.id}
                        className={`mb-2 p-2 rounded cursor-pointer ${session.id === activeSessionId ? "bg-secondary" : ""}`}
                        style={{ cursor: "pointer" }}
                        onClick={() => onSelectSession(session.id)}
                    >
                        {session.title}
                    </li>
                ))}
            </ul>
        </div>
    );
}

export default Sidebar;
