function Starter() {
    return (
        <div className="flex-grow-1 d-flex flex-column align-items-center justify-content-center bg-light p-4">
            <div className="text-center mb-5">
                <div style={{ fontSize: "3.5rem" }}>📄</div>
                <h1 className="fw-bold mt-2">Research Helper</h1>
                <p
                    className="text-muted mt-2"
                    style={{ maxWidth: "520px", fontSize: "1.05rem" }}
                >
                    Upload any research paper PDF and get an instant summary,
                    then ask questions about it in plain English.
                </p>
            </div>

            <div
                className="d-flex gap-4 flex-wrap justify-content-center"
                style={{ maxWidth: "720px" }}
            >
                {(
                    [
                        {
                            icon: "📤",
                            step: "1",
                            title: "Upload a Paper",
                            desc: "Select any research paper in PDF format from your device.",
                        },
                        {
                            icon: "🧠",
                            step: "2",
                            title: "Get a Summary",
                            desc: "The key problem, contributions, methods, and findings are extracted automatically.",
                        },
                        {
                            icon: "💬",
                            step: "3",
                            title: "Ask Questions",
                            desc: "Chat with the paper! Ask anything about its content, methods, or results.",
                        },
                    ] as const
                ).map(({ icon, step, title, desc }) => (
                    <div
                        key={step}
                        className="bg-white rounded-3 border p-4 text-center shadow-sm"
                        style={{ flex: "1 1 180px", maxWidth: "210px" }}
                    >
                        <div style={{ fontSize: "2rem" }}>{icon}</div>
                        <div className="text-muted small mt-1">Step {step}</div>
                        <h6 className="fw-semibold mt-2 mb-1">{title}</h6>
                        <p className="text-muted small mb-0">{desc}</p>
                    </div>
                ))}
            </div>

            <p className="text-muted mt-5 small">
                ← Click <strong>New Session</strong> in the sidebar to get
                started.
            </p>
        </div>
    );
}

export default Starter;
