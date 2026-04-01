import { useState } from "react";
import type { PaperSummary } from "../types";

const API_URL = import.meta.env.VITE_BACKEND_API_ENDPOINT;

type Props = {
    onUploadSuccess: (
        sessionId: string,
        title: string,
        created_at: number,
        session_ttl: number,
        summary: PaperSummary,
    ) => void;
    onCancel: () => void;
};

function PaperUpload({ onUploadSuccess, onCancel }: Props) {
    const [file, setFile] = useState<File | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);

    function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
        setFile(e.target.files?.[0] ?? null);
        setError(null);
    }

    async function handleSubmit(e: React.SubmitEvent<HTMLFormElement>) {
        e.preventDefault();
        if (!file) {
            setError("Please select a PDF file.");
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        setLoading(true);
        setError(null);

        try {
            const response = await fetch(`${API_URL}/api/upload`, {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`Upload failed: ${response.statusText}`);
            }

            const data = await response.json();
            onUploadSuccess(
                data.session_id,
                data.title,
                data.created_at,
                data.session_ttl,
                data.paper_summary,
            );
        } catch (err) {
            setError(err instanceof Error ? err.message : "Upload failed.");
        } finally {
            setLoading(false);
        }
    }

    return (
        <div className="flex-grow-1 d-flex align-items-center justify-content-center bg-light p-4">
            <div
                className="bg-white rounded-3 shadow-sm border p-5"
                style={{ width: "100%", maxWidth: "480px" }}
            >
                <div className="text-center mb-4">
                    <div style={{ fontSize: "2.5rem" }}>📄</div>
                    <h4 className="fw-bold mt-2 mb-1">Upload a Paper</h4>
                    <p className="text-muted small mb-0">
                        PDF files only. Processing may take a moment.
                    </p>
                </div>

                <form onSubmit={handleSubmit}>
                    <label
                        className="d-block text-center rounded-3 p-4 mb-3"
                        style={{
                            border: "2px dashed #dee2e6",
                            cursor: "pointer",
                            background: file ? "#f0f9ff" : "#f8f9fa",
                            transition: "background 0.2s",
                        }}
                    >
                        <input
                            type="file"
                            accept=".pdf"
                            className="d-none"
                            onChange={handleFileChange}
                        />
                        {file ? (
                            <>
                                <div style={{ fontSize: "1.5rem" }}>✅</div>
                                <div
                                    className="fw-semibold mt-1"
                                    style={{ wordBreak: "break-all" }}
                                >
                                    {file.name}
                                </div>
                                <div className="text-muted small mt-1">
                                    Click to change file
                                </div>
                            </>
                        ) : (
                            <>
                                <div style={{ fontSize: "1.5rem" }}>📁</div>
                                <div className="fw-semibold mt-1">
                                    Click to select a PDF
                                </div>
                                <div className="text-muted small mt-1">
                                    or drag and drop here
                                </div>
                            </>
                        )}
                    </label>

                    {error && (
                        <div className="alert alert-danger py-2 small">
                            {error}
                        </div>
                    )}

                    <button
                        type="submit"
                        className="btn btn-primary w-100"
                        disabled={loading || !file}
                    >
                        {loading ? (
                            <>
                                <span
                                    className="spinner-border spinner-border-sm me-2"
                                    role="status"
                                />
                                Analyzing paper...
                            </>
                        ) : (
                            "Upload & Analyze"
                        )}
                    </button>

                    <button
                        type="button"
                        className="btn btn-link w-100 mt-2 text-muted text-decoration-none"
                        onClick={onCancel}
                    >
                        Cancel
                    </button>
                </form>
            </div>
        </div>
    );
}

export default PaperUpload;
