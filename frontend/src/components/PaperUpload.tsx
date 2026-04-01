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
        <div className="flex-grow-1 p-4 bg-light">
            <h2 className="text-center">Upload Paper</h2>
            <form className="mt-3" onSubmit={handleSubmit}>
                <div className="mb-3">
                    <input
                        type="file"
                        className="form-control"
                        accept=".pdf"
                        onChange={handleFileChange}
                    />
                </div>
                {error && <div className="alert alert-danger">{error}</div>}
                <button
                    type="submit"
                    className="btn btn-primary w-100"
                    disabled={loading}
                >
                    {loading ? "Uploading..." : "Upload"}
                </button>
                <button
                    type="button"
                    className="btn btn-outline-secondary w-100"
                    onClick={onCancel}
                >
                    Cancel
                </button>
            </form>
        </div>
    );
}

export default PaperUpload;
