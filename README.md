# Research Helper

> Upload a research paper, get an instant structured summary, and chat with it using natural language.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?style=flat&logo=react&logoColor=black)
![TypeScript](https://img.shields.io/badge/TypeScript-5.9-3178C6?style=flat&logo=typescript&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-agentic_RAG-FF6B35?style=flat)

---

## Overview

Research Helper is a full-stack AI application that lets you have a conversation with any research paper PDF. Upload a paper, and the system automatically:

- Parses and sections the PDF using **PyMuPDF**
- Generates per-section summaries and a structured overall summary using an **LLM**
- Builds a **FAISS vector index** over chunked paper content using sentence embeddings
- Exposes a **RAG-powered chat agent** (via LangGraph) that retrieves relevant context before answering questions

---

## Features

- **Automatic structured summary** — extracts research problem, key contributions, method overview, experimental findings, and limitations
- **Agentic RAG chat** — a LangGraph agent with a context-retrieval tool answers questions grounded in the paper's content
- **Session management** — sessions persist in the browser via `localStorage` and expire after 1 hour (matching the backend TTL)
- **Clean, responsive UI** — built with React + Bootstrap 5

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                     Frontend (React)                │
│  Sidebar │ Starter │ PaperUpload │ Chat             │
└──────────────────────┬──────────────────────────────┘
                       │ REST API
┌──────────────────────▼──────────────────────────────┐
│                   Backend (FastAPI)                 │
│                                                     │
│  POST /api/upload                                   │
│    └─ PaperHandler                                  │
│         ├─ PDF parsing        (PyMuPDF)             │
│         ├─ Section summaries  (LLM)                 │
│         ├─ Paper summary      (LLM)                 │
│         ├─ Chunking           (LangChain splitter)  │
│         └─ Vector index       (FAISS + MiniLM)      │
│                                                     │
│  POST /api/chat                                     │
│    └─ LangGraph Agent                               │
│         └─ retrieve_context tool → FAISS search     │
└─────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer           | Technology                              |
| --------------- | --------------------------------------- |
| Frontend        | React 19, TypeScript, Vite, Bootstrap 5 |
| Backend         | FastAPI, Python 3.11+                   |
| LLM             | Llama 4 Scout via Groq API              |
| Agent framework | LangGraph + LangChain                   |
| Embeddings      | `all-MiniLM-L6-v2` (HuggingFace)        |
| Vector store    | FAISS (in-memory)                       |
| PDF parsing     | PyMuPDF (`pymupdf4llm`)                 |

---

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- A [Groq API key](https://console.groq.com/)

### Backend

```bash
cd backend
pip install -r requirements.txt
```

Create a `.env` file in `backend/`:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Start the server:

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.

### Frontend

```bash
cd frontend
npm install
```

Create a `.env` file in `frontend/`:

```env
VITE_BACKEND_API_ENDPOINT=http://localhost:8000
```

Start the dev server:

```bash
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## Usage

1. Click **New Session** in the sidebar
2. Upload a research paper PDF
3. Wait for the paper to be processed — a structured summary will appear automatically
4. Ask any questions about the paper in the chat input

Sessions expire after **1 hour**. Expired sessions remain readable in the sidebar but new messages cannot be sent.

---

## Project Structure

```
research-helper/
├── backend/
│   ├── main.py                          # FastAPI app, session management
│   └── research_helper/
│       ├── utils.py                     # LLM prompt templates
│       ├── agent/
│       │   ├── chat_agent.py            # LangGraph agent builder
│       │   └── tools/
│       │       └── retrieval.py         # FAISS retrieval tool
│       └── handlers/
│           ├── paper_handler.py         # PDF parsing, summarization orchestration
│           └── rag/
│               ├── chunk_handler.py     # LangChain text splitter wrapper
│               └── vector_store_handler.py  # FAISS index management
└── frontend/
    └── src/
        ├── App.tsx                      # Root component, session state + localStorage
        ├── types.ts                     # Shared TypeScript types
        └── components/
            ├── Sidebar.tsx              # Session list navigation
            ├── Starter.tsx              # Home / landing view
            ├── PaperUpload.tsx          # PDF upload form
            └── Chat.tsx                 # Chat interface with expiry handling
```
