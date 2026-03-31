import os
import time
import tempfile
import asyncio
import uuid
import logging
from dotenv import load_dotenv

from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from research_helper.handlers.paper_handler import PaperHandler
from research_helper.agent.chat_agent import build_chat_agent

logger = logging.getLogger(__name__)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

sessions: dict[str, dict] = {}
SESSION_TTL = 3600

class ChatRequest(BaseModel):
    session_id: str
    message: str

def cleanup_sessions():
    now = time.time()
    
    expired = [
        sid for sid, s in sessions.items() 
        if now - s["created_at"] > SESSION_TTL
    ]

    for sid in expired:
        del sessions[sid]


@app.post("/api/upload")
async def upload_paper(
    file:UploadFile = File(...)
):
    cleanup_sessions()

    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400, 
            detail="Only PDF files are accepted."
        )

    contents = await file.read()

    tmp_path = None
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    if tmp_path is None:
        raise HTTPException(
            status_code=400, 
            detail="Error during saving the pdf file."
        )

    try:
        paper_handler = PaperHandler(summarize=True)
        await asyncio.to_thread(paper_handler.process_paper, tmp_path)
    except Exception as e:
        logger.exception("Internal error during reading the pdf file.\n" + str(e))
        raise HTTPException(
            status_code=400, 
            detail="Internal error during reading the pdf file."
        )
    finally:
        os.remove(tmp_path)

    agent = build_chat_agent(paper_handler)
    session_id = str(uuid.uuid4())

    sessions[session_id] = {
        "agent": agent,
        "paper_handler": paper_handler,
        "thread_id": session_id,
        "created_at": time.time()
    }

    return {
        "session_id": session_id,
        "title": paper_handler.title,
        "paper_summary": paper_handler.paper_summary.model_dump(),
    }

@app.post("/api/chat")
async def chat(body: ChatRequest):
    session = sessions.get(body.session_id)
    if session is None:
        raise HTTPException(
            status_code=404, 
            detail="Session not found or expired."
        )
    
    agent = session["agent"]
    question = body.message
    thread_id = session["thread_id"]

    response = await asyncio.to_thread(
        agent.invoke,
        {"messages": [{"role": "user", "content": question}]},
        {"configurable": {"thread_id": thread_id}}
    )

    answer = response["messages"][-1].content
    return {"question": question, "answer": answer}

@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    
    del sessions[session_id]

    return {"detail": "Session deleted."}