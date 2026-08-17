"""
FastAPI backend for the Doc-QA RAG assistant.

Endpoints:
  POST /upload        -> upload a PDF, get back a session_id
  POST /ask            -> ask a question about the uploaded doc
  DELETE /session/{id} -> clear a session's document from memory
"""
import os
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from app.rag import process_document, answer_question, clear_session

load_dotenv()

app = FastAPI(title="Doc-QA RAG Assistant")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class AskRequest(BaseModel):
    session_id: str
    question: str


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    session_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{session_id}.pdf")

    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        num_chunks = process_document(file_path, session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")

    return {
        "session_id": session_id,
        "filename": file.filename,
        "chunks_created": num_chunks,
        "message": "Document processed. You can now ask questions using this session_id.",
    }


@app.post("/ask")
async def ask(req: AskRequest):
    try:
        result = answer_question(req.question, req.session_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to answer: {e}")

    return result


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    clear_session(session_id)
    return {"message": "Session cleared."}


@app.get("/")
async def root():
    return {"status": "ok", "docs": "/docs"}
