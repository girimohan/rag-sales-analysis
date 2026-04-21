from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.rag.pipeline import run_rag, stream_rag

app = FastAPI()


class Question(BaseModel):
    query: str


@app.post("/ask")
def ask(question: Question) -> dict:
    """Return a complete answer. Repeated identical queries are served from cache."""
    return {"answer": run_rag(question.query)}


@app.post("/ask/stream")
def ask_stream(question: Question) -> StreamingResponse:
    """Stream answer tokens as plain text for real-time display in the UI."""
    return StreamingResponse(stream_rag(question.query), media_type="text/plain")
