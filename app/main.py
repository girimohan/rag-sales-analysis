from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.rag.pipeline import run_rag, stream_rag

app = FastAPI()


class Question(BaseModel):
    query: str
    model: str = "llama3"


@app.post("/ask")
def ask(question: Question) -> dict:
    """Return a complete answer. Repeated identical queries are served from cache."""
    return {"answer": run_rag(question.query, model=question.model)}


@app.post("/ask/stream")
def ask_stream(question: Question) -> StreamingResponse:
    """Stream answer tokens as plain text for real-time display in the UI."""
    return StreamingResponse(stream_rag(question.query, model=question.model), media_type="text/plain")
