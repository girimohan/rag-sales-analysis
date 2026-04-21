from fastapi import FastAPI
from pydantic import BaseModel

from src.rag.pipeline import run_rag

app = FastAPI()


class Question(BaseModel):
    """Request body for the /ask endpoint."""
    query: str


@app.post("/ask")
def ask(question: Question) -> dict:
    """Accept a question and return the RAG answer from the local model."""
    answer = run_rag(question.query)
    return {"answer": answer}
