import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_ask_endpoint():
    response = client.post("/ask", json={"query": "What is the total sales?"})
    assert response.status_code == 200
    assert "answer" in response.json()

def test_ask_stream_endpoint():
    response = client.post("/ask/stream", json={"query": "List top 5 customers."})
    assert response.status_code == 200
    assert response.text != ""
