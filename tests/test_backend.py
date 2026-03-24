"""
Basic unit tests for backend logic.
Run with: pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend import retrieve_relevant_chunks


def test_chunking_overlap():
    """Overlapping chunking should produce correct number of segments."""
    text = "A" * 1000
    chunk_size, overlap = 500, 50
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    assert len(chunks) == 3
    assert len(chunks[0]) == 500


def test_empty_kb_returns_nothing():
    """With empty knowledge base, retrieval returns empty list."""
    result = retrieve_relevant_chunks("anything at all")
    assert result == []


def test_health_endpoint():
    """Health endpoint should respond with 200 and status field."""
    from fastapi.testclient import TestClient
    from backend import app
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert "status" in r.json()


def test_ask_empty_question():
    """Empty question should return 400."""
    from fastapi.testclient import TestClient
    from backend import app
    client = TestClient(app)
    r = client.post("/ask", json={"question": ""})
    assert r.status_code == 400
