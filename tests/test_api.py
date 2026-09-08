"""Unit tests for FastAPI REST Endpoints (Component 6)."""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_root_endpoint():
    """Verify root endpoint returns welcome message and docs link."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "Welcome" in data["message"]
    assert data["docs_url"] == "/docs"


def test_health_check_endpoint():
    """Verify health check returns store metrics."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "documents_in_store" in data


def test_list_documents_endpoint():
    """Verify listing documents returns a list."""
    response = client.get("/api/documents")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_get_facts_endpoint():
    """Verify facts endpoint returns list of facts."""
    response = client.get("/api/facts")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_reconciliations_endpoint():
    """Verify reconciliations endpoint."""
    response = client.get("/api/reconciliations")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_case_showcase_endpoint():
    """Verify that case showcase returns the 4 required cases."""
    response = client.get("/api/cases/showcase")
    assert response.status_code == 200
    data = response.json()
    assert "corroborated_example" in data
    assert "contradiction_example" in data
    assert "context_reconciled_example" in data
    assert "failure_example" in data
    assert data["corroborated_example"]["verdict"] == "corroborated"
    assert data["context_reconciled_example"]["verdict"] == "context_reconciled"


def test_upload_invalid_file_extension():
    """Verify uploading non-pdf returns 400 error."""
    files = {"file": ("test.txt", b"some text", "text/plain")}
    response = client.post("/api/documents/upload", files=files)
    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


def test_reset_endpoint():
    """Verify reset endpoint clears non-starter files and resets to 6 starter PDFs."""
    response = client.post("/api/reset?purge_uploads=false")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["documents_indexed"] == 6
    assert data["facts_indexed"] == 217

