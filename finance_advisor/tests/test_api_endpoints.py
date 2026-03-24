from fastapi.testclient import TestClient
from finance_advisor.backend.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
