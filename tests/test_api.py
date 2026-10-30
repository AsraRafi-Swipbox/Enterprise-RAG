import time

import jwt
import pytest
from fastapi.testclient import TestClient

from erag.api import create_app, issue_token

SECRET = "x" * 32
Q = "How many days notice is required to terminate the Zentara Systems agreement?"


@pytest.fixture()
def client(world):
    return TestClient(create_app(world["pipe"], world["directory"], SECRET))


def H(uid, secret=SECRET, ttl=3600):
    return {"Authorization": f"Bearer {issue_token(uid, secret, ttl)}"}


def test_requires_token(client):
    assert client.post("/v1/ask", json={"question": Q}).status_code == 401
    assert client.post("/v1/ask", json={"question": Q}, headers={"Authorization": "Bearer junk"}).status_code == 401


def test_rejects_wrong_secret_expired_and_unknown_subject(client):
    assert client.post("/v1/ask", json={"question": Q}, headers=H("erin", "y" * 32)).status_code == 401
    assert client.post("/v1/ask", json={"question": Q}, headers=H("erin", ttl=-10)).status_code == 401
    assert client.post("/v1/ask", json={"question": Q}, headers=H("mallory")).status_code == 401


def test_token_claims_cannot_self_assert_clearance(client):
    forged = jwt.encode({"sub": "grace", "exp": int(time.time()) + 600, "clearance": "RESTRICTED",
                         "groups": ["vendor-mgmt"], "departments": ["Legal"]}, SECRET, algorithm="HS256")
    r = client.post("/v1/ask", json={"question": Q}, headers={"Authorization": f"Bearer {forged}"})
    assert r.status_code == 200 and r.json()["abstained"] and r.json()["citations"] == []


def test_authorised_flow_and_citation_drilldown(client):
    r = client.post("/v1/ask", json={"question": Q}, headers=H("erin"))
    body = r.json()
    assert r.status_code == 200 and not body["abstained"] and body["citations"]
    cid = body["citations"][0]["chunk_id"]
    ok = client.get("/v1/chunks", params={"chunk_id": cid}, headers=H("erin"))
    assert ok.status_code == 200 and "agreement" in ok.json()["text"].lower()
    # same chunk, user without access: 404 (indistinguishable from non-existent)
    assert client.get("/v1/chunks", params={"chunk_id": cid}, headers=H("henry")).status_code == 404
    assert client.get("/v1/chunks", params={"chunk_id": "nope#0"}, headers=H("erin")).status_code == 404


def test_input_validation(client):
    assert client.post("/v1/ask", json={"question": "x"}, headers=H("erin")).status_code == 422
    assert client.post("/v1/ask", json={"question": "q" * 5000}, headers=H("erin")).status_code == 422
    assert client.post("/v1/ask", json={}, headers=H("erin")).status_code == 422


def test_weak_secret_refused(world):
    with pytest.raises(ValueError):
        create_app(world["pipe"], world["directory"], "short")
