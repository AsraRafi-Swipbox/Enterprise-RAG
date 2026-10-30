"""FastAPI service. Identity comes from a signed JWT (subject only); every permission attribute is
looked up server-side in the user directory, so a caller can never self-assert clearance or groups."""
from __future__ import annotations

import time

import jwt
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field

from .access import UserDirectory
from .models import User
from .pipeline import RAGPipeline


def issue_token(user_id: str, secret: str, ttl_s: int = 3600) -> str:
    now = int(time.time())
    return jwt.encode({"sub": user_id, "iat": now, "exp": now + ttl_s}, secret, algorithm="HS256")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class CitationOut(BaseModel):
    n: int
    doc_id: str
    chunk_id: str
    title: str
    section: str
    quote: str


class AskResponse(BaseModel):
    request_id: str
    answer: str
    abstained: bool
    confidence: float
    citations: list[CitationOut]
    latency_ms: float


def create_app(pipeline: RAGPipeline, directory: UserDirectory, jwt_secret: str) -> FastAPI:
    if not jwt_secret or len(jwt_secret) < 16:
        raise ValueError("jwt_secret must be at least 16 characters")
    app = FastAPI(title="Enterprise RAG", version="1.0.0")

    def current_user(authorization: str | None = Header(default=None)) -> User:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(401, "missing bearer token")
        try:
            claims = jwt.decode(authorization[7:], jwt_secret, algorithms=["HS256"],
                                options={"require": ["exp", "sub"]})
        except jwt.PyJWTError:
            raise HTTPException(401, "invalid or expired token")
        user = directory.get(claims["sub"])
        if user is None:
            raise HTTPException(401, "unknown subject")
        return user

    @app.get("/healthz")
    def health():
        return {"status": "ok", "chunks": len(pipeline.chunks)}

    @app.post("/v1/ask", response_model=AskResponse)
    def ask(req: AskRequest, user: User = Depends(current_user)):
        a = pipeline.ask(user, req.question)
        return AskResponse(request_id=a.request_id, answer=a.text, abstained=a.abstained,
                           confidence=round(a.confidence, 3), latency_ms=round(a.latency_ms, 2),
                           citations=[CitationOut(**c.__dict__) for c in a.citations])

    @app.get("/v1/chunks")
    def get_chunk(chunk_id: str = Query(min_length=1), user: User = Depends(current_user)):
        """Citation drill-down. Unauthorised and non-existent are indistinguishable (404)."""
        c = pipeline.resolve_chunk(user, chunk_id)
        if c is None:
            raise HTTPException(404, "not found")
        return {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "title": c.title, "section": c.section, "text": c.text}

    return app
