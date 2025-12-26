import os
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

app = FastAPI()

WORKER_SECRET = os.getenv("WORKER_SHARED_SECRET", "")


class QueryRequest(BaseModel):
    user_id: str
    question: str
    top_k: Optional[int] = 3


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/query")
def query(req: QueryRequest, x_worker_secret: Optional[str] = Header(None)):
    # Header name FastAPI expects here is: X-Worker-Secret
    if not WORKER_SECRET or x_worker_secret != WORKER_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Placeholder response to unblock Lovable integration.
    # Later you will:
    # - link question -> entities -> Neo4j traversal -> evidence
    # - return answer + results with citations
    return {
        "answer": [],
        "results": []
    }
