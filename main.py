import os
import asyncio
from typing import Optional, Dict, Any, List

import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

app = FastAPI()

WORKER_SECRET = os.getenv("WORKER_SHARED_SECRET", "")

# These will be the full URLs Lovable gives you after creating Edge Functions
WORKER_PROGRESS_URL = os.getenv("WORKER_PROGRESS_URL", "")
WORKER_COMPLETE_URL = os.getenv("WORKER_COMPLETE_URL", "")


class StartRequest(BaseModel):
    job_id: str
    user_id: str
    signed_url: str


class QueryRequest(BaseModel):
    user_id: str
    question: str
    top_k: Optional[int] = 3


def require_secret(x_worker_secret: Optional[str]):
    if not WORKER_SECRET or x_worker_secret != WORKER_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/start")
async def start(req: StartRequest, x_worker_secret: Optional[str] = Header(None)):
    require_secret(x_worker_secret)

    # Kick off background task and return immediately
    asyncio.create_task(process_job(req.job_id, req.user_id, req.signed_url))
    return {"ok": True, "started": True, "job_id": req.job_id}


async def process_job(job_id: str, user_id: str, signed_url: str):
    """
    Placeholder pipeline:
    - Download export via signed URL
    - (Later) parse + extract + write Neo4j + produce entities/decisions/evidence/edges
    - For now, just mark done with empty arrays
    """
    headers = {"X-Worker-Secret": WORKER_SECRET}

    async with httpx.AsyncClient(timeout=60) as client:
        # 1) progress 10
        if WORKER_PROGRESS_URL:
            await client.post(WORKER_PROGRESS_URL, headers=headers, json={"job_id": job_id, "progress": 10, "status": "running"})

        # 2) download file (proof signed URL works)
        try:
            r = await client.get(signed_url)
            r.raise_for_status()
            _raw = r.text  # not used yet
        except Exception as e:
            if WORKER_PROGRESS_URL:
                await client.post(WORKER_PROGRESS_URL, headers=headers, json={"job_id": job_id, "progress": 0, "status": "failed", "error": str(e)})
            return

        # 3) progress 70 (placeholder)
        if WORKER_PROGRESS_URL:
            await client.post(WORKER_PROGRESS_URL, headers=headers, json={"job_id": job_id, "progress": 70, "status": "running"})

        # 4) complete job (empty payload for now)
        payload = {
            "job_id": job_id,
            "user_id": user_id,
            "entities": [],
            "decisions": [],
            "evidence": [],
            "edges": []
        }

        if WORKER_COMPLETE_URL:
            await client.post(WORKER_COMPLETE_URL, headers=headers, json=payload)


@app.post("/query")
def query(req: QueryRequest, x_worker_secret: Optional[str] = Header(None)):
    require_secret(x_worker_secret)
    return {"answer": [], "results": []}
