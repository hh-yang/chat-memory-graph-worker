import os
import io
import re
import json
import uuid
import zipfile
import asyncio
from typing import Optional, List, Dict, Any, Tuple, Literal

import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from neo4j import GraphDatabase
from openai import OpenAI

app = FastAPI()

# =========================
# ENV VARS
# =========================
WORKER_SECRET = os.getenv("WORKER_SHARED_SECRET", "")

# Full invoke URLs for Supabase Edge Functions (set in Railway)
WORKER_PROGRESS_URL = os.getenv("WORKER_PROGRESS_URL", "")
WORKER_COMPLETE_URL = os.getenv("WORKER_COMPLETE_URL", "")

# Neo4j (set in Railway)
NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USER = os.getenv("NEO4J_USER", "")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# OpenAI (set in Railway)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# =========================
# CLIENTS (startup)
# =========================
neo4j_driver = None
openai_client: Optional[OpenAI] = None


# =========================
# REQUEST MODELS
# =========================
class StartRequest(BaseModel):
    job_id: str
    user_id: str
    signed_url: str


class QueryRequest(BaseModel):
    user_id: str
    question: str
    top_k: Optional[int] = 3


# =========================
# EXTRACTION SCHEMA (Pydantic)
# =========================
class ExtractedEntity(BaseModel):
    name: str
    type: str


class ExtractedEvidence(BaseModel):
    message_id: str
    snippet: str


class ExtractedDecision(BaseModel):
    title: str
    status: Literal["open", "closed", "revised"]
    rationale: str
    confidence: float
    entities: List[ExtractedEntity]
    evidence: List[ExtractedEvidence]


class DecisionExtraction(BaseModel):
    decisions: List[ExtractedDecision]


# =========================
# SECURITY
# =========================
def require_secret(x_worker_secret: Optional[str]):
    if not WORKER_SECRET or x_worker_secret != WORKER_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.on_event("startup")
def startup():
    global neo4j_driver, openai_client

    if NEO4J_URI and NEO4J_USER and NEO4J_PASSWORD:
        neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    if OPENAI_API_KEY:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)


@app.get("/health")
def health():
    return {"ok": True}


# =========================
# EDGE FUNCTION CALLBACKS
# =========================
async def post_progress(job_id: str, progress: int, status: Optional[str] = None, error: Optional[str] = None):
    """POST to worker_progress Edge Function (worker-only auth via X-Worker-Secret)."""
    if not WORKER_PROGRESS_URL:
        return

    headers = {"X-Worker-Secret": WORKER_SECRET}
    payload: Dict[str, Any] = {"job_id": job_id, "progress": int(progress)}
    if status:
        payload["status"] = status
    if error:
        payload["error"] = str(error)[:900]

    async with httpx.AsyncClient(timeout=30) as client:
        await client.post(WORKER_PROGRESS_URL, headers=headers, json=payload)


async def post_complete(payload: Dict[str, Any]):
    """POST to worker_complete Edge Function (worker-only auth via X-Worker-Secret)."""
    if not WORKER_COMPLETE_URL:
        return

    headers = {"X-Worker-Secret": WORKER_SECRET}
    async with httpx.AsyncClient(timeout=90) as client:
        await client.post(WORKER_COMPLETE_URL, headers=headers, json=payload)


# =========================
# CHATGPT EXPORT INGESTION
# =========================
def decode_and_load_json(content: bytes) -> Any:
    """
    Accepts either:
      - raw JSON bytes (conversations.json)
      - ZIP bytes (full ChatGPT export), extracts conversations.json or first .json
    Returns parsed JSON.
    """
    # ZIP signature PK..
    if content[:4] == b"PK\x03\x04":
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            target = next((n for n in z.namelist() if n.endswith("conversations.json")), None)
            if target is None:
                target = next((n for n in z.namelist() if n.endswith(".json")), None)
            if target is None:
                raise ValueError("ZIP did not contain conversations.json (or any .json file).")
            content = z.read(target)

    # Decode text safely
    try:
        text = content.decode("utf-8-sig")  # handles UTF-8 BOM too
    except UnicodeDecodeError:
        if b"\x00" in content[:200]:
            text = content.decode("utf-16")
        else:
            text = content.decode("utf-8", errors="replace")

    return json.loads(text)


def _safe_text(parts: Any) -> str:
    if parts is None:
        return ""
    if isinstance(parts, str):
        return parts
    if isinstance(parts, list):
        return "\n".join([str(p) for p in parts if p is not None]).strip()
    return str(parts).strip()


def parse_chatgpt_export(export_json: Any) -> List[Dict[str, Any]]:
    """
    Normalizes ChatGPT export into:
      [{conversation_id, title, messages:[{message_id, author, text}]}]
    Handles typical export format with 'mapping' + 'current_node'.
    """
    conversations = export_json
    if isinstance(export_json, dict) and "conversations" in export_json:
        conversations = export_json["conversations"]

    if not isinstance(conversations, list):
        return []

    out: List[Dict[str, Any]] = []

    for conv in conversations:
        if not isinstance(conv, dict):
            continue

        conv_id = conv.get("id") or conv.get("conversation_id") or str(uuid.uuid4())
        title = conv.get("title") or ""

        mapping = conv.get("mapping", {}) or {}
        current_node = conv.get("current_node")

        ordered_nodes: List[Dict[str, Any]] = []
        if current_node and current_node in mapping:
            node_id = current_node
            while node_id:
                node = mapping.get(node_id)
                if not node:
                    break
                ordered_nodes.append(node)
                node_id = node.get("parent")
            ordered_nodes.reverse()
        else:
            ordered_nodes = list(mapping.values())
            ordered_nodes.sort(key=lambda n: (n.get("message", {}) or {}).get("create_time", 0))

        messages: List[Dict[str, str]] = []
        for node in ordered_nodes:
            msg = node.get("message")
            if not msg:
                continue

            author_role = ((msg.get("author") or {}).get("role")) or "unknown"
            if author_role not in ("user", "assistant"):
                continue

            message_id = msg.get("id") or node.get("id") or str(uuid.uuid4())
            content = msg.get("content") or {}
            parts = content.get("parts") if isinstance(content, dict) else None
            text = _safe_text(parts).strip()
            if not text:
                continue

            messages.append({"message_id": str(message_id), "author": author_role, "text": text})

        if messages:
            out.append({"conversation_id": str(conv_id), "title": title, "messages": messages})

    return out


def build_windows(messages: List[Dict[str, str]], window_size: int = 10, stride: int = 6) -> List[List[Dict[str, str]]]:
    n = len(messages)
    if n <= window_size:
        return [messages]
    wins = []
    i = 0
    while i < n:
        wins.append(messages[i : i + window_size])
        i += stride
    return [w for w in wins if w]


def is_decision_like(text: str) -> bool:
    t = text.lower()
    keywords = [
        "we should", "i will", "i want to", "let's", "lets", "decide", "decision",
        "go with", "go down", "choose", "use ", "route", "option", "plan",
    ]
    return any(k in t for k in keywords)


def find_span(full_text: str, snippet: str) -> Tuple[int, int]:
    if not snippet:
        return 0, 0
    idx = full_text.find(snippet)
    if idx >= 0:
        return idx, idx + len(snippet)
    return 0, min(len(snippet), len(full_text))


# =========================
# OPENAI EXTRACTION (robust JSON parsing; no response_format param)
# =========================
def _extract_json_object(text: str) -> Dict[str, Any]:
    """
    Best-effort: pull the first top-level JSON object from text.
    Works even if the model accidentally adds preamble.
    """
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)

    # Find the first '{' and last '}' and try that slice.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")
    return json.loads(text[start : end + 1])


def extract_decisions_from_window(conversation_id: str, window: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Uses OpenAI to extract decisions. Returns list[dict] of ExtractedDecision.
    Avoids SDK-specific 'response_format' args; parses JSON ourselves + validates with Pydantic.
    """
    if openai_client is None:
        return []

    # Cheap filter to cut cost
    if not any(m["author"] == "user" and is_decision_like(m["text"]) for m in window):
        return []

    system = (
        "You extract USER DECISIONS from chat messages.\n"
        "A decision is an explicit commitment/choice (tool selection, plan, route, constraint).\n"
        "Return ONLY JSON with this exact shape:\n"
        "{\n"
        '  "decisions": [\n'
        "    {\n"
        '      "title": string,\n'
        '      "status": "open" | "closed" | "revised",\n'
        '      "rationale": string,\n'
        '      "confidence": number (0..1),\n'
        '      "entities": [{"name": string, "type": string}],\n'
        '      "evidence": [{"message_id": string, "snippet": string}]\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Rules:\n"
        "- Do NOT invent. If no explicit decision exists, return {\"decisions\": []}.\n"
        "- Evidence.message_id MUST be one of the provided message_ids.\n"
    )

    payload = {
        "conversation_id": conversation_id,
        "messages": [
            {"message_id": m["message_id"], "author": m["author"], "text": m["text"][:2000]}
            for m in window
        ],
    }

    # Try twice: if first parse fails, ask again with stricter reminder
    for attempt in range(2):
        resp = openai_client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload)},
            ],
            temperature=0.2,
        )

        # Most OpenAI Python SDK versions expose output_text for Responses
        out_text = getattr(resp, "output_text", None)
        if not out_text:
            # fallback: stringify entire resp (best-effort)
            out_text = str(resp)

        try:
            data = _extract_json_object(out_text)
            validated = DecisionExtraction.model_validate(data)
            return [d.model_dump() for d in validated.decisions]
        except Exception:
            if attempt == 0:
                # tighten system message and retry once
                system = system + "\nIMPORTANT: Output must be valid JSON only. No markdown, no commentary."
                continue
            return []

    return []


# =========================
# NEO4J WRITES (ABOUT + SUPPORTED_BY)
# =========================
def neo4j_write_graph(user_id: str, entities: List[Dict[str, Any]], decisions: List[Dict[str, Any]], evidence: List[Dict[str, Any]], edges: List[Dict[str, Any]]):
    if neo4j_driver is None:
        return

    cypher_nodes = """
    UNWIND $entities AS e
    MERGE (en:Entity {id: e.id})
    SET en.user_id = e.user_id, en.name = e.name, en.entity_type = e.entity_type;

    UNWIND $decisions AS d
    MERGE (de:Decision {id: d.id})
    SET de.user_id = d.user_id, de.title = d.title, de.status = d.status, de.confidence = d.confidence;

    UNWIND $evidence AS v
    MERGE (ev:Evidence {id: v.id})
    SET ev.user_id = v.user_id,
        ev.conversation_id = v.conversation_id,
        ev.message_id = v.message_id,
        ev.start_char = v.start_char,
        ev.end_char = v.end_char,
        ev.snippet = v.snippet;
    """

    about_edges = [e for e in edges if e.get("rel_type") == "ABOUT"]
    supported_edges = [e for e in edges if e.get("rel_type") == "SUPPORTED_BY"]

    cypher_about = """
    UNWIND $rels AS r
    MATCH (a:Entity {id: r.from_id})
    MATCH (b:Decision {id: r.to_id})
    MERGE (a)-[:ABOUT]->(b);
    """

    cypher_supported = """
    UNWIND $rels AS r
    MATCH (a:Decision {id: r.from_id})
    MATCH (b:Evidence {id: r.to_id})
    MERGE (a)-[:SUPPORTED_BY]->(b);
    """

    with neo4j_driver.session() as session:
        session.run(cypher_nodes, entities=entities, decisions=decisions, evidence=evidence)
        if about_edges:
            session.run(cypher_about, rels=about_edges)
        if supported_edges:
            session.run(cypher_supported, rels=supported_edges)


# =========================
# ENDPOINTS
# =========================
@app.post("/start")
async def start(req: StartRequest, x_worker_secret: Optional[str] = Header(None)):
    require_secret(x_worker_secret)
    asyncio.create_task(process_job(req.job_id, req.user_id, req.signed_url))
    return {"ok": True, "started": True, "job_id": req.job_id}


async def process_job(job_id: str, user_id: str, signed_url: str):
    """
    Pipeline:
      1) Download export via signed URL (JSON or ZIP)
      2) Parse into canonical messages
      3) Extract decisions/entities/evidence
      4) Write Neo4j graph (optional)
      5) Call worker_complete to upsert into Supabase (via Edge Function)
    """
    try:
        await post_progress(job_id, 5, status="running")

        # Download bytes
        async with httpx.AsyncClient(timeout=180) as client:
            r = await client.get(signed_url)
            r.raise_for_status()
            content = r.content

        export_json = decode_and_load_json(content)
        await post_progress(job_id, 15, status="running")

        conversations = parse_chatgpt_export(export_json)
        await post_progress(job_id, 25, status="running")

        # If OpenAI not configured, complete with empty arrays (keeps product usable)
        if openai_client is None:
            await post_progress(job_id, 90, status="running")
            await post_complete({"job_id": job_id, "user_id": user_id, "entities": [], "decisions": [], "evidence": [], "edges": []})
            return

        entities_map: Dict[str, Dict[str, Any]] = {}  # key: normalized name
        decisions_out: List[Dict[str, Any]] = []
        evidence_out: List[Dict[str, Any]] = []
        edges_out: List[Dict[str, Any]] = []

        total_convs = max(1, len(conversations))

        for ci, conv in enumerate(conversations):
            conv_id = conv["conversation_id"]
            msgs = conv["messages"]
            windows = build_windows(msgs, window_size=10, stride=6)

            for w in windows:
                extracted = extract_decisions_from_window(conv_id, w)
                if not extracted:
                    continue

                msg_by_id = {m["message_id"]: m["text"] for m in w}

                for d in extracted:
                    decision_id = str(uuid.uuid4())
                    decisions_out.append({
                        "id": decision_id,
                        "user_id": user_id,
                        "title": (d.get("title") or "")[:240],
                        "status": d.get("status") or "open",
                        "rationale": (d.get("rationale") or "")[:2000],
                        "confidence": float(d.get("confidence") or 0.5),
                        "decided_at": None,
                    })

                    # Entities -> Decision edges
                    for ent in d.get("entities", []):
                        name = (ent.get("name") or "").strip()
                        if not name:
                            continue
                        key = re.sub(r"\s+", " ", name.lower())
                        if key not in entities_map:
                            entities_map[key] = {
                                "id": str(uuid.uuid4()),
                                "user_id": user_id,
                                "name": name[:200],
                                "canonical_name": None,
                                "entity_type": (ent.get("type") or "other")[:50],
                            }

                        edges_out.append({
                            "id": str(uuid.uuid4()),
                            "user_id": user_id,
                            "from_type": "entity",
                            "from_id": entities_map[key]["id"],
                            "rel_type": "ABOUT",
                            "to_type": "decision",
                            "to_id": decision_id,
                        })

                    # Evidence -> Decision edges
                    for ev in d.get("evidence", []):
                        mid = (ev.get("message_id") or "").strip()
                        snippet = (ev.get("snippet") or "").strip()
                        if not mid or not snippet or mid not in msg_by_id:
                            continue

                        full_text = msg_by_id[mid]
                        start_char, end_char = find_span(full_text, snippet)

                        ev_id = str(uuid.uuid4())
                        evidence_out.append({
                            "id": ev_id,
                            "user_id": user_id,
                            "decision_id": decision_id,
                            "conversation_id": conv_id,
                            "message_id": mid,
                            "start_char": int(start_char),
                            "end_char": int(end_char),
                            "snippet": snippet[:500],
                        })

                        edges_out.append({
                            "id": str(uuid.uuid4()),
                            "user_id": user_id,
                            "from_type": "decision",
                            "from_id": decision_id,
                            "rel_type": "SUPPORTED_BY",
                            "to_type": "evidence",
                            "to_id": ev_id,
                        })

            # progress bump
            if ci % 2 == 0:
                pct = 25 + int(50 * (ci + 1) / total_convs)
                await post_progress(job_id, min(pct, 80), status="running")

        # Write Neo4j (optional)
        await post_progress(job_id, 85, status="running")
        neo4j_write_graph(user_id, list(entities_map.values()), decisions_out, evidence_out, edges_out)

        # Persist results into Supabase via worker_complete
        await post_progress(job_id, 95, status="running")
        await post_complete({
            "job_id": job_id,
            "user_id": user_id,
            "entities": list(entities_map.values()),
            "decisions": decisions_out,
            "evidence": evidence_out,
            "edges": edges_out,
        })

    except Exception as e:
        await post_progress(job_id, 0, status="failed", error=str(e))


@app.post("/query")
def query(req: QueryRequest, x_worker_secret: Optional[str] = Header(None)):
    """
    Minimal GraphRAG:
      - match entity name by substring
      - hop Entity -[:ABOUT]-> Decision -[:SUPPORTED_BY]-> Evidence
      - return decision titles as answer bullets + citations
    """
    require_secret(x_worker_secret)

    if neo4j_driver is None:
        return {"answer": [], "results": []}

    q = (req.question or "").strip()
    if not q:
        return {"answer": [], "results": []}

    m = re.search(r"\babout\b(.+)$", q, flags=re.I)
    entity_term = (m.group(1).strip() if m else q).strip()[:80]
    top_k = int(req.top_k or 3)

    with neo4j_driver.session() as session:
        ents = session.run(
            """
            MATCH (e:Entity)
            WHERE e.user_id = $user_id AND toLower(e.name) CONTAINS toLower($q)
            RETURN e.id AS id, e.name AS name
            LIMIT 5
            """,
            user_id=req.user_id,
            q=entity_term
        ).data()

        if not ents:
            return {"answer": [], "results": []}

        results: List[Dict[str, Any]] = []

        for ent in ents:
            rows = session.run(
                """
                MATCH (e:Entity {id: $eid})-[:ABOUT]->(d:Decision)
                WHERE d.user_id = $user_id
                OPTIONAL MATCH (d)-[:SUPPORTED_BY]->(v:Evidence)
                RETURN d.id AS decision_id,
                       d.title AS title,
                       d.status AS status,
                       d.confidence AS confidence,
                       collect({
                         id: v.id,
                         snippet: v.snippet,
                         conversation_id: v.conversation_id,
                         message_id: v.message_id,
                         start_char: v.start_char,
                         end_char: v.end_char
                       }) AS citations
                ORDER BY d.confidence DESC
                LIMIT $k
                """,
                eid=ent["id"],
                user_id=req.user_id,
                k=top_k
            ).data()

            for r in rows:
                citations = [c for c in (r.get("citations") or []) if c.get("id")]
                path = [
                    {"type": "Entity", "id": ent["id"], "name": ent["name"]},
                    {"type": "Decision", "id": r["decision_id"], "title": r["title"]},
                ]
                if citations:
                    c0 = citations[0]
                    path.append({
                        "type": "Evidence",
                        "id": c0["id"],
                        "snippet": c0.get("snippet"),
                        "conversation_id": c0.get("conversation_id"),
                        "message_id": c0.get("message_id"),
                        "start_char": c0.get("start_char"),
                        "end_char": c0.get("end_char"),
                    })

                results.append({
                    "decision_id": r["decision_id"],
                    "title": r["title"],
                    "status": r.get("status") or "open",
                    "confidence": r.get("confidence") or 0.0,
                    "path": path,
                    "citations": citations,
                })

        # Safe V1 answer: decision titles that have citations
        answer = [r["title"] for r in results[:top_k] if r.get("citations")]
        return {"answer": answer[:3], "results": results[:top_k]}
