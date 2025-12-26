import os
import io
import re
import json
import uuid
import zipfile
import asyncio
from typing import Optional, List, Dict, Any, Tuple

import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

# Optional (only used if env vars are set)
from neo4j import GraphDatabase
from openai import OpenAI

app = FastAPI()

# =========================
# Environment variables
# =========================
WORKER_SECRET = os.getenv("WORKER_SHARED_SECRET", "")

# These are the FULL invoke URLs for your Supabase Edge Functions
WORKER_PROGRESS_URL = os.getenv("WORKER_PROGRESS_URL", "")
WORKER_COMPLETE_URL = os.getenv("WORKER_COMPLETE_URL", "")

# Neo4j (optional but recommended for Step 5)
NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USER = os.getenv("NEO4J_USER", "")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# OpenAI (optional; if missing, we’ll skip extraction and just mark job done with empty arrays)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

# =========================
# Clients (initialized on startup)
# =========================
neo4j_driver = None
openai_client: Optional[OpenAI] = None


# =========================
# Request models
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
# Helpers
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


async def post_progress(job_id: str, progress: int, status: Optional[str] = None, error: Optional[str] = None):
    """Posts progress updates back to Lovable/Supabase Edge Function worker_progress."""
    if not WORKER_PROGRESS_URL:
        return

    headers = {"X-Worker-Secret": WORKER_SECRET}
    payload: Dict[str, Any] = {"job_id": job_id, "progress": int(progress)}
    if status:
        payload["status"] = status
    if error:
        # Keep errors reasonably short so they display nicely
        payload["error"] = str(error)[:800]

    async with httpx.AsyncClient(timeout=30) as client:
        await client.post(WORKER_PROGRESS_URL, headers=headers, json=payload)


async def post_complete(payload: Dict[str, Any]):
    """Posts final extracted rows back to Lovable/Supabase Edge Function worker_complete."""
    if not WORKER_COMPLETE_URL:
        return

    headers = {"X-Worker-Secret": WORKER_SECRET}
    async with httpx.AsyncClient(timeout=90) as client:
        await client.post(WORKER_COMPLETE_URL, headers=headers, json=payload)


def decode_and_load_json(content: bytes) -> Any:
    """
    Handles either:
      - raw JSON bytes
      - a ZIP file that contains conversations.json (or any .json)
    Returns parsed JSON.
    """
    # ZIP signature
    if content[:4] == b"PK\x03\x04":
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            # Prefer conversations.json
            target = next((n for n in z.namelist() if n.endswith("conversations.json")), None)
            if target is None:
                # Fallback: first .json file
                target = next((n for n in z.namelist() if n.endswith(".json")), None)
            if target is None:
                raise ValueError("ZIP did not contain conversations.json or any .json file")
            content = z.read(target)

    # Decode JSON text safely
    try:
        text = content.decode("utf-8-sig")  # handles UTF-8 + BOM
    except UnicodeDecodeError:
        # If we see null bytes early, it might be UTF-16
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
    Returns list of conversations:
      [{conversation_id, title, messages:[{message_id, author, text}]}]

    Supports common ChatGPT export formats:
    - List of conversations
    - {"conversations":[...]}
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

        # Traverse current_node lineage for best ordering, else fallback to create_time sorting
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
    windows = []
    i = 0
    while i < n:
        win = messages[i : i + window_size]
        if win:
            windows.append(win)
        i += stride
    return windows


def is_decision_like(text: str) -> bool:
    t = text.lower()
    keywords = [
        "let's", "lets", "we should", "i will", "i want to", "decide", "decision",
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
# OpenAI extraction
# =========================
DECISION_JSON_SCHEMA = {
    "name": "decision_extraction",
    "schema": {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "status": {"type": "string", "enum": ["open", "closed", "revised"]},
                        "rationale": {"type": "string"},
                        "confidence": {"type": "number"},
                        "entities": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {"name": {"type": "string"}, "type": {"type": "string"}},
                                "required": ["name", "type"],
                            },
                        },
                        "evidence": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {"message_id": {"type": "string"}, "snippet": {"type": "string"}},
                                "required": ["message_id", "snippet"],
                            },
                        },
                    },
                    "required": ["title", "status", "rationale", "confidence", "entities", "evidence"],
                },
            }
        },
        "required": ["decisions"],
    },
}


def extract_decisions_from_window(conversation_id: str, window: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Returns extracted decisions in this window.
    If OpenAI is not configured, returns [].
    """
    if openai_client is None:
        return []

    # cheap filter: only run LLM if there's at least one "decision-like" user message
    if not any(m["author"] == "user" and is_decision_like(m["text"]) for m in window):
        return []

    system = (
        "You extract USER DECISIONS from chat messages.\n"
        "A decision is a commitment/choice (tool selection, plan, route, constraint).\n"
        "Return ONLY valid JSON matching the provided schema.\n"
        "Do NOT invent. If no explicit decision exists, return {\"decisions\": []}.\n"
        "Evidence MUST reference a message_id that appears in the provided messages.\n"
    )

    payload = {
        "conversation_id": conversation_id,
        "messages": [
            {"message_id": m["message_id"], "author": m["author"], "text": m["text"][:2000]}
            for m in window
        ],
    }

    resp = openai_client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload)},
        ],
        response_format={"type": "json_schema", "json_schema": DECISION_JSON_SCHEMA},
    )

    data = json.loads(resp.output_text)
    return data.get("decisions", [])


# =========================
# Neo4j writes
# =========================
def neo4j_write_graph(user_id: str, entities: List[Dict[str, Any]], decisions: List[Dict[str, Any]], evidence: List[Dict[str, Any]], edges: List[Dict[str, Any]]):
    if neo4j_driver is None:
        return

    # Upsert nodes
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

    # Relationships (fixed types)
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
# Endpoints
# =========================
@app.post("/start")
async def start(req: StartRequest, x_worker_secret: Optional[str] = Header(None)):
    require_secret(x_worker_secret)
    asyncio.create_task(process_job(req.job_id, req.user_id, req.signed_url))
    return {"ok": True, "started": True, "job_id": req.job_id}


async def process_job(job_id: str, user_id: str, signed_url: str):
    """
    Pipeline:
      1) Download export via signed URL (supports JSON or ZIP)
      2) Parse conversations/messages
      3) Extract decisions/entities/evidence (OpenAI)
      4) Write Neo4j graph
      5) Callback worker_complete to store rows in Supabase (via Lovable Edge Function)
    """
    try:
        await post_progress(job_id, 5, status="running")

        # 1) download bytes
        async with httpx.AsyncClient(timeout=180) as client:
            r = await client.get(signed_url)
            r.raise_for_status()
            content = r.content

        # 2) parse JSON (supports zip)
        export_json = decode_and_load_json(content)
        await post_progress(job_id, 15, status="running")

        conversations = parse_chatgpt_export(export_json)
        await post_progress(job_id, 25, status="running")

        # If OpenAI not configured, still complete job (empty results) to keep UX unblocked.
        if openai_client is None:
            await post_progress(job_id, 90, status="running")
            await post_complete({"job_id": job_id, "user_id": user_id, "entities": [], "decisions": [], "evidence": [], "edges": []})
            return

        # 3) extract
        entities_map: Dict[str, Dict[str, Any]] = {}  # key by normalized name
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

                    # entities
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

                    # evidence
                    for ev in d.get("evidence", []):
                        mid = ev.get("message_id")
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

            # progress bump each conversation
            pct = 25 + int(50 * (ci + 1) / total_convs)
            if ci % 2 == 0:
                await post_progress(job_id, min(pct, 80), status="running")

        # 4) write Neo4j
        await post_progress(job_id, 85, status="running")
        neo4j_write_graph(
            user_id=user_id,
            entities=list(entities_map.values()),
            decisions=decisions_out,
            evidence=evidence_out,
            edges=edges_out
        )

        # 5) persist via Lovable Edge Function
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
      - hop to decisions + evidence
      - return decision titles as answer bullets (safe + cited)
    """
    require_secret(x_worker_secret)

    if neo4j_driver is None:
        return {"answer": [], "results": []}

    q = (req.question or "").strip()
    if not q:
        return {"answer": [], "results": []}

    # Extract term after "about" if present
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
                citations = [c for c in r.get("citations", []) if c.get("id")]
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

        # Safe V1 answer: decision titles (already grounded by stored citations)
        answer = [r["title"] for r in results[:top_k] if r.get("citations")]
        return {"answer": answer[:3], "results": results[:top_k]}
