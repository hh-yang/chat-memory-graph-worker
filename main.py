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

# Full invoke URLs for Lovable/Supabase Edge Functions (set in Railway)
WORKER_PROGRESS_URL = os.getenv("WORKER_PROGRESS_URL", "")
WORKER_COMPLETE_URL = os.getenv("WORKER_COMPLETE_URL", "")

# Neo4j (set in Railway)
NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USER = os.getenv("NEO4J_USER", "")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# OpenAI (set in Railway)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Speed / caps
FAST_MAX_CONVS = int(os.getenv("FAST_MAX_CONVS", "200"))         # cap conversations processed per upload
FAST_CONTEXT_PRE = int(os.getenv("FAST_CONTEXT_PRE", "4"))       # msgs before decision-like user msg
FAST_CONTEXT_POST = int(os.getenv("FAST_CONTEXT_POST", "2"))     # msgs after decision-like user msg
FAST_MAX_SEGMENTS = int(os.getenv("FAST_MAX_SEGMENTS", "25"))    # cap number of segments per conversation
MAX_CHARS_PER_MSG = int(os.getenv("MAX_CHARS_PER_MSG", "800"))   # cap chars per message sent to OpenAI
MAX_DECISIONS_PER_CONV = int(os.getenv("MAX_DECISIONS_PER_CONV", "20"))

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


# =========================
# SPEED HELPERS (segments)
# =========================
def is_decision_like(text: str) -> bool:
    t = (text or "").lower()
    keywords = [
        "we should", "i will", "i want to", "let's", "lets", "decide", "decision",
        "go with", "go down", "choose", "use ", "route", "option", "plan",
        "next step", "let’s", "let us", "i’m going to", "i am going to"
    ]
    return any(k in t for k in keywords)


def build_candidate_segments(messages: List[Dict[str, str]]) -> List[List[Dict[str, str]]]:
    """
    Build small context segments around decision-like USER messages.
    Merge overlaps so we don't spam the model.
    """
    idxs = [i for i, m in enumerate(messages) if m["author"] == "user" and is_decision_like(m["text"])]
    if not idxs:
        return []

    ranges: List[Tuple[int, int]] = []
    n = len(messages)
    for i in idxs:
        start = max(0, i - FAST_CONTEXT_PRE)
        end = min(n, i + FAST_CONTEXT_POST + 1)  # slice end
        ranges.append((start, end))

    ranges.sort()
    merged: List[Tuple[int, int]] = []
    cur_s, cur_e = ranges[0]
    for s, e in ranges[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))

    merged = merged[:FAST_MAX_SEGMENTS]

    segments: List[List[Dict[str, str]]] = []
    for s, e in merged:
        seg = []
        for m in messages[s:e]:
            seg.append({
                "message_id": m["message_id"],
                "author": m["author"],
                "text": (m["text"] or "")[:MAX_CHARS_PER_MSG],
            })
        segments.append(seg)

    return segments


def find_span(full_text: str, snippet: str) -> Tuple[int, int]:
    if not snippet:
        return 0, 0
    idx = full_text.find(snippet)
    if idx >= 0:
        return idx, idx + len(snippet)
    return 0, min(len(snippet), len(full_text))


def canonicalize_name(name: str) -> str:
    # lower + collapse whitespace; always non-empty if name non-empty
    key = re.sub(r"\s+", " ", (name or "").strip().lower())
    return key[:200] if key else "unknown"


# =========================
# OPENAI EXTRACTION (1 call per conversation)
# =========================
def _extract_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")
    return json.loads(text[start:end + 1])


def extract_decisions_from_conversation(conversation_id: str, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    1 OpenAI call per conversation:
      - build candidate segments around decision-like user messages
      - extract decisions/entities/evidence with strict JSON format
    """
    if openai_client is None:
        return []

    segments = build_candidate_segments(messages)
    if not segments:
        return []

    system = (
        "You extract USER DECISIONS from chat messages.\n"
        "A decision is an explicit commitment/choice (tool selection, plan, route, constraint).\n"
        f"Return at most {MAX_DECISIONS_PER_CONV} decisions.\n"
        "Return ONLY JSON with this exact shape:\n"
        "{\n"
        '  \"decisions\": [\n'
        "    {\n"
        '      \"title\": string,\n'
        '      \"status\": \"open\" | \"closed\" | \"revised\",\n'
        '      \"rationale\": string,\n'
        '      \"confidence\": number (0..1),\n'
        '      \"entities\": [{\"name\": string, \"type\": string}],\n'
        '      \"evidence\": [{\"message_id\": string, \"snippet\": string}]\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Rules:\n"
        "- Do NOT invent. If no explicit decision exists, return {\"decisions\": []}.\n"
        "- Evidence.message_id MUST be one of the provided message_ids.\n"
        "- Evidence.snippet MUST be a verbatim substring from that message.\n"
        "- Keep titles short and specific.\n"
    )

    payload = {"conversation_id": conversation_id, "segments": segments}

    for attempt in range(2):
        resp = openai_client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload)},
            ],
            temperature=0.2,
        )

        out_text = getattr(resp, "output_text", None) or ""
        if not out_text:
            out_text = str(resp)

        try:
            data = _extract_json_object(out_text)
            validated = DecisionExtraction.model_validate(data)
            decisions = validated.decisions[:MAX_DECISIONS_PER_CONV]
            return [d.model_dump() for d in decisions]
        except Exception:
            if attempt == 0:
                system = system + "\nIMPORTANT: Output must be valid JSON only. No markdown, no commentary."
                continue
            return []

    return []


# =========================
# NEO4J WRITES (ONE STATEMENT PER RUN)
# =========================
def neo4j_write_graph(
    user_id: str,
    entities: List[Dict[str, Any]],
    decisions: List[Dict[str, Any]],
    evidence: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
):
    if neo4j_driver is None:
        return

    cypher_entities = """
    UNWIND $entities AS e
    MERGE (en:Entity {id: e.id})
    SET en.user_id = e.user_id,
        en.name = e.name,
        en.entity_type = e.entity_type
    """

    cypher_decisions = """
    UNWIND $decisions AS d
    MERGE (de:Decision {id: d.id})
    SET de.user_id = d.user_id,
        de.title = d.title,
        de.status = d.status,
        de.confidence = d.confidence
    """

    cypher_evidence = """
    UNWIND $evidence AS v
    MERGE (ev:Evidence {id: v.id})
    SET ev.user_id = v.user_id,
        ev.conversation_id = v.conversation_id,
        ev.message_id = v.message_id,
        ev.start_char = v.start_char,
        ev.end_char = v.end_char,
        ev.snippet = v.snippet
    """

    cypher_about = """
    UNWIND $rels AS r
    MATCH (a:Entity {id: r.from_id})
    MATCH (b:Decision {id: r.to_id})
    MERGE (a)-[:ABOUT]->(b)
    """

    cypher_supported = """
    UNWIND $rels AS r
    MATCH (a:Decision {id: r.from_id})
    MATCH (b:Evidence {id: r.to_id})
    MERGE (a)-[:SUPPORTED_BY]->(b)
    """

    about_edges = [e for e in edges if e.get("rel_type") == "ABOUT"]
    supported_edges = [e for e in edges if e.get("rel_type") == "SUPPORTED_BY"]

    with neo4j_driver.session() as session:
        if entities:
            session.run(cypher_entities, entities=entities)
        if decisions:
            session.run(cypher_decisions, decisions=decisions)
        if evidence:
            session.run(cypher_evidence, evidence=evidence)
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
    Fast pipeline:
      1) Download export via signed URL (JSON or ZIP)
      2) Parse messages
      3) 1 OpenAI call per conversation (only decision-like segments)
      4) Write Neo4j graph
      5) Call worker_complete to upsert into Supabase (via Edge Function)
    """
    try:
        await post_progress(job_id, 5, status="running")

        async with httpx.AsyncClient(timeout=180) as client:
            r = await client.get(signed_url)
            r.raise_for_status()
            content = r.content

        export_json = decode_and_load_json(content)
        await post_progress(job_id, 15, status="running")

        conversations = parse_chatgpt_export(export_json)

        # Fast cap
        if FAST_MAX_CONVS > 0:
            conversations = conversations[:FAST_MAX_CONVS]

        await post_progress(job_id, 25, status="running")

        # If OpenAI not configured, complete with empty arrays
        if openai_client is None:
            await post_progress(job_id, 90, status="running")
            await post_complete({"job_id": job_id, "user_id": user_id, "entities": [], "decisions": [], "evidence": [], "edges": []})
            return

        entities_map: Dict[str, Dict[str, Any]] = {}  # key: canonical_name
        decisions_out: List[Dict[str, Any]] = []
        evidence_out: List[Dict[str, Any]] = []
        edges_out: List[Dict[str, Any]] = []

        total_convs = max(1, len(conversations))

        for ci, conv in enumerate(conversations):
            conv_id = conv["conversation_id"]
            msgs = conv["messages"]

            extracted = extract_decisions_from_conversation(conv_id, msgs)
            if not extracted:
                # progress bump still
                if ci % 10 == 0:
                    pct = 25 + int(55 * (ci + 1) / total_convs)
                    await post_progress(job_id, min(pct, 85), status="running")
                continue

            # Evidence can refer to any message in the conversation
            msg_by_id = {m["message_id"]: m["text"] for m in msgs}

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
                    raw_name = (ent.get("name") or "").strip()
                    if not raw_name:
                        continue

                    canonical = canonicalize_name(raw_name)  # ✅ never null
                    if canonical not in entities_map:
                        entities_map[canonical] = {
                            "id": str(uuid.uuid4()),
                            "user_id": user_id,
                            "name": raw_name[:200],
                            "canonical_name": canonical,  # ✅ NOT NULL for Supabase
                            "entity_type": (ent.get("type") or "other")[:50],
                        }

                    edges_out.append({
                        "id": str(uuid.uuid4()),
                        "user_id": user_id,
                        "from_type": "entity",
                        "from_id": entities_map[canonical]["id"],
                        "rel_type": "ABOUT",
                        "to_type": "decision",
                        "to_id": decision_id,
                    })

                # Evidence -> Decision edges
                for ev in d.get("evidence", []):
                    mid = (ev.get("message_id") or "").strip()
                    snippet = (ev.get("snippet") or "").strip()
