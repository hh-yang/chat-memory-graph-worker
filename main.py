import os
import json
import re
import asyncio
import uuid
from typing import Optional, List, Dict, Any, Tuple

import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from neo4j import GraphDatabase
from openai import OpenAI

app = FastAPI()

# === Secrets / Config ===
WORKER_SECRET = os.getenv("WORKER_SHARED_SECRET", "")
WORKER_PROGRESS_URL = os.getenv("WORKER_PROGRESS_URL", "")
WORKER_COMPLETE_URL = os.getenv("WORKER_COMPLETE_URL", "")

NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USER = os.getenv("NEO4J_USER", "")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
neo4j_driver = None


# === Request Models ===
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


@app.on_event("startup")
def startup():
    global neo4j_driver
    if NEO4J_URI and NEO4J_USER and NEO4J_PASSWORD:
        neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


@app.get("/health")
def health():
    return {"ok": True}


# ============================================================
# 1) EXPORT PARSING (ChatGPT export -> canonical messages)
# ============================================================

def _safe_text(parts: Any) -> str:
    """
    ChatGPT export sometimes stores content as:
    - {"content":{"parts":[...]}}
    - {"content":{"text":"..."}} (rare)
    We'll normalize to a single string.
    """
    if parts is None:
        return ""
    if isinstance(parts, str):
        return parts
    if isinstance(parts, list):
        return "\n".join([str(p) for p in parts if p is not None]).strip()
    return str(parts).strip()


def parse_chatgpt_export(export_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Returns a list of conversations, each with:
      {conversation_id, title, messages:[{message_id, author, text}]}
    Handles typical ChatGPT export shape: list of conversations with 'mapping'.
    """
    conversations = export_json
    if isinstance(export_json, dict) and "conversations" in export_json:
        conversations = export_json["conversations"]

    out = []

    if not isinstance(conversations, list):
        return out

    for conv in conversations:
        conv_id = conv.get("id") or conv.get("conversation_id") or str(uuid.uuid4())
        title = conv.get("title") or ""

        mapping = conv.get("mapping", {})
        current_node = conv.get("current_node")

        # If we have a current_node, traverse parents to root to get order
        ordered_nodes = []
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
            # fallback: best-effort order by create_time if available
            ordered_nodes = list(mapping.values())
            ordered_nodes.sort(key=lambda n: (n.get("message", {}) or {}).get("create_time", 0))

        messages = []
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
            text = _safe_text(parts)
            text = (text or "").strip()
            if not text:
                continue
            messages.append({"message_id": str(message_id), "author": author_role, "text": text})

        if messages:
            out.append({"conversation_id": str(conv_id), "title": title, "messages": messages})

    return out


# ============================================================
# 2) LLM EXTRACTION (decisions + entities + evidence)
# ============================================================

DECISION_SCHEMA = {
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
                "properties": {
                  "name": {"type": "string"},
                  "type": {"type": "string"}
                },
                "required": ["name", "type"]
              }
            },
            "evidence": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "message_id": {"type": "string"},
                  "snippet": {"type": "string"}
                },
                "required": ["message_id", "snippet"]
              }
            }
          },
          "required": ["title", "status", "rationale", "confidence", "entities", "evidence"]
        }
      }
    },
    "required": ["decisions"]
  }
}


def is_decision_like(text: str) -> bool:
    t = text.lower()
    keywords = [
        "let's", "lets", "we should", "i will", "i want to", "decide", "decision",
        "go with", "go down", "choose", "use ", "route", "option", "plan"
    ]
    return any(k in t for k in keywords)


def build_windows(messages: List[Dict[str, str]], window_size: int = 10, stride: int = 6) -> List[List[Dict[str, str]]]:
    windows = []
    n = len(messages)
    if n <= window_size:
        return [messages]
    i = 0
    while i < n:
        win = messages[i:i+window_size]
        if win:
            windows.append(win)
        i += stride
    return windows


def extract_decisions_from_window(conversation_id: str, window: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    if client is None:
        return []

    # Keep only windows with at least one decision-like user message (cheap filter)
    if not any(m["author"] == "user" and is_decision_like(m["text"]) for m in window):
        return []

    system = (
        "You extract USER DECISIONS from chat messages.\n"
        "A decision is a commitment/choice (tool selection, plan, route, constraint).\n"
        "Return ONLY valid JSON matching the provided schema.\n"
        "Do NOT invent. If no explicit decision exists, return {\"decisions\": []}.\n"
        "Evidence MUST reference a message_id that appears in the provided messages.\n"
    )

    messages_text = [
        {"message_id": m["message_id"], "author": m["author"], "text": m["text"][:2000]}
        for m in window
    ]

    user = {
        "conversation_id": conversation_id,
        "messages": messages_text
    }

    # Use Responses API with json_schema for strictness
    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user)}
        ],
        response_format={"type": "json_schema", "json_schema": DECISION_SCHEMA}
    )

    # Parse response JSON
    raw = resp.output_text
    data = json.loads(raw)
    return data.get("decisions", [])


def find_span(full_text: str, snippet: str) -> Tuple[int, int]:
    # Best-effort: find snippet in full_text
    if not snippet:
        return 0, 0
    idx = full_text.find(snippet)
    if idx >= 0:
        return idx, idx + len(snippet)
    # fallback: 0..len(snippet)
    return 0, min(len(snippet), len(full_text))


# ============================================================
# 3) NEO4J WRITES
# ============================================================

def neo4j_upsert_graph(user_id: str, entities: List[Dict[str, Any]], decisions: List[Dict[str, Any]], evidence: List[Dict[str, Any]], edges: List[Dict[str, Any]]):
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
    SET ev.user_id = v.user_id, ev.conversation_id = v.conversation_id, ev.message_id = v.message_id,
        ev.start_char = v.start_char, ev.end_char = v.end_char, ev.snippet = v.snippet;
    """

    cypher_edges = """
    UNWIND $edges AS r
    MATCH (a {id: r.from_id})
    MATCH (b {id: r.to_id})
    MERGE (a)-[rel:`REL`]->(b)
    RETURN count(rel) AS c
    """

    with neo4j_driver.session() as session:
        session.run(cypher_nodes, entities=entities, decisions=decisions, evidence=evidence)

        # Create relationships by type. (Neo4j doesn't let dynamic rel type easily without APOC.)
        for rel_type in set(e["rel_type"] for e in edges):
            rel_edges = [e for e in edges if e["rel_type"] == rel_type]
            cy = cypher_edges.replace("`REL`", rel_type)
            session.run(cy, edges=rel_edges)


# ============================================================
# 4) EDGE FUNCTION CALLBACKS
# ============================================================

async def post_progress(job_id: str, progress: int, status: Optional[str] = None, error: Optional[str] = None):
    if not WORKER_PROGRESS_URL:
        return
    headers = {"X-Worker-Secret": WORKER_SECRET}
    payload = {"job_id": job_id, "progress": progress}
    if status:
        payload["status"] = status
    if error:
        payload["error"] = error
    async with httpx.AsyncClient(timeout=30) as client_http:
        await client_http.post(WORKER_PROGRESS_URL, headers=headers, json=payload)


async def post_complete(payload: Dict[str, Any]):
    if not WORKER_COMPLETE_URL:
        return
    headers = {"X-Worker-Secret": WORKER_SECRET}
    async with httpx.AsyncClient(timeout=60) as client_http:
        await client_http.post(WORKER_COMPLETE_URL, headers=headers, json=payload)


# ============================================================
# 5) /start processing pipeline
# ============================================================

@app.post("/start")
async def start(req: StartRequest, x_worker_secret: Optional[str] = Header(None)):
    require_secret(x_worker_secret)
    asyncio.create_task(process_job(req.job_id, req.user_id, req.signed_url))
    return {"ok": True, "started": True, "job_id": req.job_id}


async def process_job(job_id: str, user_id: str, signed_url: str):
    try:
        await post_progress(job_id, 5, status="running")

        # Download export
        async with httpx.AsyncClient(timeout=120) as client_http:
            r = await client_http.get(signed_url)
            r.raise_for_status()
            export_json = r.json()

        await post_progress(job_id, 15, status="running")

        conversations = parse_chatgpt_export(export_json)
        await post_progress(job_id, 25, status="running")

        # Accumulate outputs for Supabase
        entities_out: Dict[str, Dict[str, Any]] = {}  # key by normalized name
        decisions_out: List[Dict[str, Any]] = []
        evidence_out: List[Dict[str, Any]] = []
        edges_out: List[Dict[str, Any]] = []

        # Iterate conversations
        for idx, conv in enumerate(conversations):
            conv_id = conv["conversation_id"]
            msgs = conv["messages"]

            windows = build_windows(msgs, window_size=10, stride=6)
            for w in windows:
                extracted = extract_decisions_from_window(conv_id, w)

                # Map message_id -> text for span computation
                msg_by_id = {m["message_id"]: m["text"] for m in w}

                for d in extracted:
                    decision_id = str(uuid.uuid4())
                    decisions_out.append({
                        "id": decision_id,
                        "user_id": user_id,
                        "title": d["title"][:240],
                        "status": d["status"],
                        "rationale": d["rationale"][:2000],
                        "confidence": float(d.get("confidence", 0.5)),
                        "decided_at": None
                    })

                    # Entities
                    for ent in d.get("entities", []):
                        name = (ent.get("name") or "").strip()
                        if not name:
                            continue
                        key = re.sub(r"\s+", " ", name.lower())
                        if key not in entities_out:
                            ent_id = str(uuid.uuid4())
                            entities_out[key] = {
                                "id": ent_id,
                                "user_id": user_id,
                                "name": name[:200],
                                "canonical_name": None,
                                "entity_type": (ent.get("type") or "other")[:50],
                            }

                        # Edge Entity -> Decision
                        edges_out.append({
                            "id": str(uuid.uuid4()),
                            "user_id": user_id,
                            "from_type": "entity",
                            "from_id": entities_out[key]["id"],
                            "rel_type": "ABOUT",
                            "to_type": "decision",
                            "to_id": decision_id
                        })

                    # Evidence
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
                            "snippet": snippet[:500]
                        })
                        edges_out.append({
                            "id": str(uuid.uuid4()),
                            "user_id": user_id,
                            "from_type": "decision",
                            "from_id": decision_id,
                            "rel_type": "SUPPORTED_BY",
                            "to_type": "evidence",
                            "to_id": ev_id
                        })

            # Update progress as we go
            if idx % 5 == 0:
                pct = 25 + int(55 * (idx + 1) / max(1, len(conversations)))
                await post_progress(job_id, min(pct, 80), status="running")

        # Write Neo4j graph
        await post_progress(job_id, 85, status="running")
        neo4j_upsert_graph(user_id, list(entities_out.values()), decisions_out, evidence_out, edges_out)

        # Persist via worker_complete
        await post_progress(job_id, 95, status="running")
        await post_complete({
            "job_id": job_id,
            "user_id": user_id,
            "entities": list(entities_out.values()),
            "decisions": decisions_out,
            "evidence": evidence_out,
            "edges": edges_out
        })

    except Exception as e:
        await post_progress(job_id, 0, status="failed", error=str(e))


# ============================================================
# 6) /query (Neo4j traversal + cited answer)
# ============================================================

@app.post("/query")
def query(req: QueryRequest, x_worker_secret: Optional[str] = Header(None)):
    require_secret(x_worker_secret)
    if neo4j_driver is None:
        return {"answer": [], "results": []}

    term = (req.question or "").strip()
    if not term:
        return {"answer": [], "results": []}

    # 1) naive entity term extraction: take last 1–3 words after "about"
    m = re.search(r"\babout\b(.+)$", term, flags=re.I)
    entity_term = (m.group(1).strip() if m else term).strip()
    entity_term = entity_term[:80]

    top_k = int(req.top_k or 3)

    with neo4j_driver.session() as session:
        # Find matching entities for this user
        ent_rows = session.run(
            """
            MATCH (e:Entity)
            WHERE e.user_id = $user_id AND toLower(e.name) CONTAINS toLower($q)
            RETURN e.id AS id, e.name AS name
            LIMIT 5
            """,
            user_id=req.user_id,
            q=entity_term
        ).data()

        if not ent_rows:
            return {"answer": [], "results": []}

        results = []
        for ent in ent_rows:
            # Traverse to decisions and evidence
            rows = session.run(
                """
                MATCH (e:Entity {id: $eid})-[:ABOUT]->(d:Decision)
                OPTIONAL MATCH (d)-[:SUPPORTED_BY]->(v:Evidence)
                WHERE d.user_id = $user_id
                RETURN d.id AS decision_id, d.title AS title, d.status AS status, d.confidence AS confidence,
                       collect({id: v.id, snippet: v.snippet, conversation_id: v.conversation_id, message_id: v.message_id,
                                start_char: v.start_char, end_char: v.end_char}) AS citations
                ORDER BY d.confidence DESC
                LIMIT $k
                """,
                eid=ent["id"], user_id=req.user_id, k=top_k
            ).data()

            for r in rows:
                citations = [c for c in r["citations"] if c.get("id")]
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
                    "citations": citations
                })

        # 2) Answer bullets: strictly grounded in citations (simple V1)
        # You can swap to OpenAI generation later; this is deterministic + safe:
        answer = []
        for item in results[:top_k]:
            if item["citations"]:
                answer.append(item["title"])

        return {"answer": answer[:3], "results": results[:top_k]}
