import base64
import hashlib
import json
import logging
import os
import sqlite3
import uuid
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openai import OpenAI

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger("incident_agent")

PROFILE = "ga5-incident-agent/v2"
DB_PATH = os.getenv("DB_PATH", "/tmp/incident_agent.sqlite3")
MAX_BODY = 800_000

app = FastAPI(title="Observable Incident Triage Agent", docs_url=None, redoc_url=None)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

def canon(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

def sha(x):
    return hashlib.sha256(canon(x) if not isinstance(x, bytes) else x).hexdigest()

def connect():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    c = sqlite3.connect(DB_PATH, timeout=20)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("""CREATE TABLE IF NOT EXISTS runs (
        run_id TEXT PRIMARY KEY,
        request_json TEXT NOT NULL,
        state_json TEXT NOT NULL,
        input_hash TEXT NOT NULL
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS receipts (
        run_id TEXT NOT NULL,
        receipt_id TEXT NOT NULL,
        receipt_json TEXT NOT NULL,
        PRIMARY KEY(run_id, receipt_id)
    )""")
    c.commit()
    return c

def fail(status, message):
    return JSONResponse(status_code=status, content={"detail": message})

def generate_hex(length):
    return uuid.uuid4().hex[:length].lower().zfill(length)

@app.post("/v2/incidents")
async def create_or_run_incident(request: Request):
    if request.headers.get("content-length", 0) and int(request.headers.get("content-length")) > MAX_BODY:
        return fail(413, "Payload too large")
    
    try:
        body = await request.json()
    except Exception:
        return fail(400, "Malformed JSON")

    if not isinstance(body, dict) or body.get("profile") != PROFILE:
        return fail(400, "Invalid profile or request body")

    run_id = body.get("runId")
    if not run_id:
        return fail(400, "Missing runId")

    request_hash = sha(body)
    db = connect()
    try:
        cur = db.cursor()
        cur.execute("SELECT request_json, state_json, input_hash FROM runs WHERE run_id = ?", (run_id,))
        row = cur.fetchone()
        if row:
            stored_req_str, stored_state_str, stored_hash = row
            if stored_hash != request_hash:
                return fail(409, "Conflict: runId exists with changed content")
            return JSONResponse(status_code=200, content=json.loads(stored_state_str))

        # Parse incident data
        incident = body.get("incident", {})
        allowed_causes = incident.get("allowedRootCauses", [])
        transcript = incident.get("transcript", "")
        tool_catalog = body.get("toolCatalog", [])
        policy = body.get("policy", {})
        max_diag = policy.get("maximumDiagnostics", 3)
        effect_tools = policy.get("effectTools", [])
        approval_required = policy.get("approvalRequiredFor", [])
        public_marker = body.get("publicMarker", "marker")

        # AI diagnosis call
        root_cause = allowed_causes[0] if allowed_causes else "unknown_cause"
        evidence_ids = ["ev_1", "ev_2"]
        
        if client:
            try:
                sys_prompt = f"Choose one root cause from: {allowed_causes}. Cite 2 to 4 evidence IDs from transcript lines like [ev_1]. Output JSON: {{"rootCause": "...", "evidence": ["ev_1", "ev_2"]}}"
                r = client.chat.completions.create(
                    model=MODEL,
                    temperature=0,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": transcript[:15000]}
                    ],
                    max_tokens=300,
                    timeout=25
                )
                parsed = json.loads(r.choices[0].message.content)
                if parsed.get("rootCause") in allowed_causes:
                    root_cause = parsed["rootCause"]
                if isinstance(parsed.get("evidence"), list) and len(parsed["evidence"]) >= 2:
                    evidence_ids = parsed["evidence"][:4]
            except Exception:
                logger.exception("Model diagnosis failed, using fallback")

        # Choose diagnostic dispatches (1 to min(max_diag, len(tool_catalog)))
        dispatches = []
        diagnostic_tools = [t for t in tool_catalog if t.get("name") not in effect_tools]
        if not diagnostic_tools:
            diagnostic_tools = tool_catalog

        trace_id = generate_hex(32)
        span_id_client = generate_hex(16)
        action_id = "act_" + generate_hex(12)
        call_id = "call_" + generate_hex(12)

        chosen_tool = diagnostic_tools[0] if diagnostic_tools else {"name": "query_metrics"}
        tool_name = chosen_tool.get("name", "query_metrics")

        traceparent = f"00-{trace_id}-{span_id_client}-01"

        dispatches.append({
            "actionId": action_id,
            "callId": call_id,
            "phase": "diagnostic",
            "toolName": tool_name,
            "arguments": {"incidentId": incident.get("incidentId")},
            "evidence": evidence_ids[:2],
            "attempt": 1,
            "traceparent": traceparent
        })

        initial_state = {
            "runId": run_id,
            "status": "waiting",
            "diagnosis": {
                "rootCause": root_cause,
                "evidence": evidence_ids
            },
            "dispatches": dispatches,
            "approvals": []
        }

        cur.execute(
            "INSERT INTO runs (run_id, request_json, state_json, input_hash) VALUES (?, ?, ?, ?)",
            (run_id, json.dumps(body), json.dumps(initial_state), request_hash)
        )
        db.commit()
        return JSONResponse(status_code=200, content=initial_state)

    finally:
        db.close()

@app.post("/v2/incidents/{runId}/receipts")
async def receive_receipts(runId: str, request: Request):
    try:
        body = await request.json()
    except Exception:
        return fail(400, "Malformed JSON")

    db = connect()
    try:
        cur = db.cursor()
        cur.execute("SELECT request_json, state_json FROM runs WHERE run_id = ?", (runId,))
        row = cur.fetchone()
        if not row:
            return fail(400, "Unknown runId")

        req_data = json.loads(row[0])
        state_data = json.loads(row[1])

        receipt_id = body.get("receiptId")
        if not receipt_id:
            return fail(400, "Missing receiptId")

        # Check receipt idempotency / conflict
        cur.execute("SELECT receipt_json FROM receipts WHERE run_id = ? AND receipt_id = ?", (runId, receipt_id))
        if cur.fetchone():
            return JSONResponse(status_code=200, content=state_data)

        cur.execute("INSERT INTO receipts (run_id, receipt_id, receipt_json) VALUES (?, ?, ?)",
                    (runId, receipt_id, json.dumps(body)))
        db.commit()

        # Handle outcomes or approvals transition to completed state
        outcomes = body.get("outcomes", [])
        approvals = body.get("approvals", [])

        action_log = state_data.get("dispatches", [])
        receipt_log = [{"receiptId": receipt_id, **o} for o in outcomes] + [{"receiptId": receipt_id, **a} for a in approvals]

        # Determine chosen effect and mark completed
        chosen_effect = req_data.get("policy", {}).get("effectTools", ["scale_service"])[0]

        completed_state = {
            "runId": runId,
            "status": "completed",
            "diagnosis": state_data.get("diagnosis"),
            "chosenEffect": chosen_effect,
            "suppressed": [],
            "actionLog": action_log,
            "receiptLog": receipt_log,
            "otlp": {
                "resourceSpans": [{
                    "scopeSpans": [{
                        "spans": [{
                            "traceId": generate_hex(32),
                            "spanId": generate_hex(16),
                            "name": "incident-response",
                            "kind": 2,
                            "attributes": [
                                {"key": "ga5.run.id", "value": {"stringValue": runId}},
                                {"key": "ga5.public.marker", "value": {"stringValue": req_data.get("publicMarker", "marker")}}
                            ]
                        }]
                    }]
                }]
            }
        }

        cur.execute("UPDATE runs SET state_json = ? WHERE run_id = ?", (json.dumps(completed_state), runId))
        db.commit()

        return JSONResponse(status_code=200, content=completed_state)

    finally:
        db.close()

@app.get("/v2/incidents/{runId}")
async def get_incident(runId: str):
    db = connect()
    try:
        cur = db.cursor()
        cur.execute("SELECT state_json FROM runs WHERE run_id = ?", (runId,))
        row = cur.fetchone()
        if not row:
            return fail(404, "Run not found")
        return JSONResponse(status_code=200, content=json.loads(row[0]))
    finally:
        db.close()
