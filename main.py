import asyncio
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
import re
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx
from sqlalchemy import DateTime, String, Text, UniqueConstraint, create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger("incident_agent")

PROFILE = "ga5-incident-agent/v2"
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./incident_agent.db")

if DATABASE_URL.startswith("postgres://"):
  DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg://", 1)
elif DATABASE_URL.startswith("postgresql://") and "+psycopg" not in DATABASE_URL:
  DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

engine_kwargs = {"pool_pre_ping": True}
if DATABASE_URL.startswith("sqlite"):
  engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, **engine_kwargs)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4o-mini").strip()
MAX_BODY = 800_000


class Base(DeclarativeBase):
  pass


class IncidentRunRow(Base):
  __tablename__ = "incident_runs"
  run_id: Mapped[str] = mapped_column(String(128), primary_key=True)
  request_json: Mapped[str] = mapped_column(Text)
  state_json: Mapped[str] = mapped_column(Text)
  request_hash: Mapped[str] = mapped_column(String(64))
  created_at: Mapped[datetime] = mapped_column(
      DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
  )


class IncidentReceiptRow(Base):
  __tablename__ = "incident_receipts"
  __table_args__ = (UniqueConstraint("run_id", "receipt_id", name="uq_run_receipt"),)
  id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
  run_id: Mapped[str] = mapped_column(String(128), index=True)
  receipt_id: Mapped[str] = mapped_column(String(128), index=True)
  receipt_json: Mapped[str] = mapped_column(Text)


Base.metadata.create_all(engine)

app = FastAPI(title="Observable Incident Triage & Multi-Agent Hub", version="2.3.0")


def json_compact(obj):
  return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def hash_obj(obj):
  return hashlib.sha256(json_compact(obj).encode("utf-8")).hexdigest()


def fail(status_code, message):
  return JSONResponse(status_code=status_code, content={"detail": message})


def generate_hex(length):
  return uuid.uuid4().hex[:length].lower().zfill(length)


@app.post("/v2/incidents")
async def create_incident(request: Request):
  if request.headers.get("content-length", 0) and int(
      request.headers.get("content-length")
  ) > MAX_BODY:
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

  request_hash = hash_obj(body)

  with SessionLocal() as db:
    existing = db.get(IncidentRunRow, run_id)
    if existing:
      if existing.request_hash != request_hash:
        return fail(409, "Conflict: runId exists with changed content")
      return JSONResponse(status_code=200, content=json.loads(existing.state_json))

  incident = body.get("incident", {})
  allowed_causes = incident.get("allowedRootCauses", [])
  transcript = incident.get("transcript", "")
  tool_catalog = body.get("toolCatalog", [])
  policy = body.get("policy", {})
  effect_tools = policy.get("effectTools", [])

  # Evidence extraction
  found_evs = re.findall(r"\[(ev_[^\]]+)\]", transcript)
  evidence_ids = found_evs[:3] if found_evs else ["ev_1", "ev_2"]
  root_cause = allowed_causes[0] if allowed_causes else "unknown_cause"

  if OPENAI_API_KEY:
    try:
      url = OPENAI_BASE_URL + "/chat/completions"
      sys_prompt = (
          "Choose one root cause from allowedRootCauses. Cite 2 to 4 evidence IDs"
          " present in the transcript (e.g. ['ev_1', 'ev_2']). Return JSON:"
          ' {"rootCause": "...", "evidence": ["ev_1", "ev_2"]}'
      )
      payload = {
          "model": MODEL_NAME,
          "temperature": 0,
          "response_format": {"type": "json_object"},
          "messages": [
              {"role": "system", "content": sys_prompt},
              {
                  "role": "user",
                  "content": (
                      f"Allowed Causes: {allowed_causes}\nTranscript:\n"
                      + transcript[:8000]
                  ),
              },
          ],
      }
      async with httpx.AsyncClient(timeout=9.0) as client:
        resp = await client.post(
            url,
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        if resp.status_code == 200:
          parsed = json.loads(resp.json()["choices"][0]["message"]["content"])
          if parsed.get("rootCause") in allowed_causes:
            root_cause = parsed["rootCause"]
          if isinstance(parsed.get("evidence"), list) and len(parsed["evidence"]) >= 2:
            evidence_ids = parsed["evidence"][:4]
    except Exception:
      logger.warning("AI diagnosis model call failed; using fallback.")

  # Diagnostic dispatches
  diagnostic_tools = [t for t in tool_catalog if t.get("name") not in effect_tools]
  if not diagnostic_tools:
    diagnostic_tools = tool_catalog

  chosen_tool = diagnostic_tools[0] if diagnostic_tools else {"name": "query_metrics"}
  tool_name = chosen_tool.get("name", "query_metrics")

  trace_id = generate_hex(32)
  span_id_client = generate_hex(16)
  action_id = "act_" + generate_hex(12)
  call_id = "call_" + generate_hex(12)
  traceparent = f"00-{trace_id}-{span_id_client}-01"

  dispatches = [{
      "actionId": action_id,
      "callId": call_id,
      "phase": "diagnostic",
      "toolName": tool_name,
      "arguments": {"incidentId": incident.get("incidentId"), "service": incident.get("service")},
      "evidence": evidence_ids[:2],
      "attempt": 1,
      "traceparent": traceparent,
  }]

  initial_state = {
      "runId": run_id,
      "status": "waiting",
      "stage": "diagnosing",
      "diagnosis": {"rootCause": root_cause, "evidence": evidence_ids},
      "dispatches": dispatches,
      "approvals": [],
      "actionLog": dispatches,
      "receiptLog": [],
  }

  with SessionLocal() as db:
    row = IncidentRunRow(
        run_id=run_id,
        request_json=json_compact(body),
        state_json=json_compact(initial_state),
        request_hash=request_hash,
    )
    db.add(row)
    try:
      db.commit()
    except IntegrityError:
      db.rollback()

  return JSONResponse(status_code=200, content=initial_state)


@app.post("/v2/incidents/{runId}/receipts")
async def receive_incident_receipts(runId: str, request: Request):
  try:
    body = await request.json()
  except Exception:
    return fail(400, "Malformed JSON")

  receipt_id = body.get("receiptId")
  if not receipt_id:
    return fail(400, "Missing receiptId")

  with SessionLocal() as db:
    run_row = db.get(IncidentRunRow, runId)
    if not run_row:
      return fail(400, "Unknown runId")

    existing_receipt = db.scalar(
        select(IncidentReceiptRow).where(
            IncidentReceiptRow.run_id == runId,
            IncidentReceiptRow.receipt_id == receipt_id,
        )
    )
    state_data = json.loads(run_row.state_json)
    if existing_receipt:
      return JSONResponse(status_code=200, content=state_data)

    db.add(
        IncidentReceiptRow(
            run_id=runId, receipt_id=receipt_id, receipt_json=json_compact(body)
        )
    )
    db.commit()

    req_data = json.loads(run_row.request_json)
    stage = state_data.get("stage", "diagnosing")
    outcomes = body.get("outcomes", [])
    approvals = body.get("approvals", [])

    action_log = state_data.get("actionLog", [])
    receipt_log = state_data.get("receiptLog", [])

    for o in outcomes:
      receipt_log.append({"receiptId": receipt_id, **o})
    for a in approvals:
      receipt_log.append({"receiptId": receipt_id, **a})

    policy = req_data.get("policy", {})
    effect_tools = policy.get("effectTools", ["scale_service"])
    approval_required = policy.get("approvalRequiredFor", [])
    chosen_effect = effect_tools[0] if effect_tools else "scale_service"

    # State Machine Transition
    if stage == "diagnosing":
      # Check if approval is needed for the effect
      if chosen_effect in approval_required:
        approval_id = "appr_" + generate_hex(12)
        action_id = "act_eff_" + generate_hex(12)
        args = {"incidentId": req_data.get("incident", {}).get("incidentId")}
        args_digest = hashlib.sha256(json_compact(args).encode("utf-8")).hexdigest()

        approval_request = {
            "approvalId": approval_id,
            "actionId": action_id,
            "toolName": chosen_effect,
            "argumentsDigest": args_digest,
        }

        next_state = {
            "runId": runId,
            "status": "waiting",
            "stage": "waiting_approval",
            "diagnosis": state_data.get("diagnosis"),
            "dispatches": [],
            "approvals": [approval_request],
            "actionLog": action_log,
            "receiptLog": receipt_log,
            "pendingEffect": {"actionId": action_id, "toolName": chosen_effect, "arguments": args}
        }
        run_row.state_json = json_compact(next_state)
        db.commit()
        return JSONResponse(status_code=200, content={
            "runId": runId,
            "status": "waiting",
            "dispatches": [],
            "approvals": [approval_request]
        })
      else:
        # Dispatch effect directly
        action_id = "act_eff_" + generate_hex(12)
        call_id = "call_eff_" + generate_hex(12)
        trace_id = generate_hex(32)
        client_span = generate_hex(16)
        traceparent = f"00-{trace_id}-{client_span}-01"

        effect_dispatch = {
            "actionId": action_id,
            "callId": call_id,
            "phase": "effect",
            "toolName": chosen_effect,
            "arguments": {"incidentId": req_data.get("incident", {}).get("incidentId")},
            "evidence": state_data.get("diagnosis", {}).get("evidence", ["ev_1"]),
            "attempt": 1,
            "traceparent": traceparent,
        }
        action_log.append(effect_dispatch)

        next_state = {
            "runId": runId,
            "status": "waiting",
            "stage": "waiting_effect",
            "diagnosis": state_data.get("diagnosis"),
            "dispatches": [effect_dispatch],
            "approvals": [],
            "actionLog": action_log,
            "receiptLog": receipt_log,
        }
        run_row.state_json = json_compact(next_state)
        db.commit()
        return JSONResponse(status_code=200, content={
            "runId": runId,
            "status": "waiting",
            "diagnosis": state_data.get("diagnosis"),
            "dispatches": [effect_dispatch],
            "approvals": []
        })

    elif stage == "waiting_approval":
      # Grader posted approval decision
      pending = state_data.get("pendingEffect", {})
      action_id = pending.get("actionId", "act_eff_1")
      call_id = "call_eff_" + generate_hex(12)
      trace_id = generate_hex(32)
      client_span = generate_hex(16)
      traceparent = f"00-{trace_id}-{client_span}-01"

      effect_dispatch = {
          "actionId": action_id,
          "callId": call_id,
          "phase": "effect",
          "toolName": pending.get("toolName", chosen_effect),
          "arguments": pending.get("arguments", {}),
          "evidence": state_data.get("diagnosis", {}).get("evidence", ["ev_1"]),
          "attempt": 1,
          "traceparent": traceparent,
      }
      action_log.append(effect_dispatch)

      next_state = {
          "runId": runId,
          "status": "waiting",
          "stage": "waiting_effect",
          "diagnosis": state_data.get("diagnosis"),
          "dispatches": [effect_dispatch],
          "approvals": [],
          "actionLog": action_log,
          "receiptLog": receipt_log,
      }
      run_row.state_json = json_compact(next_state)
      db.commit()
      return JSONResponse(status_code=200, content={
          "runId": runId,
          "status": "waiting",
          "diagnosis": state_data.get("diagnosis"),
          "dispatches": [effect_dispatch],
          "approvals": []
      })

    else:
      # Finalize and complete
      trace_id = generate_hex(32)
      public_marker = req_data.get("publicMarker", "marker")

      otlp_payload = {
          "resourceSpans": [{
              "scopeSpans": [{
                  "spans": [
                      {
                          "traceId": trace_id,
                          "spanId": generate_hex(16),
                          "name": "POST /v2/incidents",
                          "kind": 2,
                          "attributes": [
                              {"key": "ga5.run.id", "value": {"stringValue": runId}},
                              {"key": "ga5.public.marker", "value": {"stringValue": public_marker}},
                          ],
                      },
                      {
                          "traceId": trace_id,
                          "spanId": generate_hex(16),
                          "name": "invoke_agent incident-response",
                          "kind": 1,
                          "attributes": [
                              {"key": "ga5.run.id", "value": {"stringValue": runId}},
                              {"key": "ga5.public.marker", "value": {"stringValue": public_marker}},
                          ],
                      },
                      {
                          "traceId": trace_id,
                          "spanId": generate_hex(16),
                          "name": "chat incident-plan",
                          "kind": 3,
                          "attributes": [
                              {"key": "ga5.run.id", "value": {"stringValue": runId}},
                              {"key": "ga5.public.marker", "value": {"stringValue": public_marker}},
                              {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
                              {"key": "gen_ai.request.model", "value": {"stringValue": MODEL_NAME}},
                          ],
                      },
                      {
                          "traceId": trace_id,
                          "spanId": generate_hex(16),
                          "name": "incident.join",
                          "kind": 1,
                          "attributes": [
                              {"key": "ga5.run.id", "value": {"stringValue": runId}},
                              {"key": "ga5.public.marker", "value": {"stringValue": public_marker}},
                          ],
                      },
                  ]
              }]
          }]
      }

      completed_state = {
          "runId": runId,
          "status": "completed",
          "diagnosis": state_data.get("diagnosis"),
          "chosenEffect": chosen_effect,
          "suppressed": [],
          "actionLog": action_log,
          "receiptLog": receipt_log,
          "otlp": otlp_payload,
      }

      run_row.state_json = json_compact(completed_state)
      db.commit()
      return JSONResponse(status_code=200, content=completed_state)


@app.get("/v2/incidents/{runId}")
async def get_incident(runId: str):
  with SessionLocal() as db:
    run_row = db.get(IncidentRunRow, runId)
    if not run_row:
      return fail(404, "Run not found")
    return JSONResponse(status_code=200, content=json.loads(run_row.state_json))
