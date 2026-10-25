# Invoice Action Agent

FastAPI service for the A2A 1.0 invoice-action assignment.

## Render settings

- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

Set Render environment variables from `.env.example`. Create a managed PostgreSQL database and set `DATABASE_URL` to its internal connection string. Do not use SQLite for the graded deployment because the Render filesystem is ephemeral.

The public Agent Card is served at `/.well-known/agent-card.json`. A2A endpoints are under `/a2a/`.

## Important

Set `PUBLIC_BASE_URL` to the exact public HTTPS base URL you submit, including `/a2a/`. Set a strong `A2A_BEARER_TOKEN`; the same exact token must be used by the grader if the assignment provides it. Configure an OpenAI-compatible AI provider using `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `MODEL_NAME`.
