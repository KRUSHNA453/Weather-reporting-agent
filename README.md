# Weather Agent

## Project Structure

```text
.
|-- app.py                    # Uvicorn entrypoint (`app` object)
|-- frontend/
|   `-- index.html            # Simple UI
|-- weather_agent/
|   |-- api.py                # FastAPI routes and response logic
|   |-- autonomous_agent.py   # Plan -> tool -> observe -> reflect loop
|   |-- agent_service.py      # LLM agent setup
|   |-- config.py             # Environment and app constants
|   |-- memory_store.py       # SQLite long-term memory (profiles + history)
|   |-- personas.py           # Persona catalog and style adaptation
|   |-- prompts.py            # Agent persona/system prompt
|   |-- schemas.py            # Request/response models
|   `-- weather_service.py    # OpenWeather API integration
|-- Dockerfile
|-- docker-compose.yml
`-- requirements.txt
```

## Run Locally

```bash
uvicorn app:app --reload
```

Open:
- UI: `http://127.0.0.1:8000/ui`
- Docs: `http://127.0.0.1:8000/docs`

## Autonomous Loop + Memory

- The chat endpoint now runs a multi-step autonomous loop:
  - `plan -> tool-call -> observe -> reflect -> final-answer`
- `/chat` is now **LLM-first** when HuggingFace access is available:
  - Executes LLM agent + tool orchestration via `agent_service`
  - Extracts tool observations from intermediate steps
  - Falls back to direct deterministic tool execution if LLM call/tool parsing fails
- Main UI uses persona-first flow with 5 personas:
  - `professional`, `friendly`, `analyst`, `teacher`, `safety`
- Long-term memory and trace features remain available via API, but are disabled by default in chat requests.

### Example `POST /chat`

```json
{
  "message": "What is the future weather in Chennai?",
  "persona_id": "teacher",
  "preferences": {
    "units": "metric"
  }
}
```

### Extra endpoints

- `GET /personas`
- `GET /users/{user_id}/profile`

## Environment Variables

- `OPENWEATHER_API_KEY`
- `HUGGINGFACEHUB_API_TOKEN` or `HUGGINGFACE_API_KEY` (optional for LLM mode)
- `AGENT_MEMORY_DB_PATH` (optional; default `agent_memory.db` in project root)

## Deploy on Render

This repo includes a `render.yaml` Blueprint for Render.

1. Push this project to a GitHub/GitLab repository.
2. In Render, create a new Blueprint service and select the repo.
3. Set `OPENWEATHER_API_KEY` in Render environment variables.
4. Set `HUGGINGFACE_API_KEY` (or `HUGGINGFACEHUB_API_TOKEN`). `INSTALL_LLM` is enabled by default in `render.yaml`.

After deploy:
- Base URL: `https://<your-service>.onrender.com`
- Health check: `/health`
- UI: `/ui`
- API docs: `/docs`

## Docker Size Optimization

- Default Docker build uses a lean dependency set (no LangChain/HuggingFace packages).
- To enable full LLM agent mode in Docker, build with:

```bash
docker build --build-arg INSTALL_LLM=true -t weather-agent:latest .
```

## Docker Memory Persistence

- `docker-compose.yml` mounts `weather-agent-data` volume to `/app/data`.
- The app uses `AGENT_MEMORY_DB_PATH=/app/data/agent_memory.db`, so long-term memory persists across container restarts.
