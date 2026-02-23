# Weather Agent

## Project Structure

```text
.
|-- app.py                    # Uvicorn entrypoint (`app` object)
|-- frontend/
|   `-- index.html            # Simple UI
|-- weather_agent/
|   |-- api.py                # FastAPI routes and response logic
|   |-- agent_service.py      # LLM agent setup
|   |-- config.py             # Environment and app constants
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

## Environment Variables

- `OPENWEATHER_API_KEY`
- `HUGGINGFACEHUB_API_TOKEN` or `HUGGINGFACE_API_KEY` (optional for LLM mode)

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
