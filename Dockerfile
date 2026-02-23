FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

ARG INSTALL_LLM=false

COPY requirements.txt requirements-llm.txt ./
RUN pip install --no-cache-dir --no-compile -r requirements.txt \
    && if [ "$INSTALL_LLM" = "true" ]; then pip install --no-cache-dir --no-compile -r requirements-llm.txt; fi

COPY app.py ./app.py
COPY weather_agent ./weather_agent
COPY frontend ./frontend

EXPOSE 8000

CMD ["uvicorn", "weather_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
