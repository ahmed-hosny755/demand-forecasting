FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HISTORY_PATH=/data/demand_clean.csv \
    MODEL_DIR=/models

WORKDIR /app

# Install dependencies first so this layer is cached when only the code changes
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Only the code goes into the image; models and history data are mounted at run time
COPY src ./src

# Run as a non-root user
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
