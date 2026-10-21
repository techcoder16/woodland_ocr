# Production image: no .env baked in (config comes from ECS env/SSM), no reload.
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN rm -f .env && mkdir -p /app/uploads /tmp/ocr-cache

EXPOSE 5006
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5006"]
