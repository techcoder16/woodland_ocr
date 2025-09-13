FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git libgl1 && rm -rf /var/lib/apt/lists/*

# Copy files
COPY requirements.txt .
COPY app.py .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download model into container at build time (optional: speeds up runtime startup)
RUN python -c "from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer; \
    m='nanonets/Nanonets-OCR-s'; \
    AutoModelForImageTextToText.from_pretrained(m); \
    AutoProcessor.from_pretrained(m); \
    AutoTokenizer.from_pretrained(m)"

# Expose port
EXPOSE 5006

# Run API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5006"]
