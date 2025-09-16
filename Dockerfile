# Use lightweight Python image
FROM python:3.10-slim

# Prevents python from writing pyc files and ensures stdout/stderr are flushed
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install python deps
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install uvicorn[standard]
RUN pip install fastapi

# Copy project
COPY . .

# Expose port
EXPOSE 5006

# Start FastAPI app with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5006"]
