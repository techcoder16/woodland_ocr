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
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install python deps
RUN pip install --no-cache-dir -r requirements.txt

# Install torch + torchvision (CPU-only version, so it runs on your system)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install FastAPI + Uvicorn (in case requirements.txt didn’t have them)
RUN pip install --no-cache-dir fastapi uvicorn[standard]

# Copy project files
COPY . .

# Expose port
EXPOSE 5006

# Start FastAPI app with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5006"]
