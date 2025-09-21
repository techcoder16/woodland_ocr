# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

COPY .env ./

# Create uploads directory
RUN mkdir -p /app/uploads

# Expose port
EXPOSE 5006

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5006", "--reload"]