# Groq Free API Configuration
# Get your free API key from: https://console.groq.com/keys
import os
from dotenv import load_dotenv  # <-- make sure python-dotenv is installed
load_dotenv(dotenv_path="/app/.env")

# Groq API (Free tier: 14,400 requests per day)
GROQ_TOKEN = os.getenv("GROQ_TOKEN") or "gsk_your_token_here"  # Replace with your actual Groq token

# Rate limiting (seconds between requests)
RATE_LIMIT_DELAY = 1

# Timeout for API requests (seconds)
API_TIMEOUT = 30

# Groq models to try (free models available)
GROQ_MODELS = [
    "llama-3.1-8b-instant",  # Fast and reliable
    "llama-3.1-70b-versatile",  # More capable but slower
    "mixtral-8x7b-32768"  # Alternative model
]
GROQ_MODEL = GROQ_MODELS[0]  # Default to first model