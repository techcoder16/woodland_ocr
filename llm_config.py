# Woodland OCR LLM configuration — OpenRouter (single provider for both
# document extraction and the chat/agent side, matching agent_brain.py).
# Get an API key from: https://openrouter.ai/keys
import os
from dotenv import load_dotenv  # <-- make sure python-dotenv is installed
load_dotenv(dotenv_path="/app/.env")
load_dotenv()  # also load a local .env when not running in the /app container

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or ""

if not OPENROUTER_API_KEY:
    print("WARNING: OPENROUTER_API_KEY not configured. Set it in the environment or a .env file.")
    print("Get a key from: https://openrouter.ai/keys")

# Rate limiting (seconds between requests)
RATE_LIMIT_DELAY = 1

# Timeout for API requests (seconds)
API_TIMEOUT = 30

# Models to try, in order, for structured transaction extraction. Override
# with OPENROUTER_EXTRACT_MODELS as a comma-separated list.
OPENROUTER_MODELS = [
    m.strip()
    for m in os.getenv(
        "OPENROUTER_EXTRACT_MODELS",
        # claude-3.5-haiku (undated) 404s on OpenRouter — the dated id is the
        # routable one, and an unreachable model wastes a request per call.
        "openai/gpt-4o-mini,anthropic/claude-3.5-haiku-20241022,meta-llama/llama-3.1-8b-instruct",
    ).split(",")
    if m.strip()
]
