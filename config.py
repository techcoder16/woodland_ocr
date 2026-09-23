import os
from dotenv import load_dotenv  # <-- make sure python-dotenv is installed

# Load .env next to this file as well as the container path — loading only
# /app/.env meant a local `python app.py` run started with no OCR keys at all.
_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path="/app/.env")  # container path, when running in Docker
load_dotenv(dotenv_path=_ENV_PATH)    # local checkout

print("DocStrange keys loaded:", sum(1 for n in ("DOCSTRANGE_API_KEY1", "DOCSTRANGE_API_KEY2") if os.getenv(n)))


class Config:
    API_KEYS = [
        os.getenv("DOCSTRANGE_API_KEY1"),
        os.getenv("DOCSTRANGE_API_KEY2"),
    ]
    # Filter out None values
    API_KEYS = [key for key in API_KEYS if key is not None]
    
    MONTHLY_LIMIT = int(os.getenv("MONTHLY_REQUEST_LIMIT", "10000"))
    DOCSTRANGE_API_URL = os.getenv("DOCSTRANGE_API_URL", "https://api.docstrange.com/v1/ocr")
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "60"))

config = Config()
