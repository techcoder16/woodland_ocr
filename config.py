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
    # The bare /extract path is deprecated and now answers 429 with a migration
    # notice rather than doing any work, so default to the v1 sync endpoint.
    DOCSTRANGE_API_URL = os.getenv(
        "DOCSTRANGE_API_URL",
        "https://extraction-api.nanonets.com/api/v1/extract/sync",
    )
    # Valid: html, markdown, bank-statement, alter-bank, csv, json.
    # markdown keeps tables and labels intact, which the date/amount extraction
    # prompts read better than the flattened json shape.
    DOCSTRANGE_OUTPUT_FORMAT = os.getenv("DOCSTRANGE_OUTPUT_FORMAT", "markdown")
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "300"))

config = Config()
