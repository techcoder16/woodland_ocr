import os
from dotenv import load_dotenv  # <-- make sure python-dotenv is installed
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
print("Current working dir:", os.getcwd())

print("DOCSTRANGE_API_KEY1:", os.getenv("DOCSTRANGE_API_KEY1"))
print("DOCSTRANGE_API_KEY2:", os.getenv("DOCSTRANGE_API_KEY2"))


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
print(config.API_KEYS)  # Debugging line to check if keys are loaded correctly