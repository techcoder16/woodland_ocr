from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, Text

class ProcessResponse(BaseModel):
    cached: bool
    result: dict
    usage: dict
    used_key: Optional[str] = "main_api_key"
    message: Optional[str] = None
