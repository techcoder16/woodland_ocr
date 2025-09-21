import hashlib
import os
from datetime import datetime, timezone
from typing import Optional
import requests
from fastapi import FastAPI, UploadFile, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
import logging
from config import config
from models import ProcessedDocument, APIUsage
from database import SessionLocal, engine

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Document Processing API", version="1.0.0")

# -----------------------------
# Database Models
# -----------------------------

# -----------------------------
# Database Setup
# -----------------------------
# -----------------------------
# Configuration
# -----------------------------

# -----------------------------
# Pydantic Models
# -----------------------------
class ProcessResponse(BaseModel):
    cached: bool
    result: dict
    usage: dict
    used_key: Optional[str] = None
    message: Optional[str] = None

# -----------------------------
# Database Dependency
# -----------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -----------------------------
# Helper Functions
# -----------------------------
def hash_image(image_bytes: bytes) -> str:
    """Return a SHA256 hash of the image bytes."""
    return hashlib.sha256(image_bytes).hexdigest()

def get_current_month() -> str:
    """Get current month in YYYY-MM format."""
    return datetime.now().strftime("%Y-%m")

def get_or_create_usage_record(db: Session, key_name: str, month: str) -> APIUsage:
    """Get or create usage record for API key and month."""
    usage = db.query(APIUsage).filter(
        APIUsage.api_key_name == key_name,
        APIUsage.month_year == month
    ).first()
    
    if not usage:
        usage = APIUsage(
            api_key_name=key_name,
            month_year=month,
            request_count=0
        )
        db.add(usage)
        db.commit()
        db.refresh(usage)
    
    return usage

def check_monthly_limit(db: Session, key_name: str) -> tuple[bool, int]:
    """Check if API key is within monthly limit. Returns (can_use, current_count)."""
    current_month = get_current_month()
    usage = get_or_create_usage_record(db, key_name, current_month)
    
    can_use = usage.request_count < config.MONTHLY_LIMIT
    return can_use, usage.request_count

def increment_usage(db: Session, key_name: str):
    """Increment usage count for API key."""
    current_month = get_current_month()
    usage = get_or_create_usage_record(db, key_name, current_month)
    usage.request_count += 1
    usage.updated_at = datetime.utcnow()
    db.commit()

def get_usage_summary(db: Session) -> dict:
    """Get usage summary for all API keys."""
    current_month = get_current_month()
    summary = {}
    
    for i, key in enumerate(config.API_KEYS, 1):
        key_name = f"api_key_{i}"
        can_use, current_count = check_monthly_limit(db, key_name)
        summary[key_name] = {
            "current_count": current_count,
            "monthly_limit": config.MONTHLY_LIMIT,
            "remaining": config.MONTHLY_LIMIT - current_count,
            "can_use": can_use
        }
    
    return summary

def call_docstrange(api_key: str, image_bytes: bytes, prompt: str) -> Optional[dict]:
    """Make a request to Docstrange API with a prompt."""
    files = {"file": ("upload.png", image_bytes, "image/png")}
    data = {"prompt": prompt}
    
    try:
        response = requests.post(
            config.DOCSTRANGE_API_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            files=files,
            data=data,
            timeout=config.REQUEST_TIMEOUT
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"API request failed with status {response.status_code}: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"API request exception: {str(e)}")
        return None

# -----------------------------
# Enhanced Prompts
# -----------------------------
PROMPTS = {
    "default": "Extract all text, format tables in Markdown and equations in LaTeX. Preserve document structure.",
    "text_only": "Extract only the plain text content from the document, maintaining paragraph structure.",
    "tables_focus": "Focus on extracting tables and format them in clean Markdown table format. Include any surrounding context.",
    "equations_focus": "Extract mathematical equations and formulas in LaTeX format. Include surrounding explanatory text.",
    "structured_data": "Extract structured data including headers, lists, tables, and key-value pairs. Format in JSON structure where possible.",
    "form_data": "Extract form fields, labels, and filled values. Present as key-value pairs.",
    "financial": "Extract financial data including amounts, dates, account numbers, and transaction details.",
    "medical": "Extract medical information including patient data, diagnoses, medications, and dates (ensure HIPAA compliance).",
    "legal": "Extract legal document content including clauses, sections, dates, and parties involved.",
    "academic": "Extract academic content including citations, formulas, figures, and references in appropriate format."
}

# -----------------------------
# API Endpoints
# -----------------------------
@app.get("/")
async def root():
    return {"message": "Document Processing API is running"}

@app.get("/prompts")
async def get_available_prompts():
    """Get list of available prompt templates."""
    return {
        "prompts": {key: desc for key, desc in PROMPTS.items()},
        "usage": "Use 'prompt_type' parameter with one of these keys, or provide custom 'prompt'"
    }

@app.get("/usage")
async def get_usage_stats(db: Session = Depends(get_db)):
    """Get current API usage statistics."""
    return get_usage_summary(db)

@app.post("/process-docs", response_model=ProcessResponse)
async def process_docs(
    file: UploadFile,
    prompt_type: str = Form("default"),
    custom_prompt: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Process document with OCR and optional custom prompting.
    
    Parameters:
    - file: Image file to process
    - prompt_type: One of the predefined prompt types (see /prompts endpoint)
    - custom_prompt: Custom prompt to override prompt_type
    """
    # Validate API keys
    if not config.API_KEYS:
        raise HTTPException(status_code=500, detail="No API keys configured")
    
    # Read and hash image
    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")
    
    img_hash = hash_image(image_bytes)
    
    # Determine prompt to use
    if custom_prompt:
        final_prompt = custom_prompt
    elif prompt_type in PROMPTS:
        final_prompt = PROMPTS[prompt_type]
    else:
        final_prompt = PROMPTS["default"]
        logger.warning(f"Unknown prompt_type '{prompt_type}', using default")
    
    # Check cache first
    cached_doc = db.query(ProcessedDocument).filter(
        ProcessedDocument.image_hash == img_hash,
        ProcessedDocument.prompt == final_prompt
    ).first()
    
    if cached_doc:
        logger.info(f"Cache hit for image hash: {img_hash}")
        return ProcessResponse(
            cached=True,
            result=eval(cached_doc.result),  # Convert string back to dict
            usage=get_usage_summary(db),
            message="Result retrieved from cache"
        )
    
    # Try API keys in order
    result = None
    used_key_name = None
    
    for i, api_key in enumerate(config.API_KEYS, 1):
        key_name = f"api_key_{i}"
        
        # Check monthly limit
        can_use, current_count = check_monthly_limit(db, key_name)
        if not can_use:
            logger.warning(f"{key_name} has reached monthly limit ({current_count}/{config.MONTHLY_LIMIT})")
            continue
        
        # Make API call
        logger.info(f"Trying {key_name} (usage: {current_count}/{config.MONTHLY_LIMIT})")
        result = call_docstrange(api_key, image_bytes, final_prompt)
        
        if result:
            # Success - increment usage and cache result
            increment_usage(db, key_name)
            used_key_name = key_name
            
            # Cache the result
            cached_doc = ProcessedDocument(
                image_hash=img_hash,
                prompt=final_prompt,
                result=str(result),  # Store as string in DB
                file_name=file.filename or "unknown"
            )
            db.add(cached_doc)
            db.commit()
            
            logger.info(f"Successfully processed with {key_name}")
            break
        else:
            logger.warning(f"{key_name} failed to process document")
    
    if not result:
        raise HTTPException(
            status_code=503, 
            detail="All API keys failed or reached monthly limits"
        )
    
    return ProcessResponse(
        cached=False,
        result=result,
        usage=get_usage_summary(db),
        used_key=used_key_name,
        message="Document processed successfully"
    )

@app.delete("/cache/clear")
async def clear_cache(db: Session = Depends(get_db)):
    """Clear all cached documents."""
    count = db.query(ProcessedDocument).count()
    db.query(ProcessedDocument).delete()
    db.commit()
    return {"message": f"Cleared {count} cached documents"}

@app.get("/cache/stats")
async def cache_stats(db: Session = Depends(get_db)):
    """Get cache statistics."""
    total_docs = db.query(ProcessedDocument).count()
    return {
        "total_cached_documents": total_docs,
        "cache_size": f"{total_docs} documents"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5006)