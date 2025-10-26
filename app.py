import hashlib
import os
from datetime import datetime, timezone
from typing import Optional
import requests
from fastapi import FastAPI, UploadFile, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
# from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, Boolean
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker, Session
import logging
from config import config

from pydantic import BaseModel
# from database import Base

# from models import ProcessedDocument, APIUsage
# from database import SessionLocal, engine

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# from pydantic_model import ProcessResponse
app = FastAPI(title="Document Processing API", version="1.0.0")
# Base.metadata.create_all(bind=engine)

# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# -----------------------------
# Helper Functions
# -----------------------------
def hash_image(image_bytes: bytes) -> str:
    """Return a SHA256 hash of the image bytes."""
    return hashlib.sha256(image_bytes).hexdigest()

def get_current_month() -> str:
    """Get current month in YYYY-MM format."""
    return datetime.now().strftime("%Y-%m")

# def get_or_create_usage_record(db: Session, key_name: str, month: str) -> APIUsage:
#     """Get or create usage record for API key and month."""
#     usage = db.query(APIUsage).filter(
#         APIUsage.api_key_name == key_name,
#         APIUsage.month_year == month
#     ).first()
    
#     if not usage:
#         usage = APIUsage(
#             api_key_name=key_name,
#             month_year=month,
#             request_count=0
#         )
#         db.add(usage)
#         db.commit()
#         db.refresh(usage)
    
#     return usage

# def check_monthly_limit(db: Session, key_name: str) -> tuple[bool, int]:
#     """Check if API key is within monthly limit. Returns (can_use, current_count)."""
#     current_month = get_current_month()
#     usage = get_or_create_usage_record(db, key_name, current_month)
    
#     can_use = usage.request_count < config.MONTHLY_LIMIT
#     return can_use, usage.request_count

# def increment_usage(db: Session, key_name: str):
#     """Increment usage count for API key."""
#     current_month = get_current_month()
#     usage = get_or_create_usage_record(db, key_name, current_month)
#     usage.request_count += 1
#     usage.updated_at = datetime.utcnow()
#     db.commit()

# def get_usage_summary(db: Session) -> dict:
#     """Get usage summary for all API keys."""
#     current_month = get_current_month()
#     summary = {}
    
#     for i, key in enumerate(config.API_KEYS, 1):
#         key_name = f"api_key_{i}"
#         can_use, current_count = check_monthly_limit(db, key_name)
#         summary[key_name] = {
#             "current_count": current_count,
#             "monthly_limit": config.MONTHLY_LIMIT,
#             "remaining": config.MONTHLY_LIMIT - current_count,
#             "can_use": can_use
#         }
    
#     return summary

def call_docstrange(api_key: str, image_bytes: bytes, prompt: str) -> Optional[dict]:
    """Make a request to Docstrange API with a prompt."""
    files = {"file": ("upload.png", image_bytes, "*")}

    data = {"output_type": "flat-json"}
    print("API key being used:", api_key[:10] + "...")  # Debugging line
    try:
        response = requests.post(
            url=config.DOCSTRANGE_API_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            files=files,
            data=data,
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"OCR API response keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            return result
        else:
            logger.error(f"API request failed with status {response.status_code}: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"API request exception: {str(e)}")
        return None

def create_transaction_extraction_prompt(ocr_content):
    """Create a prompt for extracting transaction data from OCR content."""
    return f"""Extract transaction data from this invoice and return as JSON:

{{
  "toLandlordDate": "YYYY-MM-DD or null",
  "toLandLordMode": "string or null", 
  "toLandlordRentReceived": "float or null",
  "toLandlordLessManagementFees": "float or null",
  "toLandlordLessBuildingExpenditure": "float or null",
  "toLandlordLessBuildingExpenditureActual": "float or null",
  "toLandlordLessBuildingExpenditureDifference": "float or null",
  "toLandlordNetPaid": "float or null",
  "toLandlordLessVAT": "float or null",
  "toLandlordChequeNo": "string or null",
  "toLandlordExpenditureDescription": "string or null",
  "toLandlordPaidBy": "string or null",
  "toLandlordDefaultExpenditure": "string or null",
  "toLandlordNetReceived": "float or null"
}}

Invoice content:
{ocr_content}

Return only the JSON object."""

def extract_transaction_data(ocr_content):
    """Extract transaction data using LLM."""
    url = "https://apifreellm.com/api/chat"
    headers = {
        "Content-Type": "application/json"
    }
    
    # Create the extraction prompt
    prompt = create_transaction_extraction_prompt(ocr_content)
    
    data = {
        "message": prompt
    }
    print("Prompt being sent to LLM API:", prompt[:100] + "..." if len(prompt) > 100 else prompt)
    try:
        # Set a shorter timeout to avoid the 2-minute limit
        resp = requests.post(url, headers=headers, json=data, timeout=90)  # 90 seconds timeout
        
        if resp.status_code != 200:
            return {
                'success': False,
                'error': f'API returned status {resp.status_code}: {resp.text}'
            }
        
        js = resp.json()+
        
        if js.get('status') == 'success':
            response_text = js.get('response', '')
            if not response_text.strip():
                return {
                    'success': False,
                    'error': 'Empty response from LLM API',
                    'raw_response': response_text
                }
            
            # Try to parse the JSON response
            import json
            try:
                transaction_data = json.loads(response_text)
                return {
                    'success': True,
                    'data': transaction_data,
                    'raw_response': response_text
                }
            except json.JSONDecodeError as e:
                return {
                    'success': False,
                    'error': f'Failed to parse JSON response: {str(e)}',
                    'raw_response': response_text
                }
        else:
            return {
                'success': False,
                'error': js.get('error', 'Unknown error'),
                'status': js.get('status')
            }
    except requests.exceptions.Timeout:
        logger.warning("LLM API timed out, using fallback extraction")
        return extract_basic_transaction_data(ocr_content)
    except requests.exceptions.RequestException as e:
        return {
            'success': False,
            'error': f'Request error: {str(e)}'
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Unexpected error: {str(e)}'
        }

def extract_basic_transaction_data(ocr_content):
    """Fallback: Extract basic transaction data using simple pattern matching."""
    import re
    from datetime import datetime
    
    result = {
        "toLandlordDate": None,
        "toLandLordMode": None,
        "toLandlordRentReceived": None,
        "toLandlordLessManagementFees": None,
        "toLandlordLessBuildingExpenditure": None,
        "toLandlordLessBuildingExpenditureActual": None,
        "toLandlordLessBuildingExpenditureDifference": None,
        "toLandlordNetPaid": None,
        "toLandlordLessVAT": None,
        "toLandlordChequeNo": None,
        "toLandlordExpenditureDescription": None,
        "toLandlordPaidBy": None,
        "toLandlordDefaultExpenditure": None,
        "toLandlordNetReceived": None
    }
    
    try:
        # Extract monetary amounts
        money_pattern = r'[£$€]\s*(\d+(?:\.\d{2})?)'
        amounts = re.findall(money_pattern, ocr_content)
        if amounts:
            # Convert to float and use the largest amount as rent received
            amounts_float = [float(amount) for amount in amounts]
            result["toLandlordRentReceived"] = max(amounts_float)
        
        # Extract dates
        date_patterns = [
            r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})',  # DD/MM/YYYY or DD-MM-YYYY
            r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})',  # YYYY/MM/DD or YYYY-MM-DD
        ]
        
        for pattern in date_patterns:
            dates = re.findall(pattern, ocr_content)
            if dates:
                try:
                    if len(dates[0]) == 3:
                        if len(dates[0][2]) == 4:  # Full year
                            if len(dates[0][0]) == 4:  # YYYY/MM/DD
                                year, month, day = dates[0]
                            else:  # DD/MM/YYYY
                                day, month, year = dates[0]
                        else:  # 2-digit year
                            day, month, year = dates[0]
                            year = f"20{year}" if int(year) < 50 else f"19{year}"
                        
                        result["toLandlordDate"] = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                        break
                except:
                    continue
        
        # Extract description from the content
        lines = ocr_content.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['descaling', 'heating', 'hot water', 'maintenance', 'repair']):
                result["toLandlordExpenditureDescription"] = line.strip()
                break
        
        # Look for VAT information
        vat_pattern = r'VAT[:\s]*[£$€]?\s*(\d+(?:\.\d{2})?)'
        vat_match = re.search(vat_pattern, ocr_content, re.IGNORECASE)
        if vat_match:
            result["toLandlordLessVAT"] = float(vat_match.group(1))
        
        return {
            'success': True,
            'data': result,
            'raw_response': 'Basic extraction fallback used',
            'fallback': True
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Basic extraction failed: {str(e)}'
        }

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

# @app.get("/usage")
# async def get_usage_stats(db: Session = Depends(get_db)):
#     """Get current API usage statistics."""
#     return get_usage_summary(db)

@app.post("/process-docs")
async def process_docs(
    file: UploadFile,
    prompt_type: str = Form("default"),
    custom_prompt: Optional[str] = Form(None),
    extract_transaction: bool = Form(False)
):
    """
    Process document with OCR and optional custom prompting.
    
    Parameters:
    - file: Image file to process
    - prompt_type: One of the predefined prompt types (see /prompts endpoint)
    - custom_prompt: Custom prompt to override prompt_type
    - extract_transaction: If True, extract structured transaction data using LLM
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
    
    # Skip cache for now (database not available)
    logger.info("Processing document without cache")
    
    # Try API keys in order
    result = None
    used_key_name = None
    
    for i, api_key in enumerate(config.API_KEYS, 1):
        key_name = f"api_key_{i}"
        
        # Make API call
        logger.info(f"Trying {key_name}")
        result = call_docstrange(api_key, image_bytes, final_prompt)
        
        if result:
            used_key_name = key_name
            logger.info(f"Successfully processed with {key_name}")
            break
        else:
            logger.warning(f"{key_name} failed to process document")
    
    if not result:
        raise HTTPException(
            status_code=503, 
            detail="All API keys failed"
        )
    
    # Extract transaction data if requested
    transaction_data = None
    if extract_transaction and 'content' in result:
        logger.info("Extracting transaction data from OCR content")
        transaction_result = extract_transaction_data(result['content'])
        if transaction_result['success']:
            transaction_data = transaction_result['data']
            logger.info("Transaction data extracted successfully")
        else:
            logger.warning(f"Failed to extract transaction data: {transaction_result['error']}")
    
    # Add transaction data to result if available
    if transaction_data:
        result['transaction_data'] = transaction_data
    
    return {
        "cached": False,
        "result": result,
        "used_key": used_key_name,
        "message": "Document processed successfully"
    }

# @app.delete("/cache/clear")
# async def clear_cache(db: Session = Depends(get_db)):
#     """Clear all cached documents."""
#     count = db.query(ProcessedDocument).count()
#     db.query(ProcessedDocument).delete()
#     db.commit()
#     return {"message": f"Cleared {count} cached documents"}

# @app.get("/cache/stats")
# async def cache_stats(db: Session = Depends(get_db)):
#     """Get cache statistics."""
#     total_docs = db.query(ProcessedDocument).count()
#     return {
#         "total_cached_documents": total_docs,
#         "cache_size": f"{total_docs} documents"
#     }

@app.post("/extract-transaction")
async def extract_transaction_from_ocr(
    file: UploadFile
):
    """
    Extract structured transaction data from document image.
    This endpoint processes the image with OCR and then extracts transaction data.
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
    
    # Process with OCR first
    result = None
    used_key_name = None
    
    for i, api_key in enumerate(config.API_KEYS, 1):
        key_name = f"api_key_{i}"
        
        # Make API call
        logger.info(f"Trying {key_name} for OCR")
        result = call_docstrange(api_key, image_bytes, "Extract all text and tables from this document")
        
        if result:
            used_key_name = key_name
            logger.info(f"OCR completed with {key_name}")
            break
        else:
            logger.warning(f"{key_name} failed OCR processing")
    
    if not result or 'content' not in result:
        raise HTTPException(
            status_code=503, 
            detail="OCR processing failed with all API keys"
        )
    
    # Extract transaction data
    logger.info("Extracting transaction data from OCR content")
    logger.info(f"OCR result structure: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
    logger.info(f"Looking for 'content' field in result: {'content' in result if isinstance(result, dict) else 'Result is not a dict'}")
    
    # Check if 'content' field exists, otherwise try other common field names
    ocr_content = None
    if isinstance(result, dict):
        if 'content' in result:
            ocr_content = result['content']
        elif 'text' in result:
            ocr_content = result['text']
        elif 'data' in result:
            ocr_content = result['data']
        else:
            # If no standard field, use the whole result as string
            ocr_content = str(result)
    
    if not ocr_content:
        raise HTTPException(
            status_code=500,
            detail="No OCR content found in API response"
        )
    
    logger.info(f"OCR content preview: {str(ocr_content)[:200]}...")
    transaction_result = extract_transaction_data(ocr_content)
    
    if not transaction_result['success']:
        raise HTTPException(
            status_code=500,
            detail=f"Transaction extraction failed: {transaction_result['error']}"
        )
    
    return {
        "success": True,
        "transaction_data": transaction_result['data'],
        "ocr_content": result['content'],
        "used_key": used_key_name,
        "cached": False,
        "message": "Transaction data extracted successfully"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5006)