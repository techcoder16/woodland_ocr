import hashlib
import os
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
import requests
from fastapi import FastAPI, UploadFile, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from pdf2image import convert_from_bytes
from PIL import Image
import io
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

def convert_pdf_to_images(pdf_bytes: bytes) -> List[bytes]:
    """Convert PDF to images for better OCR processing."""
    try:
        logger.info("Converting PDF to images...")
        
        # Convert PDF to images
        images = convert_from_bytes(
            pdf_bytes,
            dpi=300,  # High DPI for better OCR quality
            first_page=1,
            last_page=None,  # Convert all pages
            fmt='PNG'
        )
        
        image_bytes_list = []
        for i, image in enumerate(images):
            # Convert PIL Image to bytes
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG', optimize=True)
            img_bytes = img_buffer.getvalue()
            image_bytes_list.append(img_bytes)
            
            logger.info(f"Converted page {i+1} to image ({len(img_bytes)} bytes)")
        
        logger.info(f"Successfully converted PDF to {len(image_bytes_list)} images")
        return image_bytes_list
        
    except Exception as e:
        error_msg = str(e)
        if "poppler" in error_msg.lower() or "page count" in error_msg.lower():
            logger.error(f"Poppler not installed or not in PATH: {error_msg}")
            raise HTTPException(
                status_code=500, 
                detail="PDF conversion requires poppler-utils to be installed. Please install it or use image files instead."
            )
        else:
            logger.error(f"Error converting PDF to images: {error_msg}")
            raise HTTPException(status_code=500, detail=f"PDF conversion failed: {error_msg}")

def process_file_with_images(file_bytes: bytes, filename: str, api_key: str) -> Dict[str, Any]:
    """Process file, converting PDF to images if needed."""
    file_extension = filename.lower().split('.')[-1]
    
    if file_extension == 'pdf':
        try:
            logger.info("PDF detected, converting to images first...")
            images = convert_pdf_to_images(file_bytes)
            
            # Process each image with OCR
            all_results = []
            for i, image_bytes in enumerate(images):
                logger.info(f"Processing image {i+1}/{len(images)}...")
                result = call_docstrange(api_key, image_bytes, "Extract all text and data from this document")
                if result:
                    result['page_number'] = i + 1
                    all_results.append(result)
            
            # Combine results from all pages
            if all_results:
                combined_result = {
                    'success': True,
                    'content': '',
                    'format': 'combined',
                    'file_type': 'pdf',
                    'pages_processed': len(all_results),
                    'processing_time': sum(r.get('processing_time', 0) for r in all_results),
                    'record_id': all_results[0].get('record_id'),
                    'processing_status': 'completed',
                    'multiple_outputs': True
                }
                
                # Combine content from all pages
                combined_content = []
                for result in all_results:
                    if result.get('content'):
                        combined_content.append(f"--- Page {result.get('page_number', 'Unknown')} ---\n{result['content']}")
                
                combined_result['content'] = '\n\n'.join(combined_content)
                return combined_result
            else:
                return {'success': False, 'error': 'No pages could be processed'}
                
        except HTTPException as e:
            if "poppler" in str(e.detail).lower():
                logger.warning("PDF to image conversion failed (poppler not available), trying direct PDF processing...")
                # Fallback: try to process PDF directly
                result = call_docstrange(api_key, file_bytes, "Extract all text and data from this PDF document")
                if result:
                    result['file_type'] = 'pdf'
                    result['pages_processed'] = 1
                    result['processing_note'] = 'Processed directly (no image conversion)'
                return result
            else:
                raise e
    
    else:
        # For non-PDF files, process directly
        logger.info(f"Processing {file_extension} file directly...")
        return call_docstrange(api_key, file_bytes, "Extract all text and data from this document")

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
        print(response.json())
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
    return f"""You must return ONLY a valid JSON object. Do not include any code, explanations, or markdown.

Extract data from this invoice and return ONLY this JSON format:

{{
  "toLandlordDate": "YYYY-MM-DD or null",
  "toLandLordMode": "string or null", 
  "toLandlordRentReceived": "number or null",
  "toLandlordLessManagementFees": "number or null",
  "toLandlordLessBuildingExpenditure": "number or null",
  "toLandlordLessBuildingExpenditureActual": "number or null",
  "toLandlordLessBuildingExpenditureDifference": "number or null",
  "toLandlordNetPaid": "number or null",
  "toLandlordLessVAT": "number or null",
  "toLandlordChequeNo": "string or null",
  "toLandlordExpenditureDescription": "string or null",
  "toLandlordPaidBy": "string or null",
  "toLandlordDefaultExpenditure": "string or null",
  "toLandlordNetReceived": "number or null"
}}

Map fields: statement_date->toLandlordDate, total/amount->toLandlordRentReceived, tax->toLandlordLessVAT, bill_to_name->toLandlordPaidBy

Invoice data: {ocr_content}

Return ONLY the JSON object above, nothing else."""

def extract_transaction_data(ocr_content):
    """Extract transaction data using Groq free API with multiple model fallback."""
    import time
    from llm_config import GROQ_TOKEN, RATE_LIMIT_DELAY, API_TIMEOUT, GROQ_MODELS

    time.sleep(RATE_LIMIT_DELAY)  # Rate limiting

    # Create the extraction prompt
    prompt = create_transaction_extraction_prompt(ocr_content)

    # Groq API configuration
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Try multiple models
    for model in GROQ_MODELS:
        try:
            logger.info(f"Trying Groq API with model: {model}")
            
            data = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 1000
            }
            
            resp = requests.post(url, headers=headers, json=data, timeout=API_TIMEOUT)
            
            if resp.status_code == 200:
                logger.info(f"Groq API responded successfully with {model}")
                result = parse_groq_response(resp.json(), ocr_content)
                if result.get('success') and not result.get('fallback'):
                    return result
                else:
                    logger.warning(f"Model {model} returned fallback, trying next model")
                    continue
            else:
                logger.warning(f"Model {model} returned status {resp.status_code}: {resp.text[:200]}")
                continue
                
        except Exception as e:
            logger.warning(f"Model {model} failed: {str(e)}, trying next model")
            continue
    
    # If all models failed, use fallback
    logger.warning("All Groq models failed, using fallback extraction")
    return extract_basic_transaction_data(ocr_content)

def parse_groq_response(response_data, ocr_content):
    """Parse Groq API response and extract transaction data."""
    import json
    import re
    
    try:
        # Extract text from Groq response
        response_text = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        if not response_text.strip():
            logger.warning("Groq returned empty response, using fallback")
            return extract_basic_transaction_data(ocr_content)
        
        logger.info(f"Groq response preview: {response_text[:200]}...")
        
        # Try multiple JSON extraction patterns
        json_patterns = [
            r'\{[^{}]*"toLandlordDate"[^{}]*\}',  # Look for our specific fields
            r'\{.*?"toLandlordDate".*?\}',  # More flexible pattern
            r'```json\s*(\{.*?\})\s*```',  # JSON in code blocks
            r'```\s*(\{.*?\})\s*```',  # Any code block with JSON
            r'\{.*\}',  # Any JSON object
        ]
        
        transaction_data = None
        for pattern in json_patterns:
            json_match = re.search(pattern, response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
                try:
                    transaction_data = json.loads(json_text)
                    logger.info(f"Successfully parsed JSON with pattern: {pattern}")
                    break
                except json.JSONDecodeError:
                    continue
        
        if not transaction_data:
            logger.warning("Could not extract valid JSON from Groq response, using fallback")
            return extract_basic_transaction_data(ocr_content)
        
        # Validate the response format
        required_fields = [
            "toLandlordDate", "toLandLordMode", "toLandlordRentReceived",
            "toLandlordLessManagementFees", "toLandlordLessBuildingExpenditure",
            "toLandlordLessBuildingExpenditureActual", "toLandlordLessBuildingExpenditureDifference",
            "toLandlordNetPaid", "toLandlordLessVAT", "toLandlordChequeNo",
            "toLandlordExpenditureDescription", "toLandlordPaidBy",
            "toLandlordDefaultExpenditure", "toLandlordNetReceived"
        ]
        
        # Check if response has the correct field names
        has_correct_fields = all(field in transaction_data for field in required_fields)
        
        if has_correct_fields:
            logger.info("Groq API extracted data successfully")
            return {
                'success': True,
                'data': transaction_data,
                'raw_response': response_text,
                'api_used': 'Groq'
            }
        else:
            logger.warning(f"Groq returned wrong format. Fields found: {list(transaction_data.keys())}, using fallback")
            return extract_basic_transaction_data(ocr_content)
            
    except Exception as e:
        logger.warning(f"Error parsing Groq response: {str(e)}, using fallback")
        return extract_basic_transaction_data(ocr_content)

def extract_basic_transaction_data(ocr_content):
    """Fallback: Extract basic transaction data using simple pattern matching."""
    import re
    import json
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
        logger.info("Using fallback extraction method")
        
        # Try to parse as JSON first (if OCR returned structured data)
        try:
            # Clean up OCR content - remove page markers and extract JSON
            cleaned_content = ocr_content.strip()
            
            # Remove page markers like "--- Page 1 ---\n"
            if cleaned_content.startswith('--- Page'):
                lines = cleaned_content.split('\n')
                # Find the first line that starts with {
                for line in lines:
                    if line.strip().startswith('{'):
                        cleaned_content = line.strip()
                        break
            
            if cleaned_content.startswith('{'):
                data = json.loads(cleaned_content)
                
                # Extract from structured JSON
                # Try different date field names
                date_fields = ['invoice_date', 'date', 'invoiceDate', 'Date', 'InvoiceDate', 'invoice_date', 'INVOICE_DATE', 'statement_date', 'StatementDate', 'STATEMENT_DATE']
                for field in date_fields:
                    if field in data:
                        date_str = data[field]
                        if '/' in date_str:
                            parts = date_str.split('/')
                            if len(parts) == 3:
                                month, day, year = parts
                                result["toLandlordDate"] = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                            else:
                                result["toLandlordDate"] = date_str
                        else:
                            result["toLandlordDate"] = date_str
                        break
                
                # Try different total field names
                total_fields = [
                    'total', 'amount', 'total_amount', 'grand_total', 'sum', 
                    'Total', 'Amount', 'TOTAL', 'AMOUNT', 'TotalAmount', 'GrandTotal', 'Subtotal',
                    'item_amount_exclusive_of_vat_1', 'total_amount_exclusive_of_vat',
                    'item_amount_1', 'amount_1', 'total_exclusive_of_vat'
                ]
                for field in total_fields:
                    if field in data and data[field] is not None:
                        result["toLandlordRentReceived"] = float(data[field])
                        break
                
                # Set net received as the same as rent received
                if result["toLandlordRentReceived"]:
                    result["toLandlordNetReceived"] = result["toLandlordRentReceived"]
                
                # Try different tax field names
                tax_fields = [
                    'sales_tax', 'tax', 'vat', 'VAT', 'tax_amount', 'vat_total',
                    'item_vat_net_1', 'vat_net', 'VAT_net', 'total_vat'
                ]
                for field in tax_fields:
                    if field in data and data[field] is not None:
                        result["toLandlordLessVAT"] = float(data[field])
                        break
                
                # Try different name field names
                name_fields = [
                    'bill_to_name', 'customer_name', 'client_name', 'name', 'billToName', 
                    'CustomerName', 'customerName', 'CUSTOMER_NAME', 'from_name', 'FromName',
                    'company_name', 'CompanyName', 'vendor_name', 'VendorName'
                ]
                for field in name_fields:
                    if field in data and data[field]:
                        result["toLandlordPaidBy"] = data[field]
                        break
                
                # Combine item descriptions - try different patterns
                descriptions = []
                
                # Pattern 1: Item1Description, Item2Description, etc. (your format)
                for i in range(1, 10):
                    item_key = f'Item{i}Description'
                    if item_key in data and data[item_key]:
                        descriptions.append(data[item_key])
                
                # Pattern 1.5: item_description_1, item_description_2, etc.
                for i in range(1, 10):
                    item_key = f'item_description_{i}'
                    if item_key in data and data[item_key]:
                        descriptions.append(data[item_key])
                
                # Pattern 2: item_1_description, item_2_description, etc.
                if not descriptions:
                    for i in range(1, 10):
                        item_key = f'item_{i}_description'
                        if item_key in data and data[item_key]:
                            descriptions.append(data[item_key])
                
                # Pattern 3: description, description_1, description_2, etc.
                if not descriptions:
                    desc_fields = ['description', 'item_description', 'service_description']
                    for field in desc_fields:
                        if field in data and data[field]:
                            descriptions.append(data[field])
                    
                    for i in range(1, 10):
                        desc_key = f'description_{i}'
                        if desc_key in data and data[desc_key]:
                            descriptions.append(data[desc_key])
                
                if descriptions:
                    result["toLandlordExpenditureDescription"] = "; ".join(descriptions)
                
                # Set payment mode based on terms
                terms_fields = ['terms', 'payment_terms', 'PaymentTerms', 'PAYMENT_TERMS']
                for field in terms_fields:
                    if field in data:
                        terms = data[field].lower()
                        if 'net' in terms:
                            result["toLandLordMode"] = "net payment"
                        elif 'cash' in terms:
                            result["toLandLordMode"] = "cash"
                        elif 'cheque' in terms or 'check' in terms:
                            result["toLandLordMode"] = "cheque"
                        elif 'bank' in terms:
                            result["toLandLordMode"] = "bank transfer"
                        else:
                            result["toLandLordMode"] = "unknown"
                        break
                
                # If no terms found, set default
                if not result["toLandLordMode"]:
                    result["toLandLordMode"] = "unknown"
                
                # Extract vendor information for additional context
                vendor_fields = ['vendor_name', 'VendorName', 'supplier_name', 'company_name']
                for field in vendor_fields:
                    if field in data and data[field]:
                        # Add vendor info to description if not already present
                        if not result["toLandlordExpenditureDescription"]:
                            result["toLandlordExpenditureDescription"] = f"Vendor: {data[field]}"
                        break
                
                # Try to extract invoice number as cheque number if available
                invoice_fields = ['invoice_number', 'InvoiceNumber', 'invoice_no', 'InvoiceNo']
                for field in invoice_fields:
                    if field in data and data[field]:
                        result["toLandlordChequeNo"] = str(data[field])
                        break
                
                logger.info("Extracted data from structured JSON")
                return {
                    'success': True,
                    'data': result,
                    'raw_response': 'JSON extraction fallback used',
                    'fallback': True
                }
        except json.JSONDecodeError:
            pass  # Continue with regex extraction
        
        # Fallback to regex extraction for non-JSON content
        logger.info("Using regex extraction for non-JSON content")
        
        # Extract monetary amounts
        money_patterns = [
            r'[£$€]\s*(\d+(?:\.\d{2})?)',  # £50.00
            r'(\d+(?:\.\d{2})?)\s*[£$€]',  # 50.00£
            r'Amount[^£]*[£$€]\s*(\d+(?:\.\d{2})?)',  # Amount exclusive of VAT £50.00
        ]
        
        amounts = []
        for pattern in money_patterns:
            matches = re.findall(pattern, ocr_content, re.IGNORECASE)
            amounts.extend(matches)
        
        if amounts:
            amounts_float = [float(amount) for amount in amounts]
            result["toLandlordRentReceived"] = max(amounts_float)
            logger.info(f"Extracted amounts: {amounts_float}, using {max(amounts_float)} as rent received")
        
        # Extract dates
        date_patterns = [
            r'Invoice\s+Date\s+(\d+)',  # Invoice Date 062
            r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})',  # DD/MM/YYYY or DD-MM-YYYY
            r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})',  # YYYY/MM/DD or YYYY-MM-DD
        ]
        
        for pattern in date_patterns:
            dates = re.findall(pattern, ocr_content, re.IGNORECASE)
            if dates:
                try:
                    if len(dates[0]) == 1:  # Invoice Date 062 format
                        day = dates[0]
                        current_date = datetime.now()
                        result["toLandlordDate"] = f"{current_date.year}-{current_date.month:02d}-{day.zfill(2)}"
                        break
                    elif len(dates[0]) == 3:
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
                except Exception as e:
                    logger.warning(f"Date extraction failed: {e}")
                    continue
        
        # Extract description
        lines = ocr_content.split('\n')
        description_keywords = ['descaling', 'heating', 'hot water', 'maintenance', 'repair', 'problem', 'working']
        
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in description_keywords):
                description = line.strip()
                description = re.sub(r'<[^>]+>', ' ', description)
                description = re.sub(r'\s+', ' ', description).strip()
                result["toLandlordExpenditureDescription"] = description
                break
        
        # Set payment mode
        if 'cheque' in ocr_content.lower():
            result["toLandLordMode"] = "cheque"
        elif 'bank' in ocr_content.lower():
            result["toLandLordMode"] = "bank transfer"
        elif 'cash' in ocr_content.lower():
            result["toLandLordMode"] = "cash"
        else:
            result["toLandLordMode"] = "unknown"
        
        # Set net received
        if result["toLandlordRentReceived"]:
            result["toLandlordNetReceived"] = result["toLandlordRentReceived"]
        
        logger.info(f"Fallback extraction completed. Extracted data: {[k for k, v in result.items() if v is not None]}")
        
        return {
            'success': True,
            'data': result,
            'raw_response': 'Regex extraction fallback used',
            'fallback': True
        }
        
    except Exception as e:
        logger.error(f"Basic extraction failed: {str(e)}")
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
    - file: Document file to process (PDF, PNG, JPG, etc.)
    - prompt_type: One of the predefined prompt types (see /prompts endpoint)
    - custom_prompt: Custom prompt to override prompt_type
    - extract_transaction: If True, extract structured transaction data using LLM
    """
    # Validate API keys
    if not config.API_KEYS:
        raise HTTPException(status_code=500, detail="No API keys configured")
    
    # Read file
    try:
        file_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")
    
    img_hash = hash_image(file_bytes)
    
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
        
        # Process file (with PDF to image conversion if needed)
        logger.info(f"Trying {key_name}")
        result = process_file_with_images(file_bytes, file.filename, api_key)
        
        if result and result.get('success'):
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
    Extract structured transaction data from document.
    This endpoint processes the document (PDF, PNG, JPG, etc.) with OCR and then extracts transaction data.
    """
    # Validate API keys
    if not config.API_KEYS:
        raise HTTPException(status_code=500, detail="No API keys configured")
    
    # Read file
    try:
        file_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")
    
    img_hash = hash_image(file_bytes)
    
    # Process with OCR first (with PDF to image conversion if needed)
    result = None
    used_key_name = None
    
    for i, api_key in enumerate(config.API_KEYS, 1):
        key_name = f"api_key_{i}"
        
        # Process file (with PDF to image conversion if needed)
        logger.info(f"Trying {key_name} for OCR")
        result = process_file_with_images(file_bytes, file.filename, api_key)
        
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