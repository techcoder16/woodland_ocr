import os
import json
from datetime import datetime
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
import torch
import uvicorn
from typing import Optional, Dict, Any
import re
import time

# Disable FlashAttention
os.environ["FLASH_ATTENTION_FORCE_DISABLE"] = "1"

# Model path
MODEL_PATH = "/app/hf_cache/models--nanonets--Nanonets-OCR-s/snapshots/3baad182cc87c65a1861f0c30357d3467e978172"

def load_model_with_retry(max_retries=3, retry_delay=5):
    """Load model with retry logic and better error handling"""
    # Try loading with different configurations
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        attn_implementation="eager",
        trust_remote_code=True,
        local_files_only=False,
        resume_download=True,
    )
    
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    processor = AutoProcessor.from_pretrained(MODEL_PATH)


    print("Model loaded successfully!")
    return model, tokenizer, processor

# Load model with retry mechanism
try:
    model, tokenizer, processor = load_model_with_retry()
    model.eval()
    MODEL_LOADED = True
    print("Model initialization completed successfully!")
except Exception as e:
    print(f"Failed to load model: {e}")
    MODEL_LOADED = False
    model, tokenizer, processor = None, None, None

# Init FastAPI
app = FastAPI(title="Enhanced OCR Transaction API", version="2.0")

def create_transaction_extraction_prompt():
    """Creates a specialized prompt for extracting transaction data from invoices/receipts"""
    return """You are an expert financial document processor. Extract all transaction-related information from the above document and return it in a structured JSON format.

**EXTRACTION REQUIREMENTS:**

1. **TRANSACTION IDENTIFICATION:**
   - Transaction ID/Invoice Number/Receipt Number
   - Transaction Date (extract and format as YYYY-MM-DD)
   - Document Type (invoice, receipt, payment slip, etc.)

2. **TENANT/CUSTOMER INFORMATION:**
   - Tenant/Customer Name
   - Payment Date
   - Payment Method/Mode (cash, cheque, bank transfer, online, etc.)
   - Rent Amount/Base Amount
   - Additional Benefits/Charges (if any)
   - Other Debits/Charges
   - Total Amount Received from Tenant
   - Description of Payment/Services
   - Received By (person/department name)
   - Any private notes or remarks

3. **LANDLORD/SUPPLIER INFORMATION:**
   - Payment to Landlord/Supplier Date
   - Payment Mode to Landlord
   - Rent/Amount to be Paid
   - Management Fees (if deducted)
   - Building Expenditure (budgeted)
   - Building Expenditure (actual)
   - Building Expenditure Difference
   - Net Amount Paid
   - VAT Amount (if applicable)
   - Cheque Number (if payment by cheque)
   - Expenditure Description
   - Paid By (person/department)
   - Default Expenditure Category
   - Net Amount Received

4. **FINANCIAL CALCULATIONS:**
   - Perform any necessary calculations
   - Identify discrepancies between expected and actual amounts
   - Calculate net amounts after deductions

**OUTPUT FORMAT:**
Return ONLY a valid JSON object with the following structure:

{
  "tranid": "extracted_transaction_id_or_generate_number",
  "document_type": "invoice/receipt/payment_slip",
  "extraction_confidence": "high/medium/low",
  
  "fromTenant": {
    "date": "YYYY-MM-DD or null",
    "mode": "payment_method or null",
    "hBenefit1": "number or null",
    "hBenefit2": "number or null", 
    "rentReceived": "number or null",
    "otherDebit": "number or null",
    "description": "description or null",
    "receivedBy": "person_name or null",
    "privateNote": "any_notes or null"
  },
  
  "toLandlord": {
    "date": "YYYY-MM-DD or null",
    "mode": "payment_method or null",
    "rentReceived": "number or null",
    "lessManagementFees": "number or null",
    "lessBuildingExpenditure": "number or null",
    "lessBuildingExpenditureActual": "number or null",
    "lessBuildingExpenditureDifference": "number or null",
    "netPaid": "number or null",
    "lessVAT": "number or null",
    "chequeNo": "cheque_number or null",
    "expenditureDescription": "description or null",
    "paidBy": "person_name or null",
    "defaultExpenditure": "category or null",
    "netReceived": "number or null"
  },
  
  "rawText": "complete_extracted_text_for_reference",
  "extractedAmounts": ["list of all numerical amounts found"],
  "extractedDates": ["list of all dates found"],
  "potentialIssues": ["list any unclear or missing information"]
}

**IMPORTANT INSTRUCTIONS:**
- Extract ALL numerical values, even if you're unsure of their purpose
- Convert all dates to YYYY-MM-DD format
- If information is unclear or missing, set to null
- Include the complete raw text for verification
- Flag any potential issues or ambiguities
- Ensure all numbers are properly parsed (remove currency symbols, commas)
- If multiple transactions exist in one document, focus on the primary transaction

EXTRACT AND RETURN ONLY THE JSON OBJECT ABOVE."""

def ocr_transaction_extraction(image_path: str, property_id: str = None, max_new_tokens: int = 4096):
    """Enhanced OCR function specifically for transaction data extraction"""
    
    if not MODEL_LOADED:
        raise Exception("Model not loaded. Please check model loading errors.")
    
    prompt = create_transaction_extraction_prompt()
    
    image = Image.open(image_path)
    messages = [
        {"role": "system", "content": "You are a specialized financial document processor focused on extracting transaction data accurately."},
        {"role": "user", "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text", "text": prompt},
        ]},
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    return output_text[0]

def parse_and_validate_transaction_data(raw_output: str, property_id: str = None) -> Dict[str, Any]:
    """Parse the OCR output and validate transaction data"""
    
    try:
        # Try to extract JSON from the response
        json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            extracted_data = json.loads(json_str)
        else:
            # If no JSON found, create a basic structure with raw text
            extracted_data = {
                "tranid": None,
                "document_type": "unknown",
                "extraction_confidence": "low",
                "fromTenant": {},
                "toLandlord": {},
                "rawText": raw_output,
                "extractedAmounts": [],
                "extractedDates": [],
                "potentialIssues": ["Failed to parse structured data"]
            }
    except json.JSONDecodeError:
        # Fallback structure
        extracted_data = {
            "tranid": None,
            "document_type": "unknown", 
            "extraction_confidence": "low",
            "fromTenant": {},
            "toLandlord": {},
            "rawText": raw_output,
            "extractedAmounts": [],
            "extractedDates": [],
            "potentialIssues": ["JSON parsing failed"]
        }
    
    # Prepare transaction draft data for your database
    transaction_draft = {
        "propertyId": property_id,
        "tranid": extracted_data.get("tranid"),
        
        # From Tenant fields
        "fromTenantDate": parse_date(extracted_data.get("fromTenant", {}).get("date")),
        "fromTenantMode": extracted_data.get("fromTenant", {}).get("mode"),
        "fromTenantHBenefit1": parse_float(extracted_data.get("fromTenant", {}).get("hBenefit1")),
        "fromTenantHBenefit2": parse_float(extracted_data.get("fromTenant", {}).get("hBenefit2")),
        "fromTenantRentReceived": parse_float(extracted_data.get("fromTenant", {}).get("rentReceived")),
        "fromTenantOtherDebit": parse_float(extracted_data.get("fromTenant", {}).get("otherDebit")),
        "fromTenantDescription": extracted_data.get("fromTenant", {}).get("description"),
        "fromTenantReceivedBy": extracted_data.get("fromTenant", {}).get("receivedBy"),
        "fromTenantPrivateNote": extracted_data.get("fromTenant", {}).get("privateNote"),
        
        # To Landlord fields
        "toLandlordDate": parse_date(extracted_data.get("toLandlord", {}).get("date")),
        "toLandLordMode": extracted_data.get("toLandlord", {}).get("mode"),
        "toLandlordRentReceived": parse_float(extracted_data.get("toLandlord", {}).get("rentReceived")),
        "toLandlordLessManagementFees": parse_float(extracted_data.get("toLandlord", {}).get("lessManagementFees")),
        "toLandlordLessBuildingExpenditure": parse_float(extracted_data.get("toLandlord", {}).get("lessBuildingExpenditure")),
        "toLandlordLessBuildingExpenditureActual": parse_float(extracted_data.get("toLandlord", {}).get("lessBuildingExpenditureActual")),
        "toLandlordLessBuildingExpenditureDifference": parse_float(extracted_data.get("toLandlord", {}).get("lessBuildingExpenditureDifference")),
        "toLandlordNetPaid": parse_float(extracted_data.get("toLandlord", {}).get("netPaid")),
        "toLandlordLessVAT": parse_float(extracted_data.get("toLandlord", {}).get("lessVAT")),
        "toLandlordChequeNo": extracted_data.get("toLandlord", {}).get("chequeNo"),
        "toLandlordExpenditureDescription": extracted_data.get("toLandlord", {}).get("expenditureDescription"),
        "toLandlordPaidBy": extracted_data.get("toLandlord", {}).get("paidBy"),
        "toLandlordDefaultExpenditure": extracted_data.get("toLandlord", {}).get("defaultExpenditure"),
        "toLandlordNetReceived": parse_float(extracted_data.get("toLandlord", {}).get("netReceived")),
        
        # Status and metadata
        "status": "DRAFT",
        "createdAt": datetime.now().isoformat(),
        "updatedAt": datetime.now().isoformat(),
        
        # Additional extraction metadata
        "extractionMetadata": {
            "confidence": extracted_data.get("extraction_confidence"),
            "documentType": extracted_data.get("document_type"),
            "rawText": extracted_data.get("rawText"),
            "extractedAmounts": extracted_data.get("extractedAmounts", []),
            "extractedDates": extracted_data.get("extractedDates", []),
            "potentialIssues": extracted_data.get("potentialIssues", [])
        }
    }
    
    return {
        "transactionDraft": transaction_draft,
        "extractedData": extracted_data,
        "rawOcrOutput": raw_output
    }

def parse_date(date_str):
    """Parse date string to proper format"""
    if not date_str or date_str == "null":
        return None
    try:
        # Try different date formats
        for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"]:
            try:
                return datetime.strptime(str(date_str), fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
        return None
    except:
        return None

def parse_float(value):
    """Parse float value safely"""
    if not value or value == "null":
        return None
    try:
        # Remove currency symbols and commas
        clean_value = str(value).replace(',', '').replace('$', '').replace('£', '').replace('€', '').strip()
        return float(clean_value)
    except (ValueError, TypeError):
        return None

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if MODEL_LOADED else "unhealthy",
        "model_loaded": MODEL_LOADED,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/ocr/transaction")
async def extract_transaction_data(
    file: UploadFile = File(...),
    property_id: Optional[str] = None
):
    """
    Enhanced endpoint for extracting transaction data from invoices/receipts
    """
    if not MODEL_LOADED:
        return JSONResponse(
            content={
                "success": False,
                "error": "Model not loaded. Service is not available.",
                "transactionDraft": None
            }, 
            status_code=503
        )
    
    try:
        # Save uploaded file
        file_path = f"/tmp/{file.filename}"
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Extract transaction data
        raw_ocr_output = ocr_transaction_extraction(file_path, property_id, max_new_tokens=15000)
        
        # Parse and validate the data
        processed_data = parse_and_validate_transaction_data(raw_ocr_output, property_id)
        
        # Clean up temp file
        os.remove(file_path)
        
        return JSONResponse(content={
            "success": True,
            "transactionDraft": processed_data["transactionDraft"],
            "extractionMetadata": processed_data["extractedData"],
            "confidence": processed_data["extractedData"].get("extraction_confidence", "medium"),
            "issues": processed_data["extractedData"].get("potentialIssues", [])
        })
        
    except Exception as e:
        return JSONResponse(
            content={
                "success": False,
                "error": str(e),
                "transactionDraft": None
            }, 
            status_code=500
        )

@app.post("/ocr")
async def extract_text(file: UploadFile = File(...)):
    """Original OCR endpoint for general text extraction"""
    if not MODEL_LOADED:
        return JSONResponse(
            content={"error": "Model not loaded. Service is not available."}, 
            status_code=503
        )
    
    try:
        file_path = f"/tmp/{file.filename}"
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        prompt = """Extract the text from the above document as if you were reading it naturally.
        Return the tables in json format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present,
        add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>.
        Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or 
        <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."""
        
        image = Image.open(file_path)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "image", "image": f"file://{file_path}"},
                {"type": "text", "text": prompt},
            ]},
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
        inputs = inputs.to(model.device)
        
        output_ids = model.generate(**inputs, max_new_tokens=15000, do_sample=False)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        result = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        os.remove(file_path)
        return JSONResponse(content={"text": result[0]})
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
async def root():
    return {
        "message": "Enhanced OCR Transaction API", 
        "version": "2.0",
        "model_loaded": MODEL_LOADED,
        "model_path": MODEL_PATH
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5006, reload=False)