#!/usr/bin/env python3
"""
Simple test for transaction extraction without database dependencies
"""

import requests
import json

def create_transaction_extraction_prompt(ocr_content):
    """Create a prompt for extracting transaction data from OCR content."""
    return f"""
You are a financial data extraction specialist. Extract transaction data from the following OCR content and return it as a JSON object matching the Transaction model structure.

Focus on extracting "toLandlord" fields from invoice/receipt data. Here's the expected JSON structure:

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

Extraction Guidelines:
1. Look for dates in various formats (DD/MM/YYYY, MM/DD/YYYY, YYYY-MM-DD, etc.)
2. Extract monetary amounts (look for £, $, € symbols and decimal numbers)
3. Identify VAT amounts and calculations
4. Find payment methods (cheque, bank transfer, cash, etc.)
5. Extract descriptions of services/expenditures
6. Look for cheque numbers or reference numbers
7. Identify who made the payment
8. Calculate net amounts when possible

OCR Content to analyze:
{ocr_content}

Return ONLY the JSON object, no additional text or explanations.
"""

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
    
    try:
        resp = requests.post(url, headers=headers, json=data)
        js = resp.json()
        
        if js.get('status') == 'success':
            # Try to parse the JSON response
            try:
                transaction_data = json.loads(js['response'])
                return {
                    'success': True,
                    'data': transaction_data,
                    'raw_response': js['response']
                }
            except json.JSONDecodeError:
                return {
                    'success': False,
                    'error': 'Failed to parse JSON response',
                    'raw_response': js['response']
                }
        else:
            return {
                'success': False,
                'error': js.get('error', 'Unknown error'),
                'status': js.get('status')
            }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def test_transaction_extraction():
    """Test the transaction extraction with LLM"""
    print("Testing transaction extraction...")
    
    # Sample OCR content from your invoice
    ocr_content = """Invoice Date 062

From Dreams Creation Ltd
Banking 10900
VAT Reg'd No.

To
18 MAID Road

<table>
<thead>
<tr>
<td></td>
<td></td>
<td>Amount<br>exclusive of<br>VAT</td>
<td>VAT<br>NET</td>
</tr>
</thead>
<tbody>
<tr>
<td></td>
<td>Hot water not working and heating problem Descaling</td>
<td>£50.00</td>
<td>£</td>
</tr>
<tr>
<td colspan="3"></td>
<td>£50.00</td>
<td></td>
</tr>
</tbody>
</table>

VAT
TOTAL"""
    
    result = extract_transaction_data(ocr_content)
    
    print(f"Success: {result['success']}")
    if result['success']:
        print("Extracted data:")
        for key, value in result['data'].items():
            if value is not None:
                print(f"  {key}: {value}")
    else:
        print(f"Error: {result['error']}")
        if 'raw_response' in result:
            print(f"Raw response: {result['raw_response']}")
    
    return result

if __name__ == "__main__":
    print("=" * 50)
    print("Transaction Data Extraction Test")
    print("=" * 50)
    
    # Test extraction
    result = test_transaction_extraction()
    
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    print(f"Extraction: {'✅ PASS' if result['success'] else '❌ FAIL'}")
    
    if result['success']:
        print("✅ Transaction data extracted successfully!")
    else:
        print(f"❌ Extraction failed: {result['error']}")
