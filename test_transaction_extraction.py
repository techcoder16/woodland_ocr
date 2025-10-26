#!/usr/bin/env python3
"""
Test script for transaction data extraction
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import extract_transaction_data

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
