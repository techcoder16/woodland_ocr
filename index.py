

# from paddleocr import PaddleOCR  

# import cv2
# from PIL import Image
# import numpy as np
# import pytesseract

# def preprocess_for_ocr(input_path, output_path="processed.png", target_char_height=32):
#     # 1. Read image
#     img = cv2.imread(input_path)
#     if img is None:
#         raise ValueError(f"Cannot read image: {input_path}")

#     # Convert to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Rough binarization for contour detection
#     _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

#     # Find connected components (potential characters/words)
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     heights = []
#     for c in contours:
#         x, y, w, h = cv2.boundingRect(c)
#         if 5 < h < gray.shape[0] * 0.5:  # filter out noise & very large regions
#             heights.append(h)

#     # Compute average text height
#     avg_height = np.median(heights) if heights else 20  # fallback if no contours found

#     # Scaling factor
#     scale = target_char_height / avg_height
#     new_w = int(img.shape[1] * scale)
#     new_h = int(img.shape[0] * scale)

#     # Resize adaptively
#     resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

#     # --- Continue normal preprocessing ---
#     gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
#     _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#     # Remove noise
#     kernel = np.ones((1, 1), np.uint8)
#     cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

#     # Deskew
#     coords = np.column_stack(np.where(cleaned > 0))
#     angle = cv2.minAreaRect(coords)[-1]
#     if angle < -45:
#         angle = 90 + angle
#     elif angle > 45:
#         angle = angle - 90
#     (h, w) = cleaned.shape
#     M = cv2.getRotationMatrix2D((w // 2, h // 2), -angle, 1.0)
#     deskewed = cv2.warpAffine(cleaned, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

#     # Sharpen
#     blur = cv2.GaussianBlur(deskewed, (0, 0), 3)
#     sharpen = cv2.addWeighted(deskewed, 1.5, blur, -0.5, 0)

#     # Save
#     cv2.imwrite(output_path, sharpen)

#     return output_path

# ocr = PaddleOCR(
#     text_detection_model_name="PP-OCRv5_server_det",
#     text_recognition_model_name="PP-OCRv5_server_rec",
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=False,
#     use_textline_orientation=True,

# )



# processed_img = preprocess_for_ocr("./abc.png")

# result = ocr.predict(processed_img)
# for res in result:
#     res.print()
#     res.save_to_img("output")
#     res.save_to_json("output")

# custom_config = r'--oem 3 --psm 6'  # 3 = default LSTM, 6 = assume block of text

# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# tess_text =pytesseract.image_to_string(processed_img, config=custom_config, lang='eng')

# print(tess_text)


import requests



# Transaction data extraction prompt
def create_transaction_extraction_prompt(ocr_content):
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

# Sample OCR result (replace with actual OCR data from your API)
ocr_result = {
    'success': True, 
    'content': 'Invoice Date 062\n\nFrom Dreams Creation Ltd\nBanking 10900\nVAT Reg\'d No.\n\nTo\n18 MAID Road\n\n<table>\n<thead>\n<tr>\n<td></td>\n<td></td>\n<td>Amount<br>exclusive of<br>VAT</td>\n<td>VAT<br>NET</td>\n</tr>\n</thead>\n<tbody>\n<tr>\n<td></td>\n<td>Hot water not working and heating problem Descaling</td>\n<td>£50.00</td>\n<td>£</td>\n</tr>\n<tr>\n<td colspan="3"></td>\n<td>£50.00</td>\n<td></td>\n</tr>\n</tbody>\n</table>\n\nVAT\nTOTAL', 
    'format': 'markdown', 
    'file_type': 'unknown -> image', 
    'pages_processed': 1, 
    'processing_time': 6.311602354049683, 
    'record_id': 99654, 
    'processing_status': 'completed'
}

def extract_transaction_data(ocr_content):
    """Extract transaction data using LLM"""
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
            import json
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

# Example usage
if __name__ == "__main__":
    # Extract transaction data from OCR result
    result = extract_transaction_data(ocr_result['content'])
    
    if result['success']:
        print("Transaction data extracted successfully:")
        print(result['data'])
    else:
        print("Error extracting transaction data:")
        print(result['error'])
        if 'raw_response' in result:
            print("Raw LLM response:")
            print(result['raw_response'])
  