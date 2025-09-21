

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

import json
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load a small free model (Phi-2 here)
model_name = "microsoft/phi-2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cpu",  # CPU only
    torch_dtype="auto"
)

def run_llm_on_ocr(ocr_json: dict, prompt: str) -> str:
    """Run Hugging Face model on OCR JSON with a prompt."""
    input_text = f"""
    OCR JSON:
    {json.dumps(ocr_json, indent=2)}

    Task:
    {prompt}
    """

    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.3,
        top_p=0.9
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


ocr_result =   {'success': True, 'content': 'Invoice Date 062\n\nFrom Dreams Creation Ltd\nBanking 10900\nVAT Reg\'d No.\n\nTo\n18 MAID Road\n\n<table>\n<thead>\n<tr>\n<td></td>\n<td></td>\n<td>Amount<br>exclusive of<br>VAT</td>\n<td>VAT<br>NET</td>\n</tr>\n</thead>\n<tbody>\n<tr>\n<td></td>\n<td>Hot water not working and heating problem Descaling</td>\n<td>£50.00</td>\n<td>£</td>\n</tr>\n<tr>\n<td colspan="3"></td>\n<td>£50.00</td>\n<td></td>\n</tr>\n</tbody>\n</table>\n\nVAT\nTOTAL', 'format': 'markdown', 'file_type': 'unknown -> image', 'pages_processed': 1, 'processing_time': 6.311602354049683, 'record_id': 99654, 'processing_status': 'completed'}


if ocr_result:
    structured_data = run_llm_on_ocr(
        ocr_result,
        "Extract only the invoice number, total amount, and date in JSON format."
    )
    print(structured_data)
