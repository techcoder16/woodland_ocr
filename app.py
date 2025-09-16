
# from transformers import TrOCRProcessor, VisionEncoderDecoderModel
# from PIL import Image
# import requests

# # load image from the IAM database
# url = 'abc.png'
# image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

# processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
# model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
# pixel_values = processor(images=image, return_tensors="pt").pixel_values

# generated_ids = model.generate(pixel_values)
# generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]



# from paddleocr import PaddleOCR
# from transformers import AutoTokenizer, AutoModelForCausalLM

# # Step 1: OCR
# ocr = PaddleOCR(use_angle_cls=True, lang='en')
# results = ocr.predict("abc.png")

# raw_text = "\n".join([line[1][0] for line in results[0]])

# # Step 2: LLM Cleanup
# model_name = "HuggingFaceTB/SmolLM-135M-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)
# prompt = f"Clean and structure this OCR text properly:\n\n{raw_text}"
# inputs = tokenizer(prompt, return_tensors="pt")
# outputs = model.generate(**inputs, max_new_tokens=500)
# print(outputs)
# clean_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(clean_text)
import torch
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

app = FastAPI()

model_name = "nanonets/Nanonets-OCR-s"

# Load model + components
model = VisionEncoderDecoderModel.from_pretrained(model_name).to("cpu")
image_processor = AutoImageProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

@app.post("/ocr/")
async def ocr_endpoint(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Preprocess
    pixel_values = image_processor(image, return_tensors="pt").pixel_values

    # Run inference
    output_ids = model.generate(pixel_values)
    text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

    return {"ocr_text": text}
