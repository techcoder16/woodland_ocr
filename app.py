import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
import torch
import uvicorn

# Disable FlashAttention issues if needed
os.environ["FLASH_ATTENTION_FORCE_DISABLE"] = "1"

# Load model
MODEL_PATH = "nanonets/Nanonets-OCR-s"

print("Loading model...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="eager" 
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

print("Model loaded successfully!")

# FastAPI app
app = FastAPI()


def run_ocr(image_path: str, max_new_tokens: int = 4096):
    """Run OCR on a single page with NanoNets model"""
    prompt = """Extract the text from the above document as if you were reading it naturally. 
    Return the tables in html format. Return the equations in LaTeX representation. 
    If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; 
    otherwise, add the image caption inside <img></img>. 
    Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. 
    Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number>. 
    Prefer using ☐ and ☑ for check boxes.
    """

    image = Image.open(image_path)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
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


@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    """API endpoint for OCR"""
    try:
        file_path = f"/tmp/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())

        result = run_ocr(file_path, max_new_tokens=8000)
        return {"text": result}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5006)
