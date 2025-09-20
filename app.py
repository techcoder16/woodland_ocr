from fastapi import FastAPI, UploadFile, File
from docstrange import DocumentExtractor
import shutil
import os

app = FastAPI()

# Load extractor once (cached in container)
extractor = DocumentExtractor(cpu=True)

@app.post("/extract")
async def extract_text(file: UploadFile = File(...)):
    temp_path = f"/tmp/{file.filename}"
    
    # Save uploaded file temporarily
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Run OCR
    result = extractor.extract(temp_path)
    text = result.extract_text()

    # Clean up
    os.remove(temp_path)

    return {"filename": file.filename, "text": text}
