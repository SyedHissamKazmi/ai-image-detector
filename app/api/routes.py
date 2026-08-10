# app/api/routes.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image, UnidentifiedImageError
import os
import shutil
from app.services.analyzer import ImageAnalyzer
from app.api.schemas import AnalysisResponse

router = APIRouter()
analyzer = ImageAnalyzer()

UPLOAD_DIR = "uploads"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB limit (example)

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(file: UploadFile = File(...)):
    # 1. Validate content type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are allowed.")

    # 2. Read file content and check size (optional)
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large (max 10 MB).")

    # 3. Save to disk (temporarily)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(contents)

    # 4. Validate it's a real image using Pillow
    try:
        img = Image.open(file_path)
        img.verify()  # check file integrity, but closes the file
        # After verify(), we must reopen because verify() closes the file handle
        img = Image.open(file_path)
    except (UnidentifiedImageError, Exception) as e:
        # Remove the invalid file
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=400, detail=f"Invalid or corrupted image: {str(e)}")

    # 5. Run analysis
    result = analyzer.analyze(img, file.filename, file_path)

    # 6. (Optional) Clean up uploaded file – keep it for now
    # os.remove(file_path)   # uncomment later if you don't need the files stored

    return result