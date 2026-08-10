import os
import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from PIL import Image, ImageFile, UnidentifiedImageError

from app.api.schemas import AnalysisResponse
from app.core.config import settings
from app.core.logging_config import get_logger
from app.services.analyzer import ImageAnalyzer

router = APIRouter()
analyzer = ImageAnalyzer()
logger = get_logger("routes")

ImageFile.LOAD_TRUNCATED_IMAGES = False

def _safe_filename(filename: str | None) -> str:
    original = Path(filename or "uploaded_image").name
    if not original or original in {".", ".."}:
        original = "uploaded_image"
    return original

def _create_upload_path(filename: str) -> Path:
    safe_name = _safe_filename(filename)
    suffix = Path(safe_name).suffix.lower()
    if not suffix:
        suffix = ".img"
    return settings.UPLOAD_DIR / f"{uuid4().hex}{suffix}"

def _delete_file(file_path: str | Path) -> None:
    path = Path(file_path)
    try:
        path.unlink(missing_ok=True)
        logger.info("CLEANUP | deleted=%s", path.name)
    except PermissionError as exc:
        logger.error("CLEANUP_FAILED | file=%s | permission_error=%s", path.name, exc)
    except OSError as exc:
        logger.error("CLEANUP_FAILED | file=%s | error=%s", path.name, exc)

def _save_image(image: Image.Image, upload_file: UploadFile, destination: Path) -> bool:
    width, height = image.size
    needs_resize = width > settings.MAX_IMAGE_DIMENSION or height > settings.MAX_IMAGE_DIMENSION

    upload_file.file.seek(0)
    if not needs_resize:
        with destination.open("wb") as output:
            shutil.copyfileobj(upload_file.file, output, length=1024 * 1024)
        return False

    scale = min(settings.MAX_IMAGE_DIMENSION / width, settings.MAX_IMAGE_DIMENSION / height)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized_image = image.resize(new_size, Image.Resampling.LANCZOS)

    image_format = (image.format or "JPEG").upper()
    if image_format == "JPEG" and resized_image.mode not in {"RGB", "L"}:
        resized_image = resized_image.convert("RGB")

    save_kwargs = {}
    if image_format == "JPEG":
        save_kwargs = {"quality": 90, "optimize": True}
    elif image_format == "PNG":
        save_kwargs = {"optimize": True}
    elif image_format == "WEBP":
        save_kwargs = {"quality": 90}

    try:
        resized_image.save(destination, format=image_format, **save_kwargs)
    finally:
        resized_image.close()
    return True

def _validate_image_size(image: Image.Image) -> None:
    width, height = image.size
    pixels = width * height
    if pixels > settings.MAX_IMAGE_PIXELS:
        raise HTTPException(
            status_code=400,
            detail=f"Image dimensions are too large to process safely. Maximum allowed pixels: {settings.MAX_IMAGE_PIXELS:,}.",
        )

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    filename = _safe_filename(file.filename)
    saved_path: Path | None = None

    try:
        if not file.content_type or not file.content_type.startswith("image/"):
            logger.warning("ANALYSIS_FAILED | filename=%s | reason=not_an_image", filename)
            raise HTTPException(status_code=400, detail="Only image files are allowed.")

        try:
            # Use the underlying file object (not UploadFile wrapper) for
            # seek-with-whence to get the file size without reading.
            file.file.seek(0, 2)          # go to end
            file_size = file.file.tell()  # current position = size
            file.file.seek(0)            # back to beginning
        except (OSError, ValueError) as exc:
            logger.error("ANALYSIS_FAILED | filename=%s | size_check_error=%s", filename, exc)
            raise HTTPException(status_code=400, detail="Could not read the uploaded file.")

        if file_size == 0:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")

        if file_size > settings.MAX_FILE_SIZE:
            logger.warning("ANALYSIS_FAILED | filename=%s | size=%d | reason=file_too_large", filename, file_size)
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum allowed size is {settings.MAX_FILE_SIZE / (1024 * 1024):.1f} MB.",
            )

        await file.seek(0)
        try:
            image = Image.open(file.file)
            image.verify()
            await file.seek(0)
            image = Image.open(file.file)
            _validate_image_size(image)
            image.load()
        except UnidentifiedImageError:
            logger.warning("ANALYSIS_FAILED | filename=%s | reason=invalid_image", filename)
            raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")
        except Image.DecompressionBombError:
            logger.warning("ANALYSIS_FAILED | filename=%s | reason=image_too_large", filename)
            raise HTTPException(status_code=400, detail="Image dimensions are too large to process safely.")
        except HTTPException:
            raise
        except (OSError, ValueError) as exc:
            logger.warning("ANALYSIS_FAILED | filename=%s | image_error=%s", filename, exc)
            raise HTTPException(status_code=400, detail="The image could not be processed.")

        metadata = analyzer.extract_metadata(image)
        original_format = image.format or "UNKNOWN"

        try:
            settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        except PermissionError as exc:
            logger.error("ANALYSIS_FAILED | filename=%s | upload_dir_permission=%s", filename, exc)
            raise HTTPException(status_code=500, detail="The server cannot access the upload directory.")

        saved_path = _create_upload_path(filename)
        resized = _save_image(image=image, upload_file=file, destination=saved_path)

        if resized:
            processed_image = Image.open(saved_path)
            processed_image.load()
        else:
            processed_image = image

        try:
            result = analyzer.analyze(
                image=processed_image,
                filename=filename,
                file_path=saved_path,
                metadata_summary=metadata,
                resized=resized,
            )
        finally:
            if processed_image is not image:
                processed_image.close()

        logger.info(
            "ANALYSIS_SUCCESS | filename=%s | format=%s | dimensions=%sx%s | saved_bytes=%s | resized=%s",
            filename, original_format, result.width, result.height, result.file_size_bytes, resized,
        )

        background_tasks.add_task(_delete_file, saved_path)
        return result

    except HTTPException:
        if saved_path and saved_path.exists():
            _delete_file(saved_path)
        raise
    except (PermissionError, OSError) as exc:
        logger.exception("ANALYSIS_FAILED | filename=%s | filesystem_error=%s", filename, exc)
        if saved_path and saved_path.exists():
            _delete_file(saved_path)
        raise HTTPException(status_code=500, detail="A file-system error occurred while processing the image.")
    except Exception as exc:
        logger.exception("ANALYSIS_FAILED | filename=%s | unexpected_error=%s", filename, exc)
        if saved_path and saved_path.exists():
            _delete_file(saved_path)
        raise HTTPException(status_code=500, detail="An unexpected error occurred while analyzing the image.")
    finally:
        try:
            await file.close()
        except Exception:
            pass