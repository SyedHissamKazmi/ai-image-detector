# app/services/analyzer.py
from PIL import Image
from PIL.ExifTags import TAGS
import os
from app.api.schemas import AnalysisResponse

class ImageAnalyzer:
    """Extracts metadata and (in future) runs ML detection."""

    def analyze(self, image: Image.Image, filename: str, file_path: str) -> AnalysisResponse:
        width, height = image.size
        img_format = image.format or os.path.splitext(filename)[1].upper().replace(".", "")
        file_size = os.path.getsize(file_path)

        # Extract EXIF metadata safely
        metadata = {}
        try:
            exif = image.getexif()
            if exif:
                for tag_id, value in exif.items():
                    tag_name = TAGS.get(tag_id, tag_id)
                    metadata[tag_name] = str(value)
        except Exception:
            # Some images may throw during EXIF reading
            pass

        return AnalysisResponse(
            filename=filename,
            format=img_format,
            width=width,
            height=height,
            file_size_bytes=file_size,
            ai_probability=None,
            human_probability=None,
            metadata_summary=metadata,
            signals=[],
            confidence=None,
            note="Image uploaded successfully. ML analysis not yet available."
        )