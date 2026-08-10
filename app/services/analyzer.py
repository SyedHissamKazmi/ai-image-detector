from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image
from PIL.ExifTags import TAGS
from app.api.schemas import AnalysisResponse
from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger("analyzer")

class ImageAnalyzer:
    def analyze(
        self,
        image: Image.Image,
        filename: str,
        file_path: str | Path,
        metadata_summary: Optional[Dict[str, str]] = None,
        resized: bool = False,
    ) -> AnalysisResponse:
        width, height = image.size

        try:
            file_size = Path(file_path).stat().st_size
        except (FileNotFoundError, PermissionError, OSError) as exc:
            logger.warning("Could not read final file size for %s: %s", filename, exc)
            file_size = 0

        metadata = metadata_summary or self.extract_metadata(image)
        dominant_colors = self.extract_dominant_colors(image)

        signals: List[str] = []
        if resized:
            signals.append(f"Image resized to fit within {settings.MAX_IMAGE_DIMENSION}px.")

        return AnalysisResponse(
            filename=filename,
            format=image.format or "UNKNOWN",
            width=width,
            height=height,
            file_size_bytes=file_size,
            ai_probability=None,
            human_probability=None,
            metadata_summary=metadata,
            signals=signals,
            confidence=None,
            dominant_colors=dominant_colors,
            note="Image uploaded and analyzed successfully. ML analysis is not yet available.",
        )

    def extract_metadata(self, image: Image.Image) -> Dict[str, str]:
        metadata: Dict[str, str] = {}
        try:
            exif = image.getexif()
            if not exif:
                return metadata
            for tag_id, value in exif.items():
                try:
                    tag_name = TAGS.get(tag_id, str(tag_id))
                    metadata[str(tag_name)] = str(value)
                except Exception as exc:
                    logger.debug("Could not decode EXIF tag %s: %s", tag_id, exc)
        except Exception as exc:
            logger.warning("EXIF extraction failed: %s", exc)
        return metadata

    def extract_dominant_colors(
        self,
        image: Image.Image,
        color_count: Optional[int] = None,
    ) -> List[str]:
        count = color_count or settings.DOMINANT_COLOR_COUNT
        try:
            sample = image.copy()
            sample.thumbnail((settings.COLOR_ANALYSIS_SIZE, settings.COLOR_ANALYSIS_SIZE), Image.Resampling.BILINEAR)
            sample = sample.convert("RGB")
            sample = sample.quantize(colors=max(count, 3), method=Image.Quantize.MEDIANCUT)
            palette = sample.getpalette()
            color_counts = sample.getcolors()
            if not palette or not color_counts:
                return []
            color_counts.sort(reverse=True)
            colors: List[str] = []
            for pixel_count, palette_index in color_counts[:count]:
                base = palette_index * 3
                if base + 2 >= len(palette):
                    continue
                red = palette[base]
                green = palette[base + 1]
                blue = palette[base + 2]
                colors.append(f"#{red:02X}{green:02X}{blue:02X}")
            return colors
        except Exception as exc:
            logger.warning("Dominant color extraction failed: %s", exc)
            return []