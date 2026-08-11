from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image
from PIL.ExifTags import TAGS

from app.api.schemas import AnalysisResponse
from app.core.config import settings
from app.core.logging_config import get_logger
from app.detector.model import AIDetector


logger = get_logger("analyzer")

# One detector instance shared across requests.
# The actual Hugging Face model is loaded lazily on first prediction.
detector = AIDetector()


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

        # ---------------------------------------------------------
        # File size
        # ---------------------------------------------------------
        try:
            file_size = Path(file_path).stat().st_size
        except (FileNotFoundError, PermissionError, OSError) as exc:
            logger.warning(
                "FILE_SIZE_FAILED | filename=%s | error=%s",
                filename,
                exc,
            )
            file_size = 0

        # ---------------------------------------------------------
        # Metadata
        # ---------------------------------------------------------
        metadata = (
            metadata_summary
            if metadata_summary is not None
            else self.extract_metadata(image)
        )

        # ---------------------------------------------------------
        # Dominant colours
        # ---------------------------------------------------------
        dominant_colors = self.extract_dominant_colors(image)

        # ---------------------------------------------------------
        # Existing signals
        # ---------------------------------------------------------
        signals: List[str] = []

        if resized:
            signals.append(
                f"Image resized to fit within "
                f"{settings.MAX_IMAGE_DIMENSION}px."
            )

        # ---------------------------------------------------------
        # AI detector
        # ---------------------------------------------------------
        ai_probability: Optional[float] = None
        human_probability: Optional[float] = None
        confidence: Optional[str] = None

        try:
            ai_probability = detector.predict(file_path)
        except Exception as exc:
            # Extra safety: detector failures must never break the
            # existing metadata/colour analysis.
            logger.warning(
                "ML_ANALYSIS_FAILED | filename=%s | error=%s",
                filename,
                exc,
            )
            ai_probability = None

        if ai_probability is not None:
            human_probability = 1.0 - ai_probability

            # Required confidence thresholds.
            if ai_probability < 0.6:
                confidence = "LOW"
            elif ai_probability <= 0.8:
                confidence = "MEDIUM"
            else:
                confidence = "HIGH"

            # Required ML signals.
            if ai_probability > 0.8:
                signals.append("ML detector: strong AI signal")
            elif 0.4 < ai_probability < 0.6:
                signals.append("ML detector: uncertain")

            note = (
                "Image analyzed. This result is probabilistic and should "
                "not be treated as definitive proof."
            )

        else:
            signals.append("ML detector: unavailable")

            note = (
                "Image analyzed successfully, but ML detection was "
                "skipped because the AI detector is unavailable."
            )

        # ---------------------------------------------------------
        # Final response
        # ---------------------------------------------------------
        return AnalysisResponse(
            filename=filename,
            format=image.format or "UNKNOWN",
            width=width,
            height=height,
            file_size_bytes=file_size,
            ai_probability=ai_probability,
            human_probability=human_probability,
            metadata_summary=metadata,
            signals=signals,
            confidence=confidence,
            dominant_colors=dominant_colors,
            note=note,
        )

    # =============================================================
    # METADATA
    # =============================================================

    def extract_metadata(
        self,
        image: Image.Image,
    ) -> Dict[str, str]:

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
                    logger.debug(
                        "EXIF_TAG_FAILED | tag=%s | error=%s",
                        tag_id,
                        exc,
                    )

        except Exception as exc:
            # Corrupt/unusual EXIF must not stop the ML analysis.
            logger.warning(
                "EXIF_EXTRACTION_FAILED | error=%s",
                exc,
            )

        return metadata

    # =============================================================
    # DOMINANT COLOURS
    # =============================================================

    def extract_dominant_colors(
        self,
        image: Image.Image,
        color_count: Optional[int] = None,
    ) -> List[str]:

        count = color_count or settings.DOMINANT_COLOR_COUNT

        try:
            sample = image.copy()

            sample.thumbnail(
                (
                    settings.COLOR_ANALYSIS_SIZE,
                    settings.COLOR_ANALYSIS_SIZE,
                ),
                Image.Resampling.BILINEAR,
            )

            sample = sample.convert("RGB")

            sample = sample.quantize(
                colors=max(count, 3),
                method=Image.Quantize.MEDIANCUT,
            )

            palette = sample.getpalette()
            color_counts = sample.getcolors()

            if not palette or not color_counts:
                sample.close()
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

                colors.append(
                    f"#{red:02X}{green:02X}{blue:02X}"
                )

            sample.close()

            return colors

        except Exception as exc:
            logger.warning(
                "DOMINANT_COLOR_EXTRACTION_FAILED | error=%s",
                exc,
            )
            return []