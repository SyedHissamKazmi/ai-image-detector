import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from app.core.config import settings

def configure_logging() -> None:
    log_path = Path(settings.LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("ai_image_detector")
    logger.setLevel(settings.LOG_LEVEL.upper())
    logger.propagate = False

    if logger.handlers:
        return

    handler = RotatingFileHandler(
        log_path,
        maxBytes=settings.LOG_MAX_BYTES,
        backupCount=settings.LOG_BACKUP_COUNT,
        encoding="utf-8",
    )

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"ai_image_detector.{name}")