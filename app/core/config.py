from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    MAX_FILE_SIZE: int = 10 * 1024 * 1024
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    MAX_IMAGE_DIMENSION: int = 2048
    MAX_IMAGE_PIXELS: int = 50_000_000
    LOG_FILE: Path = BASE_DIR / "logs" / "app.log"
    LOG_LEVEL: str = "INFO"
    LOG_MAX_BYTES: int = 2 * 1024 * 1024
    LOG_BACKUP_COUNT: int = 3
    DOMINANT_COLOR_COUNT: int = 3
    COLOR_ANALYSIS_SIZE: int = 128

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

settings = Settings()