from pydantic import BaseModel
from typing import Optional, List, Dict

class AnalysisResponse(BaseModel):
    filename: str
    format: str
    width: int
    height: int
    file_size_bytes: int

    ai_probability: Optional[float] = None
    human_probability: Optional[float] = None

    metadata_summary: Dict[str, str] = {}   # e.g., {"Make": "Canon"}
    signals: List[str] = []                  # human-readable clues

    confidence: Optional[str] = None         # LOW / MEDIUM / HIGH

    note: str