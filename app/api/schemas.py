from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class AnalysisResponse(BaseModel):
    filename: str
    format: str
    width: int
    height: int
    file_size_bytes: int

    ai_probability: Optional[float] = None
    human_probability: Optional[float] = None

    metadata_summary: Dict[str, str] = Field(default_factory=dict)
    signals: List[str] = Field(default_factory=list)

    confidence: Optional[str] = None

    dominant_colors: List[str] = Field(default_factory=list)

    model_predictions: Dict[str, float] = Field(default_factory=dict)

    note: str