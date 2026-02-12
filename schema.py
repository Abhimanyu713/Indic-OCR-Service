from pydantic import BaseModel, Field
from typing import List, Optional

class OCRResponse(BaseModel):
    extracted_text: str = Field(..., description="The full text extracted from the image")
    language_detected: Optional[str] = Field(None, description="Primary language identified in the document")
    confidence_score: Optional[float] = Field(None, description="Self-reported confidence from the model (0-1)")

class ErrorResponse(BaseModel):
    error: str
    detail: str