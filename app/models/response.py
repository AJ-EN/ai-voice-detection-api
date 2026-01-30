from pydantic import BaseModel, Field
from typing import Literal, Optional


class VoiceDetectionResponse(BaseModel):
    """Successful response model for voice detection."""
    
    status: Literal["success", "error"] = Field(
        ..., description="Request status"
    )
    language: str = Field(
        ..., description="Detected or provided language"
    )
    classification: Literal["AI_GENERATED", "HUMAN"] = Field(
        ..., description="Voice classification result"
    )
    confidenceScore: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score between 0 and 1"
    )
    explanation: str = Field(
        ..., min_length=10, max_length=500, 
        description="Human-readable explanation for the decision"
    )
    processingTime: Optional[float] = Field(
        None, description="Processing time in seconds"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "language": "Tamil",
                "classification": "AI_GENERATED",
                "confidenceScore": 0.92,
                "explanation": "Detected: unnatural pitch consistency, robotic amplitude stability",
                "processingTime": 0.45
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""
    
    status: Literal["error"] = "error"
    message: str = Field(..., description="Error description")
    errorCode: Optional[str] = Field(None, description="Machine-readable error code")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "error",
                "message": "Invalid API key",
                "errorCode": "AUTH_INVALID"
            }
        }
