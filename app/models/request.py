from pydantic import BaseModel, Field, validator
import base64
import binascii


VALID_LANGUAGES = {"tamil", "english", "hindi", "malayalam", "telugu"}
LANGUAGE_MAP = {lang.lower(): lang.title() for lang in ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]}


class VoiceDetectionRequest(BaseModel):
    """Request model for voice detection endpoint."""
    
    language: str = Field(
        ..., description="Language of the audio sample (Tamil, English, Hindi, Malayalam, Telugu)"
    )
    audioFormat: str = Field(
        default="mp3", description="Audio format (only mp3 supported)"
    )
    audioBase64: str = Field(
        ..., 
        min_length=100,  # Minimum valid base64 audio
        description="Base64 encoded MP3 audio data"
    )
    
    @validator('language', pre=True)
    def normalize_language(cls, v: str) -> str:
        """Normalize language to title case and validate."""
        if not isinstance(v, str):
            raise ValueError("Language must be a string")
        
        normalized = v.strip().lower()
        if normalized not in VALID_LANGUAGES:
            raise ValueError(f"Invalid language '{v}'. Must be one of: Tamil, English, Hindi, Malayalam, Telugu")
        
        return LANGUAGE_MAP[normalized]
    
    @validator('audioFormat', pre=True)
    def normalize_audio_format(cls, v: str) -> str:
        """Normalize audio format to lowercase."""
        if not isinstance(v, str):
            raise ValueError("Audio format must be a string")
        
        normalized = v.strip().lower()
        if normalized != "mp3":
            raise ValueError("Only 'mp3' audio format is supported")
        
        return normalized
    
    @validator('audioBase64')
    def validate_base64(cls, v: str) -> str:
        """Validate base64 encoding and basic MP3 structure."""
        try:
            # Check if valid base64
            decoded = base64.b64decode(v, validate=True)
            
            # Check size (prevent DoS) - 10MB limit
            if len(decoded) > 10 * 1024 * 1024:
                raise ValueError("Audio data too large (max 10MB)")
            
            # Check minimum size for valid audio
            if len(decoded) < 100:
                raise ValueError("Audio data too small to be valid")
            
            # Check if it's actually an MP3 (magic bytes)
            # MP3 files start with either ID3 tag or frame sync
            is_id3 = decoded[:3] == b'ID3'
            is_frame_sync = len(decoded) >= 2 and decoded[0] == 0xFF and (decoded[1] & 0xE0) == 0xE0
            
            if not is_id3 and not is_frame_sync:
                raise ValueError("Invalid MP3 data: unrecognized format")
                
            return v
        except binascii.Error:
            raise ValueError("Invalid base64 encoding")
        except Exception as e:
            if "Invalid" in str(e) or "too" in str(e):
                raise
            raise ValueError(f"Error validating audio: {str(e)}")
    
    class Config:
        schema_extra = {
            "example": {
                "language": "Tamil",
                "audioFormat": "mp3",
                "audioBase64": "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU2LjM2LjEwMAAAAAAA..."
            }
        }
