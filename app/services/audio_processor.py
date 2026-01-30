import base64
import tempfile
import os
import numpy as np
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)

# Try to import torchaudio, fall back to librosa if not available
try:
    import torchaudio
    import torch
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    logger.warning("torchaudio not available, using librosa for audio loading")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

if not TORCHAUDIO_AVAILABLE and not LIBROSA_AVAILABLE:
    raise ImportError("Either torchaudio or librosa must be installed")


class AudioProcessor:
    """Process base64 encoded MP3 audio to normalized waveform."""
    
    def __init__(self, target_sr: int = 16000, max_duration: float = 60.0, min_duration: float = 0.5):
        self.target_sr = target_sr
        self.max_duration = max_duration
        self.min_duration = min_duration
    
    async def process(self, base64_audio: str) -> np.ndarray:
        """
        Process base64 MP3 to normalized waveform.
        
        Args:
            base64_audio: Base64 encoded MP3 audio data
            
        Returns:
            numpy array of shape (samples,) with normalized audio
        """
        try:
            # Decode base64
            audio_bytes = base64.b64decode(base64_audio)
            
            # Validate size (DoS protection)
            if len(audio_bytes) > 10 * 1024 * 1024:
                raise HTTPException(413, "Audio file too large (max 10MB)")
            
            # Use tempfile for processing
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_in:
                tmp_in.write(audio_bytes)
                tmp_in.flush()
                tmp_path = tmp_in.name
            
            try:
                waveform, sample_rate = self._load_audio(tmp_path)
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            
            # Convert to mono if stereo
            if len(waveform.shape) > 1 and waveform.shape[0] > 1:
                waveform = np.mean(waveform, axis=0)
            elif len(waveform.shape) > 1:
                waveform = waveform.squeeze()
            
            # Resample if needed
            if sample_rate != self.target_sr:
                waveform = self._resample(waveform, sample_rate, self.target_sr)
            
            # Validate duration
            duration = len(waveform) / self.target_sr
            if duration > self.max_duration:
                raise HTTPException(413, f"Audio too long (max {self.max_duration} seconds)")
            
            if duration < self.min_duration:
                raise HTTPException(400, f"Audio too short (min {self.min_duration} seconds)")
            
            # Normalize
            max_val = np.max(np.abs(waveform))
            if max_val > 0:
                waveform = waveform / (max_val + 1e-8)
            
            return waveform.astype(np.float32)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Audio processing error: {e}", exc_info=True)
            raise HTTPException(400, f"Failed to process audio: {str(e)}")
    
    def _load_audio(self, filepath: str) -> tuple:
        """Load audio from file using available backend."""
        if TORCHAUDIO_AVAILABLE:
            try:
                waveform, sample_rate = torchaudio.load(filepath)
                return waveform.numpy(), sample_rate
            except Exception as e:
                logger.warning(f"torchaudio failed: {e}, trying librosa")
        
        if LIBROSA_AVAILABLE:
            waveform, sample_rate = librosa.load(filepath, sr=None, mono=False)
            return waveform, sample_rate
        
        raise RuntimeError("No audio backend available")
    
    def _resample(self, waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        if TORCHAUDIO_AVAILABLE:
            import torch
            tensor = torch.from_numpy(waveform).unsqueeze(0) if len(waveform.shape) == 1 else torch.from_numpy(waveform)
            resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
            resampled = resampler(tensor)
            return resampled.squeeze().numpy()
        
        if LIBROSA_AVAILABLE:
            return librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)
        
        raise RuntimeError("No audio backend available for resampling")
