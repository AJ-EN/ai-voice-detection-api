import numpy as np
from typing import Dict
import logging

try:
    import librosa
except ImportError:
    raise ImportError("librosa is required for feature extraction")

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract discriminative features for AI vs Human voice detection."""
    
    def __init__(self, sr: int = 16000):
        self.sr = sr
    
    def extract(self, waveform: np.ndarray) -> Dict:
        """
        Extract features from audio waveform.
        
        Args:
            waveform: Normalized audio waveform as numpy array
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        try:
            # 1. Fundamental Frequency (F0) analysis using PYIN
            f0, voiced_flag, voiced_probs = librosa.pyin(
                waveform,
                fmin=librosa.note_to_hz('C2'),  # ~65 Hz
                fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
                sr=self.sr
            )
            
            # Filter only voiced frames (where we have valid F0)
            f0_voiced = f0[~np.isnan(f0)]
            
            if len(f0_voiced) > 1:
                # 2. Jitter (pitch perturbation) - KEY FEATURE for AI detection
                # AI voices typically have very low jitter (< 1%)
                f0_diff = np.abs(np.diff(f0_voiced))
                features['jitter'] = float(np.mean(f0_diff) / (np.mean(f0_voiced) + 1e-8))
                features['jitter_abs'] = float(np.mean(f0_diff))
                features['f0_mean'] = float(np.mean(f0_voiced))
                features['f0_std'] = float(np.std(f0_voiced))
            else:
                features['jitter'] = 0.02  # Default to human-like value
                features['jitter_abs'] = 0.0
                features['f0_mean'] = 0.0
                features['f0_std'] = 0.0
            
            # 3. Shimmer (amplitude perturbation)
            frame_length = int(0.03 * self.sr)  # 30ms frames
            hop_length = frame_length // 2
            rms = librosa.feature.rms(y=waveform, frame_length=frame_length, hop_length=hop_length)[0]
            
            if len(rms) > 1:
                rms_diff = np.abs(np.diff(rms))
                features['shimmer'] = float(np.mean(rms_diff) / (np.mean(rms) + 1e-8))
            else:
                features['shimmer'] = 0.05  # Default
            
            # 4. Spectral features
            spec_centroid = librosa.feature.spectral_centroid(y=waveform, sr=self.sr)[0]
            spec_rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=self.sr)[0]
            spec_bandwidth = librosa.feature.spectral_bandwidth(y=waveform, sr=self.sr)[0]
            
            features['spectral_centroid_mean'] = float(np.mean(spec_centroid))
            features['spectral_centroid_std'] = float(np.std(spec_centroid))
            features['spectral_rolloff_mean'] = float(np.mean(spec_rolloff))
            features['spectral_bandwidth_mean'] = float(np.mean(spec_bandwidth))
            
            # 5. Spectral flux (rate of change in spectrum)
            spec_flux = librosa.onset.onset_strength(y=waveform, sr=self.sr)
            features['spectral_flux_mean'] = float(np.mean(spec_flux))
            features['spectral_flux_variance'] = float(np.var(spec_flux))
            
            # 6. Harmonic-to-Noise Ratio (HNR)
            # High HNR indicates cleaner signal - AI tends to have higher HNR
            harmonic = librosa.effects.harmonic(waveform)
            noise = waveform - harmonic
            
            harmonic_energy = np.sum(harmonic ** 2)
            noise_energy = np.sum(noise ** 2)
            
            if noise_energy > 0:
                features['hnr'] = float(harmonic_energy / (noise_energy + 1e-8))
                features['hnr_db'] = float(10 * np.log10(features['hnr'] + 1e-8))
            else:
                features['hnr'] = 100.0  # Very clean signal
                features['hnr_db'] = 20.0
            
            # 7. MFCCs (Mel-frequency cepstral coefficients) for timbre
            mfccs = librosa.feature.mfcc(y=waveform, sr=self.sr, n_mfcc=13)
            features['mfcc_means'] = [float(x) for x in np.mean(mfccs, axis=1)]
            features['mfcc_vars'] = [float(x) for x in np.var(mfccs, axis=1)]
            
            # 8. Zero crossing rate (indicates noisiness/fricatives)
            zcr = librosa.feature.zero_crossing_rate(waveform)[0]
            features['zcr_mean'] = float(np.mean(zcr))
            features['zcr_std'] = float(np.std(zcr))
            
            # 9. Voiced frame ratio
            total_frames = len(voiced_flag)
            voiced_frames = np.sum(voiced_flag) if len(voiced_flag) > 0 else 0
            features['voiced_ratio'] = float(voiced_frames / (total_frames + 1e-8))
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}", exc_info=True)
            # Return default features
            features = {
                'jitter': 0.02,
                'shimmer': 0.05,
                'hnr_db': 15.0,
                'spectral_flux_variance': 0.5,
                'zcr_mean': 0.1,
                'spectral_centroid_mean': 2000.0,
                'voiced_ratio': 0.8,
            }
        
        return features
