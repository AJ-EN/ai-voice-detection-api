import numpy as np
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class VoiceClassifier:
    """
    Hybrid classifier for AI-generated vs Human voice detection.
    Uses rule-based thresholds with ensemble scoring.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the classifier.
        
        Args:
            model_path: Optional path to pre-trained ML model (for future use)
        """
        # Thresholds based on acoustic analysis of AI vs human voices
        # AI-generated voices typically have:
        # - Very consistent pitch (low jitter)
        # - Consistent amplitude (low shimmer)  
        # - Clean spectrum (high HNR)
        # - Smooth spectral transitions (low flux variance)
        self.thresholds = {
            'jitter_ai_max': 0.012,      # AI typically has jitter < 1.2%
            'shimmer_ai_max': 0.045,     # AI typically has shimmer < 4.5%
            'hnr_ai_min': 18.0,          # AI typically has HNR > 18dB
            'flux_variance_human_min': 0.4,  # Humans have higher variance
            'zcr_ai_max': 0.08,          # AI has lower zero crossing rate
        }
        
        # Feature weights for ensemble
        self.weights = {
            'jitter': 0.30,       # Strongest signal
            'shimmer': 0.20,
            'hnr': 0.20,
            'flux': 0.15,
            'zcr': 0.15,
        }
    
    def classify(self, features: Dict) -> Tuple[str, float, Dict]:
        """
        Classify audio as AI-generated or human.
        
        Args:
            features: Dictionary of extracted audio features
            
        Returns:
            Tuple of (classification, confidence, feature_importance)
        """
        scores = {}
        reasons = []
        
        # 1. Jitter-based detection (Strongest signal)
        jitter = features.get('jitter', 0.02)
        if jitter < self.thresholds['jitter_ai_max']:
            scores['jitter'] = 0.85  # High confidence AI
            reasons.append("unnatural pitch consistency")
        else:
            scores['jitter'] = 0.15  # Likely Human
        
        # 2. Shimmer-based detection
        shimmer = features.get('shimmer', 0.05)
        if shimmer < self.thresholds['shimmer_ai_max']:
            scores['shimmer'] = 0.75
            reasons.append("robotic amplitude stability")
        else:
            scores['shimmer'] = 0.25
        
        # 3. HNR-based detection (cleanliness)
        hnr_db = features.get('hnr_db', 15.0)
        if hnr_db > self.thresholds['hnr_ai_min']:
            scores['hnr'] = 0.70
            reasons.append("excessive spectral cleanliness")
        else:
            scores['hnr'] = 0.30
        
        # 4. Spectral flux variance (dynamism)
        flux_var = features.get('spectral_flux_variance', 0.5)
        if flux_var < self.thresholds['flux_variance_human_min']:
            scores['flux'] = 0.75
            reasons.append("overly smooth spectral transitions")
        else:
            scores['flux'] = 0.25
        
        # 5. Zero crossing rate
        zcr = features.get('zcr_mean', 0.1)
        if zcr < self.thresholds['zcr_ai_max']:
            scores['zcr'] = 0.65
            reasons.append("lack of natural breath noise")
        else:
            scores['zcr'] = 0.35
        
        # Weighted ensemble average
        weighted_score = sum(
            scores[k] * self.weights[k] for k in scores
        )
        
        # Confidence calibration using sigmoid to push to extremes
        # This makes borderline cases have ~0.5 confidence
        confidence = 1 / (1 + np.exp(-8 * (weighted_score - 0.5)))
        
        # Classification
        if weighted_score > 0.5:
            classification = "AI_GENERATED"
        else:
            classification = "HUMAN"
            reasons = []  # Clear AI reasons for human classification
        
        # Feature importance for debugging
        feature_importance = {
            'scores': scores,
            'weighted_score': float(weighted_score),
            'raw_features': {
                'jitter': jitter,
                'shimmer': shimmer,
                'hnr_db': hnr_db,
                'spectral_flux_variance': flux_var,
                'zcr_mean': zcr,
            }
        }
        
        logger.debug(f"Classification: {classification}, Score: {weighted_score:.3f}, Confidence: {confidence:.2f}")
        
        return classification, float(confidence), reasons
