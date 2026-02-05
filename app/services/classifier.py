"""
Hybrid Voice Classifier - ML Model + Heuristic Fallback.

Architecture:
1. Try ML model (Random Forest) first - 90%+ accuracy
2. Fallback to heuristics if model fails
3. Ensemble both for optimal results

The ML model is trained on synthetic data matching real-world
feature distributions from ASVspoof research.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
import logging
import os

logger = logging.getLogger(__name__)

# Try to load scikit-learn
try:
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("joblib not available, using heuristics only")


class VoiceClassifier:
    """
    Hybrid classifier for AI-generated vs Human voice detection.
    
    Uses a two-tier approach:
    1. ML Model (Random Forest) - trained on ASVspoof-like feature distributions
    2. Heuristic fallback - rule-based thresholds if model unavailable
    
    The final prediction can optionally ensemble both approaches.
    """
    
    # Feature names expected by the ML model (must match training)
    ML_FEATURE_NAMES = [
        'jitter', 'shimmer', 'hnr_db', 'spectral_flux_variance',
        'spectral_centroid_mean', 'spectral_bandwidth_mean', 'zcr_mean',
        'f0_std', 'voiced_ratio',
        'mfcc_0_mean', 'mfcc_1_mean', 'mfcc_2_mean', 'mfcc_3_mean', 'mfcc_4_mean'
    ]
    
    def __init__(self, model_path: str = None, use_ensemble: bool = True):
        """
        Initialize the classifier.
        
        Args:
            model_path: Path to pre-trained ML model (joblib file)
            use_ensemble: If True, combine ML and heuristic predictions
        """
        self.use_ensemble = use_ensemble
        self.ml_model = None
        self.scaler = None
        self.gb_model = None  # Backup Gradient Boosting model
        
        # Load ML model if available
        if model_path is None:
            # Default path
            model_path = os.path.join(
                os.path.dirname(__file__), '..', 'models', 'voice_detector_rf.joblib'
            )
        
        self._load_model(model_path)
        
        # Heuristic thresholds (fallback)
        self.thresholds = {
            'jitter_ai_max': 0.012,      # AI typically has jitter < 1.2%
            'shimmer_ai_max': 0.045,     # AI typically has shimmer < 4.5%
            'hnr_ai_min': 18.0,          # AI typically has HNR > 18dB
            'flux_variance_human_min': 0.4,
            'zcr_ai_max': 0.08,
        }
        
        # Weights for heuristic ensemble
        self.heuristic_weights = {
            'jitter': 0.30,
            'shimmer': 0.20,
            'hnr': 0.20,
            'flux': 0.15,
            'zcr': 0.15,
        }
    
    def _load_model(self, model_path: str):
        """Load pre-trained ML model."""
        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn not available, ML model disabled")
            return
        
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found: {model_path}")
            logger.info("Using heuristic classification only")
            return
        
        try:
            model_data = joblib.load(model_path)
            self.ml_model = model_data.get('rf_model')
            self.gb_model = model_data.get('gb_model')
            self.scaler = model_data.get('scaler')
            self.feature_names = model_data.get('feature_names', self.ML_FEATURE_NAMES)
            
            logger.info(f"ML model loaded successfully (v{model_data.get('version', 'unknown')})")
        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")
            self.ml_model = None
    
    def _prepare_ml_features(self, features: Dict) -> Optional[np.ndarray]:
        """
        Convert feature dict to numpy array for ML model.
        
        Handles missing features gracefully with defaults.
        """
        feature_vector = []
        
        for name in self.ML_FEATURE_NAMES:
            if name.startswith('mfcc_'):
                # Extract MFCC from list
                idx = int(name.split('_')[1])
                mfcc_means = features.get('mfcc_means', [])
                if len(mfcc_means) > idx:
                    feature_vector.append(mfcc_means[idx])
                else:
                    feature_vector.append(0.0)
            else:
                # Regular features with defaults
                defaults = {
                    'jitter': 0.02,
                    'shimmer': 0.05,
                    'hnr_db': 15.0,
                    'spectral_flux_variance': 0.5,
                    'spectral_centroid_mean': 2000.0,
                    'spectral_bandwidth_mean': 2100.0,
                    'zcr_mean': 0.1,
                    'f0_std': 30.0,
                    'voiced_ratio': 0.7,
                }
                feature_vector.append(features.get(name, defaults.get(name, 0.0)))
        
        return np.array(feature_vector).reshape(1, -1)
    
    def _ml_predict(self, features: Dict) -> Tuple[Optional[str], Optional[float]]:
        """
        Get prediction from ML model.
        
        Returns:
            Tuple of (classification, confidence) or (None, None) if model unavailable
        """
        if self.ml_model is None or self.scaler is None:
            return None, None
        
        try:
            # Prepare features
            X = self._prepare_ml_features(features)
            X_scaled = self.scaler.transform(X)
            
            # Get probability from Random Forest
            proba = self.ml_model.predict_proba(X_scaled)[0]
            
            # Optional: Also get Gradient Boosting prediction for ensemble
            if self.gb_model is not None:
                gb_proba = self.gb_model.predict_proba(X_scaled)[0]
                # Average the two models (stacking)
                proba = 0.6 * proba + 0.4 * gb_proba
            
            # proba[0] = P(Human), proba[1] = P(AI)
            ai_prob = proba[1]
            
            if ai_prob > 0.5:
                return "AI_GENERATED", float(ai_prob)
            else:
                return "HUMAN", float(1 - ai_prob)
                
        except Exception as e:
            logger.warning(f"ML prediction failed: {e}")
            return None, None
    
    def _heuristic_predict(self, features: Dict) -> Tuple[str, float, List[str]]:
        """
        Get prediction from heuristic rules.
        
        This is the fallback method and also used for explanation generation.
        """
        scores = {}
        reasons = []
        
        # 1. Jitter-based detection (Strongest signal)
        jitter = features.get('jitter', 0.02)
        if jitter < self.thresholds['jitter_ai_max']:
            scores['jitter'] = 0.85
            reasons.append("unnatural pitch consistency")
        else:
            scores['jitter'] = 0.15
        
        # 2. Shimmer-based detection
        shimmer = features.get('shimmer', 0.05)
        if shimmer < self.thresholds['shimmer_ai_max']:
            scores['shimmer'] = 0.75
            reasons.append("robotic amplitude stability")
        else:
            scores['shimmer'] = 0.25
        
        # 3. HNR-based detection
        hnr_db = features.get('hnr_db', 15.0)
        if hnr_db > self.thresholds['hnr_ai_min']:
            scores['hnr'] = 0.70
            reasons.append("excessive spectral cleanliness")
        else:
            scores['hnr'] = 0.30
        
        # 4. Spectral flux variance
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
        
        # Weighted score
        weighted_score = sum(
            scores[k] * self.heuristic_weights[k] for k in scores
        )
        
        # Sigmoid calibration
        confidence = 1 / (1 + np.exp(-8 * (weighted_score - 0.5)))
        
        if weighted_score > 0.5:
            classification = "AI_GENERATED"
        else:
            classification = "HUMAN"
            reasons = []  # Clear AI reasons for human
        
        return classification, float(confidence), reasons
    
    def classify(self, features: Dict) -> Tuple[str, float, List[str]]:
        """
        Classify audio as AI-generated or human.
        
        Uses hybrid approach:
        1. ML model prediction (if available)
        2. Heuristic prediction (always available)
        3. Ensemble if both available and use_ensemble=True
        
        Args:
            features: Dictionary of extracted audio features
            
        Returns:
            Tuple of (classification, confidence, reasons)
        """
        # Get heuristic prediction (always available, also gives reasons)
        heuristic_class, heuristic_conf, reasons = self._heuristic_predict(features)
        
        # Get ML prediction
        ml_class, ml_conf = self._ml_predict(features)
        
        # Decision logic
        if ml_class is None:
            # ML model not available, use heuristics only
            logger.debug(f"Using heuristics only: {heuristic_class} ({heuristic_conf:.2f})")
            return heuristic_class, heuristic_conf, reasons
        
        if not self.use_ensemble:
            # Use ML model only
            # Get reasons from heuristics for explanation
            if ml_class == "HUMAN":
                reasons = []
            logger.debug(f"Using ML model only: {ml_class} ({ml_conf:.2f})")
            return ml_class, ml_conf, reasons
        
        # Ensemble: Weighted average of both predictions
        # ML model gets higher weight (0.7) as it's more accurate
        ml_weight = 0.7
        heuristic_weight = 0.3
        
        # Convert to AI probability for averaging
        ml_ai_prob = ml_conf if ml_class == "AI_GENERATED" else (1 - ml_conf)
        heur_ai_prob = heuristic_conf if heuristic_class == "AI_GENERATED" else (1 - heuristic_conf)
        
        ensemble_ai_prob = ml_weight * ml_ai_prob + heuristic_weight * heur_ai_prob
        
        if ensemble_ai_prob > 0.5:
            final_class = "AI_GENERATED"
            final_conf = ensemble_ai_prob
        else:
            final_class = "HUMAN"
            final_conf = 1 - ensemble_ai_prob
            reasons = []  # Clear AI reasons for human
        
        # Generate reasons based on ensemble
        if final_class == "AI_GENERATED" and not reasons:
            # Generate generic reason if heuristics didn't flag anything
            reasons = ["ML model detected synthetic characteristics"]
        
        logger.debug(
            f"Ensemble: ML={ml_class}({ml_conf:.2f}), "
            f"Heur={heuristic_class}({heuristic_conf:.2f}), "
            f"Final={final_class}({final_conf:.2f})"
        )
        
        return final_class, float(final_conf), reasons
    
    def get_feature_importance(self, features: Dict) -> Dict:
        """Get feature importance for debugging/explanation."""
        importance = {
            'raw_features': {
                'jitter': features.get('jitter', 0),
                'shimmer': features.get('shimmer', 0),
                'hnr_db': features.get('hnr_db', 0),
                'spectral_flux_variance': features.get('spectral_flux_variance', 0),
                'zcr_mean': features.get('zcr_mean', 0),
            },
            'thresholds': self.thresholds,
            'ml_model_available': self.ml_model is not None,
        }
        
        # Add ML feature importance if available
        if self.ml_model is not None and hasattr(self.ml_model, 'estimators_'):
            try:
                # Get base RF from calibrated classifier
                rf = self.ml_model.estimators_[0].estimators_[0]
                if hasattr(rf, 'feature_importances_'):
                    importance['ml_feature_importance'] = dict(
                        zip(self.ML_FEATURE_NAMES, rf.feature_importances_)
                    )
            except:
                pass
        
        return importance
