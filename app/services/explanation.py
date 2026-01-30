from typing import Dict, List


def generate_explanation(features: Dict, classification: str, reasons: List[str] = None) -> str:
    """
    Generate human-readable explanations based on detected artifacts.
    
    Args:
        features: Dictionary of extracted audio features
        classification: The classification result (AI_GENERATED or HUMAN)
        reasons: Optional list of detected reasons from classifier
        
    Returns:
        Human-readable explanation string
    """
    if classification == "AI_GENERATED":
        if reasons and len(reasons) > 0:
            # Use the reasons from classifier
            top_reasons = reasons[:3]
            return f"Detected: {', '.join(top_reasons)}"
        
        # Fallback: generate reasons from features
        detected_artifacts = []
        
        # Primary indicators (strongest signals first)
        if features.get('jitter', 1) < 0.012:
            detected_artifacts.append("unnatural pitch consistency")
        
        if features.get('shimmer', 1) < 0.045:
            detected_artifacts.append("robotic amplitude stability")
        
        if features.get('hnr_db', 0) > 18:
            detected_artifacts.append("excessive spectral cleanliness")
        
        if features.get('spectral_flux_variance', 1) < 0.4:
            detected_artifacts.append("overly smooth spectral transitions")
        
        if features.get('zcr_mean', 0.5) < 0.08:
            detected_artifacts.append("lack of natural breath noise")
        
        if not detected_artifacts:
            detected_artifacts.append("synthetic vocal patterns detected")
        
        return f"Detected: {', '.join(detected_artifacts[:3])}"
    
    else:  # HUMAN
        positive_indicators = []
        
        if features.get('jitter', 0) > 0.015:
            positive_indicators.append("natural pitch variation")
        
        if features.get('shimmer', 0) > 0.05:
            positive_indicators.append("organic amplitude dynamics")
        
        if features.get('spectral_flux_variance', 0) > 0.5:
            positive_indicators.append("natural spectral dynamics")
        
        if features.get('zcr_mean', 0) > 0.1:
            positive_indicators.append("natural breath patterns")
        
        if features.get('hnr_db', 100) < 16:
            positive_indicators.append("natural harmonic texture")
        
        if positive_indicators:
            return f"Natural vocal characteristics: {', '.join(positive_indicators[:3])}"
        else:
            return "Natural speech patterns with human-like variations detected"
