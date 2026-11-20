"""Prediction and suggestion logic for water quality."""
import os
import joblib
import pandas as pd
from config import MODELS_DIR, QUALITY_LABELS, PARAM_RANGES

FEATURES = [
    "ph", "turbidity", "dissolved_oxygen", "temperature",
    "conductivity", "total_dissolved_solids", "nitrate",
    "fecal_coliform", "bod"
]

def load_model():
    model_path = os.path.join(MODELS_DIR, "water_quality_model.joblib")
    le_path = os.path.join(MODELS_DIR, "label_encoder.joblib")
    if not os.path.exists(model_path):
        return None, None
    return joblib.load(model_path), joblib.load(le_path)

def predict_quality(params_dict):
    """Predict quality from dict of parameters. Returns (label, suggestions)."""
    model, le = load_model()
    if model is None or le is None:
        return None, ["Model not found. Please run train_model.py first."]
    X = pd.DataFrame([params_dict])[FEATURES]
    pred_enc = model.predict(X)[0]
    label = le.inverse_transform([pred_enc])[0]
    suggestions = get_suggestions(label, params_dict)
    return label, suggestions

def get_suggestions(quality, params):
    """Generate improvement suggestions based on parameters."""
    suggestions = []
    ph_lo, ph_hi = PARAM_RANGES["ph"]
    if params.get("ph", 7) < ph_lo:
        suggestions.append("pH is low: consider adding alkaline treatment or aeration to raise pH toward 6.5–8.5.")
    elif params.get("ph", 7) > ph_hi:
        suggestions.append("pH is high: consider neutralization or dilution to bring pH within 6.5–8.5.")
    if params.get("dissolved_oxygen", 8) < 5:
        suggestions.append("Low dissolved oxygen: increase aeration, reduce organic load, or add oxygen.")
    if params.get("turbidity", 0) > 5:
        suggestions.append("High turbidity: use coagulation, flocculation, or filtration to reduce suspended solids.")
    if params.get("total_dissolved_solids", 0) > 500:
        suggestions.append("High TDS: consider reverse osmosis or dilution with cleaner water.")
    if params.get("nitrate", 0) > 10:
        suggestions.append("Elevated nitrate: reduce fertilizer runoff and check for sewage contamination.")
    if params.get("fecal_coliform", 0) > 0:
        suggestions.append("Fecal coliform detected: disinfect (chlorination/UV) and identify contamination source.")
    if params.get("bod", 0) > 3:
        suggestions.append("High BOD: reduce organic discharge and improve aeration.")
    if quality == "Good" and not suggestions:
        suggestions.append("Water quality is good. Maintain current treatment and monitoring.")
    return suggestions if suggestions else ["No specific improvements needed; keep monitoring."]
