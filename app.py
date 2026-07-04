"""
app.py — Flask API for CGM glucose prediction (standalone XGBoost).

Serves the 3 XGBoost models (30/60/120-min horizons) trained by your
standalone XGBoost script. Deployed on Render as a web service.

Endpoints:
  GET  /health        -> simple liveness check
  POST /predict        -> body: {"window": [[...18 features...], x36 rows]}
                           returns predicted glucose (mg/dL) for h1/h2/h3
                           + Clarke Error Grid zone (if "reference" given)
"""

import os
import joblib
import numpy as np
from flask import Flask, request, jsonify

# ==========================================
# PATHS (relative to this file — must match
# the folder structure you deploy to Render)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "xgboost_standalone_results")
DATA_DIR = os.path.join(BASE_DIR, "cgmacros_cleaned")

MODELS_PATH = os.path.join(MODEL_DIR, "xgb_models.pkl")
SCALER_PATH = os.path.join(DATA_DIR, "glucose_scaler.pkl")

EXPECTED_TIMESTEPS = 36   # window length used in training
EXPECTED_FEATURES = 18    # feature count per timestep used in training

app = Flask(__name__)

# ==========================================
# LOAD MODELS + SCALER ONCE AT STARTUP
# ==========================================
print("Loading models from:", MODELS_PATH)
models = joblib.load(MODELS_PATH)   # list of 3 XGBRegressor objects (h1, h2, h3)

print("Loading glucose scaler from:", SCALER_PATH)
scaler = joblib.load(SCALER_PATH)

HORIZON_LABELS = ["30min", "60min", "120min"]

print(f"✅ Loaded {len(models)} models. Ready to serve predictions.")


# ==========================================
# CLARKE ERROR GRID (single point) — same
# logic as your training script
# ==========================================
def clarke_zone_single(ref, pred):
    if (ref <= 70 and pred <= 70) or (0.8 * ref <= pred <= 1.2 * ref):
        return "A"
    if (ref >= 180 and pred <= 70) or (ref <= 70 and pred >= 180):
        return "E"
    if ((70 <= ref <= 290) and pred >= ref + 110) or \
       ((130 <= ref <= 180) and pred <= (7 / 5) * ref - 182):
        return "C"
    if (ref >= 240 and 70 <= pred <= 180) or \
       (ref <= 175 / 3 and 70 <= pred <= 180) or \
       ((175 / 3 <= ref <= 70) and pred >= (6 / 5) * ref):
        return "D"
    return "B"


def risk_label(glucose_mgdl, hypo_th=70, hyper_th=180):
    if glucose_mgdl < hypo_th:
        return "Hypoglycemia"
    elif glucose_mgdl > hyper_th:
        return "Hyperglycemia"
    return "Normal"


# ==========================================
# HEALTH CHECK (Render uses this to confirm
# the service is alive)
# ==========================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "models_loaded": len(models)}), 200


# ==========================================
# PREDICT ENDPOINT
# ==========================================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    if not payload or "window" not in payload:
        return jsonify({
            "error": "Missing 'window' field.",
            "expected_shape": [EXPECTED_TIMESTEPS, EXPECTED_FEATURES]
        }), 400

    window = np.array(payload["window"], dtype=float)

    if window.shape != (EXPECTED_TIMESTEPS, EXPECTED_FEATURES):
        return jsonify({
            "error": f"'window' must have shape "
                     f"({EXPECTED_TIMESTEPS}, {EXPECTED_FEATURES}), "
                     f"got {list(window.shape)}"
        }), 400

    X_flat = window.reshape(1, -1)  # (1, 36*18)

    predictions_norm = []
    for m in models:
        predictions_norm.append(m.predict(X_flat)[0])

    predictions_norm = np.array(predictions_norm).reshape(-1, 1)
    predictions_mgdl = scaler.inverse_transform(predictions_norm).flatten()

    response = {"predictions": {}}
    reference = payload.get("reference")  # optional true values for CEG

    for i, label in enumerate(HORIZON_LABELS):
        pred_val = float(predictions_mgdl[i])
        entry = {
            "glucose_mgdl": round(pred_val, 2),
            "risk": risk_label(pred_val)
        }
        if reference and len(reference) == 3:
            ref_val = float(reference[i])
            entry["clarke_zone"] = clarke_zone_single(ref_val, pred_val)
        response["predictions"][label] = entry

    return jsonify(response), 200


# ==========================================
# LOCAL DEV ENTRY POINT
# (Render uses gunicorn instead — see Procfile)
# ==========================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
