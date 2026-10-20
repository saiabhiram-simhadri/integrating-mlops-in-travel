from flask import Flask, request, jsonify
import joblib
import pandas as pd
import sys
import os

app = Flask(__name__)

# Check environment variable first, then fallback to current working directory path
DEFAULT_PATH = os.path.join(os.getcwd(), "models", "flight_price_pipeline.joblib")
MODEL_PATH = os.environ.get("MODEL_PATH", DEFAULT_PATH)

try:
    artifact = joblib.load(MODEL_PATH)
    # Support both dictionary artifact payload and raw pipeline object
    if isinstance(artifact, dict):
        pipeline = artifact.get("pipeline") or artifact.get("model_pipeline")
        required_features = artifact.get("features", [])
    else:
        pipeline = artifact
        required_features = []
    print(f"[*] Successfully loaded flight price pipeline from {MODEL_PATH}")
except Exception as e:
    print(f"[!] Error loading model artifact: {e}", file=sys.stderr)
    pipeline = None
    required_features = []

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint to verify service and model status."""
    if pipeline is not None:
        return jsonify({"status": "healthy", "model_loaded": True}), 200
    return jsonify({"status": "unhealthy", "model_loaded": False}), 503

@app.route("/predict", methods=["POST"])
def predict():
    """Batch and single prediction endpoint for flight fares."""
    if pipeline is None:
        return jsonify({"error": "Model is not loaded on server"}), 503

    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "Empty or invalid JSON payload"}), 400

        # Support single object or list of objects
        records = data if isinstance(data, list) else [data]
        df_input = pd.DataFrame(records)

        # Dynamic feature derivations matching training pipeline
        if "route" not in df_input.columns:
            if "from" in df_input.columns and "to" in df_input.columns:
                df_input["route"] = df_input["from"].astype(str) + "_" + df_input["to"].astype(str)
            else:
                return jsonify({"error": "Missing 'from' and 'to' or 'route' in payload"}), 400

        # Handle optional temporal defaults if omitted
        if "flight_month" not in df_input.columns:
            df_input["flight_month"] = 6
        if "flight_dayofweek" not in df_input.columns:
            df_input["flight_dayofweek"] = 2
        if "is_weekend" not in df_input.columns:
            df_input["is_weekend"] = df_input["flight_dayofweek"].isin([5, 6]).astype(int)

        # Run inference
        raw_preds = pipeline.predict(df_input)
        predictions = [round(float(p), 2) for p in raw_preds]

        return jsonify({
            "status": "success",
            "count": len(predictions),
            "predictions": predictions
        }), 200

    except Exception as err:
        return jsonify({"error": str(err)}), 400

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "Flight Price Prediction API",
        "endpoints": {
            "health": "/health (GET)",
            "predict": "/predict (POST)"
        }
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)