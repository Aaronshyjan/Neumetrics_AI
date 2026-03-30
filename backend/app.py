import os
import time
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

import os

BASE_DIR_ROOT = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR_ROOT, "..", "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')
CORS(app)

@app.route("/")
def serve_frontend():
    return app.send_static_file("index.html")

# =====================================================
# 🔹 MONGODB CONNECTION (SAFE + ENVIRONMENT VARIABLE)
# =====================================================
logs_collection = None
predictions_collection = None

try:
    # Get URI from environment variable
    MONGO_URI = os.getenv("MONGO_URI")

    if not MONGO_URI:
        raise ValueError("MONGO_URI not found in environment variables")

    client = MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=5000,
        tls=True
    )

    # Test connection
    client.admin.command("ping")

    db = client["cognitive_load_db"]
    logs_collection = db["keystroke_logs"]
    predictions_collection = db["predictions"]

    print("✅ MongoDB connected successfully")

except Exception as e:
    print("❌ MongoDB connection failed:", e)

# =====================================================
# 🔹 LOAD ML MODEL & ACCURACY
# =====================================================
model = None
model_accuracy = 0.0

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "cognitive_model.pkl")
    acc_path = os.path.join(BASE_DIR, "accuracy_score.pkl")

    model = joblib.load(model_path)
    if os.path.exists(acc_path):
        model_accuracy = joblib.load(acc_path)
    
    print(f"✅ ML Model loaded (Accuracy: {model_accuracy*100:.2f}%)")

except Exception as e:
    print("❌ Model loading failed:", e)

# =====================================================
# 🔹 FEATURE EXTRACTION
# =====================================================
def extract_features(keystrokes):

    valid_keys = [
        k for k in keystrokes
        if isinstance(k, dict)
        and "dwell" in k
        and "flight" in k
        and "timestamp" in k
    ]

    if len(valid_keys) < 3:
        return None

    dwells = [k["dwell"] for k in valid_keys]
    flights = [k["flight"] for k in valid_keys]

    duration_ms = valid_keys[-1]["timestamp"] - valid_keys[0]["timestamp"]
    duration_s = max(duration_ms / 1000, 1)

    typing_speed = len(valid_keys) / duration_s

    return {
        "avg_dwell": float(np.mean(dwells)),
        "avg_flight": float(np.mean(flights)),
        "typing_speed": float(typing_speed),
    }

# =====================================================
# 🔹 ML PREDICTION
# =====================================================
def predict_load(features):

    if model is None or features is None:
        return "Waiting...", 0.0

    try:
        X = np.array([[
            features["avg_dwell"],
            features["avg_flight"],
            features["typing_speed"]
        ]])

        prediction = model.predict(X)[0]
        # Get confidence (max probability)
        probabilities = model.predict_proba(X)[0]
        confidence = float(np.max(probabilities))

        mapping = {
            0: "Low",
            1: "Medium",
            2: "High"
        }

        return mapping.get(prediction, "Medium"), confidence

    except Exception as e:
        print("Prediction error:", e)
        return "Waiting...", 0.0

# =====================================================
# 🔹 API ENDPOINT
# =====================================================
@app.route("/keystrokes", methods=["POST"])
def receive_keystrokes():

    try:
        if not request.is_json:
            return jsonify({"load": "Invalid Request"}), 400

        data = request.json.get("data", [])

        if not isinstance(data, list) or len(data) < 3:
            return jsonify({"load": "Waiting...", "accuracy": f"{model_accuracy*100:.1f}%"})

        features = extract_features(data)
        cognitive_load, confidence = predict_load(features)

        # Save safely (NO truth-value testing)
        if logs_collection is not None and predictions_collection is not None:
            try:
                logs_collection.insert_one({
                    "timestamp": time.time(),
                    "keystrokes": data
                })

                predictions_collection.insert_one({
                    "timestamp": time.time(),
                    "features": features,
                    "prediction": cognitive_load,
                    "confidence": confidence
                })
            except Exception as db_error:
                print("Database insert error:", db_error)

        return jsonify({
            "load": cognitive_load,
            "confidence": f"{confidence*100:.1f}%",
            "model_accuracy": f"{model_accuracy*100:.1f}%",
            "accuracy": f"{model_accuracy*100:.1f}%",
            "features": features
        })

    except Exception as e:
        print("❌ Endpoint crash:", e)
        return jsonify({"load": "Server Error"}), 500

# =====================================================
# 🔹 HEALTH CHECK
# =====================================================
@app.route("/health")
def health():
    return jsonify({"status": "Backend running"})

# =====================================================
# 🔹 RUN SERVER
# =====================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
