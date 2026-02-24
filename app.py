from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
try:
    model = joblib.load("phishing_model.pkl")
except Exception as e:
    model = None
    print("Error loading model:", e)
# Feature names in correct order
feature_names = [
    "SFH",
    "popUpWidnow",
    "SSLfinal_State",
    "Request_URL",
    "URL_of_Anchor",
    "web_traffic",
    "URL_Length",
    "age_of_domain",
    "having_IP_Address"
]

@app.route("/health")
def home():
    return "Phishing Detection API is running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
   data = request.json
if model is None:
    return jsonify({"error": "Model not loaded properly"}), 500

# Validate input
if not data:
    return jsonify({"error": "No input data provided"}), 400

missing_features = [f for f in feature_names if f not in data]
if missing_features:
    return jsonify({"error": f"Missing features: {missing_features}"}), 400

try:
    features = [data[feature] for feature in feature_names]
    features_array = np.array([features])
    prediction = model.predict(features_array)[0]
except Exception as e:
    return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
   app.run(host="0.0.0.0", port=5000)