from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("phishing_model.pkl")

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

@app.route("/")
def home():
    return "Phishing Detection API is running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    
    features = [data[feature] for feature in feature_names]
    features_array = np.array([features])
    
    prediction = model.predict(features_array)[0]
    
    result = "Phishing Website" if prediction == 1 else "Legitimate Website"
    
    return jsonify({
        "prediction": int(prediction),
        "result": result
    })

if __name__ == "__main__":
   app.run(host="0.0.0.0", port=5000)