# app.py
from flask import Flask, request, jsonify
import joblib, numpy as np, os

app = Flask(__name__)
model = joblib.load("logistic_regression.pkl")  # Load trained model

@app.route('/')
def home():
    return jsonify({
        "message": "ML Model API is running!",
        "status": "ok"
    })


@app.post("/predict")
def predict():
    data = request.get_json(force=True)
    if "features" not in data:
        return jsonify({"error": "JSON must include 'features'"}), 400
    X = np.array(data["features"]).reshape(1, -1)
    y = model.predict(X).tolist()
    return jsonify({"prediction": y})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Needed for Render
    app.run(host="0.0.0.0", port=port)
