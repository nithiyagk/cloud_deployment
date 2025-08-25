from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Load model and scaler safely
BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        try:
            # Get all 3 features from the form
            f1 = float(request.form["Feature1"])
            f2 = float(request.form["Feature2"])
            f3 = float(request.form["Feature3"])
            
            # Scale the features
            features_scaled = scaler.transform([[f1, f2, f3]])
            
            # Predict
            prediction = model.predict(features_scaled)[0]
        except Exception as e:
            prediction = f"Error: {e}"
    
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
