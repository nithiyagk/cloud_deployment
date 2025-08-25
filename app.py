from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Make sure the path works on Render
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
model = joblib.load(model_path)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        value = float(request.form["value"])
        prediction = model.predict([[value]])[0]
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
