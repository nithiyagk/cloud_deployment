from flask import Flask, render_template, request
import joblib

app = Flask(__name__,template_folder="templates")

# Load your ML model
model = joblib.load("logistic_regression.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        value = float(request.form["value"])
        prediction = model.predict([[value]])[0]
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
