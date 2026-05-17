import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("models/model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])

def predict():
    # Get form data
    # Age = float(request.form.get("Age"))
    # Height = float(request.form.get("Height"))

    try:
        Age = float(request.form.get("Age"))
        Height = float(request.form.get("Height"))
    except:
        return "Invalid numeric input"

    Smoke = 1 if request.form.get("Smoke", "").lower() == "yes" else 0
    Gender = 1 if request.form.get("Gender", "").lower() == "male" else 0

    # Prepare features for prediction
   
    feature_names = ["Age", "Height", "Smoke", "Gender"]
    features = pd.DataFrame([[Age, Height, Smoke, Gender]], columns=feature_names)

    # Predict charges
    prediction = model.predict(features)

    #  Format to float and 2 decimal places
    formatted_prediction = f"The predicted value is ${round(float(prediction[0]), 2)}"


    return render_template("result.html", prediction=formatted_prediction)


if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0')