import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from datetime import datetime

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("models/model.pkl", "rb"))
model_info = pickle.load(open("models/model_info.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])

def predict():
    # Get form data

    try:
        Age = float(request.form.get("Age"))
        Height = float(request.form.get("Height"))
    except:
        return "Invalid numeric input"

    Smoke = 1 if request.form.get("Smoke", "").lower() == "yes" else 0

    Gender = request.form.get("Gender", "").lower()

# One-hot encoding manually
    if Gender == "male":
        gender_female = 0
        gender_male = 1
    else:
        gender_female = 1
        gender_male = 0


    # Prepare features for prediction
   
    features = pd.DataFrame([[gender_female, gender_male, Age, Height, Smoke]])

    # Predict charges
    prediction = model.predict(features)

    #  Format to float and 2 decimal places
    #formatted_prediction = f"The predicted value is {round(float(prediction[0]), 2)}"

    formatted_prediction = (
        f"The predicted value is {round(float(prediction[0]), 2)} | "
        f"Best Model: {model_info['name']} | "
        f"R2 Score: {round(model_info['r2'], 3)}")

    with open("prediction_logs.txt", "a") as f:
        f.write(
            f"{datetime.now()} | "
            f"Age={Age}, Height={Height}, Smoke={Smoke}, "
            f"Gender={Gender}, Prediction={prediction[0]}\n"
                )

    return render_template("result.html", prediction=formatted_prediction)



if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0')