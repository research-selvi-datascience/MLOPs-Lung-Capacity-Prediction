# importing the libraries

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

from preprocess_data.preprocess import load_and_preprocess

# ML Flow tracking:
# mlflow.set_tracking_uri("http://127.0.0.1:5000")
# mlflow.set_experiment("lung-capacity-prediction")

# Loading the data
X, y = load_and_preprocess()

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the training data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Models:
models = {
    "LinearRegression": LinearRegression(),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "SVR": SVR(kernel="rbf", C=100),
    "RandomForest": RandomForestRegressor(n_estimators=100),
    "DecisionTree": DecisionTreeRegressor(max_depth=5),
    "ANN": MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=500, random_state=42)
    }

best_model = None
best_score = -999
best_name = ""

for name, model in models.items():

    with mlflow.start_run(run_name=name):

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        r2 = r2_score(y_test, preds)

        mlflow.log_param("model", name)
        mlflow.log_metric("r2_score", r2)

        print(f"{name} R2: {r2}")

        if r2 > best_score:
            best_score = r2
            best_model = model
            best_name = name

print("\nBEST MODEL:", best_name, "R2:", best_score)

os.makedirs("models", exist_ok=True)

pickle.dump(best_model, open("models/model.pkl", "wb"))
pickle.dump(scaler, open("models/scaler.pkl", "wb"))


best_model_info = {
    "name": best_name,
    "r2": best_score
}

pickle.dump(best_model_info, open("models/model_info.pkl", "wb"))