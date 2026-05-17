# This step involves preprocessing of the data:

# Importing the libraries:
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_and_preprocess(path="data/LungCap.csv"):
    lung_df = pd.read_csv(path)

    # check for null data 
    print(f"Null values:", lung_df.isnull().sum())

    #check for duplicated 
    print(f"Duplicated values: ", lung_df.duplicated().sum())

    #print the statistical properties
    lung_df.describe().to_csv("data_quality_report.csv")

    X = lung_df.iloc[:, 1:].values   # Age, Height, Smoke, Gender
    y = lung_df.iloc[:, 0].values    # LungCap

    # Encode Smoke using Labelencoder (yes/no)
    le = LabelEncoder()
    X[:, 2] = le.fit_transform(X[:, 2])

    # One-hot encoding the Gender
    ct = ColumnTransformer(
        transformers=[("gender", OneHotEncoder(), [3])],
        remainder="passthrough"
    )

    X = ct.fit_transform(X)

    return X, y