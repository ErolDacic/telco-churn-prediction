import pandas as pd
from sklearn.preprocessing import LabelEncoder

def prepare_features(df):
    df_model = df.copy()

    df_model["Churn"] = df_model["Churn"].map({"Yes": 1, "No": 0})

    df_model["TotalCharges"] = pd.to_numeric(df_model["TotalCharges"], errors="coerce")
    df_model["TotalCharges"] = df_model["TotalCharges"].fillna(0)

    df_model["AvgMonthlySpend"] = df_model["TotalCharges"] / (df_model["tenure"] + 1)
    df_model["CustomerValue"] = df_model["MonthlyCharges"] * df_model["tenure"]
    df_model["ShortTenure"] = (df_model["tenure"] <= 12).astype(int)
    df_model["HighMonthlyCharges"] = (df_model["MonthlyCharges"] > df_model["MonthlyCharges"].median()).astype(int)

    for col in df_model.select_dtypes(include=["object"]).columns:
        if col != "customerID":
            df_model[col] = LabelEncoder().fit_transform(df_model[col].astype(str))

    X = df_model.drop(["Churn", "customerID"], axis=1)
    y = df_model["Churn"]

    return X, y