import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib
from model import prepare_features

df = pd.read_csv("data/telco_churn.csv")

X, y = prepare_features(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

joblib.dump(model, "models/churn_model.pkl")

print("Model trained and saved with feature engineering!")