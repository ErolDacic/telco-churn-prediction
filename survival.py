import pandas as pd
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/telco_churn.csv")

# Event
df["Churn_Flag"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

# Time variable
T = df["tenure"]

# Event occurred
E = df["Churn_Flag"]

kmf = KaplanMeierFitter()

kmf.fit(T, E)

plt.figure(figsize=(10, 6))
kmf.plot_survival_function()

plt.title("Customer Survival Curve")
plt.xlabel("Months")
plt.ylabel("Probability of Remaining Customer")

plt.savefig("models/survival_curve.png")

print("Survival analysis completed.")