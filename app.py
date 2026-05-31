import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px
from model import prepare_features

st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="📉",
    layout="wide"
)

st.title("📉 Telco Customer Churn Prediction")
st.write("Dashboard za predikciju korisnika koji su u riziku da napuste telekom kompaniju.")

# Load data and model
df = pd.read_csv("data/telco_churn.csv")
model = joblib.load("models/churn_model.pkl")

# Prepare data with feature engineering
X, y = prepare_features(df)

# Predictions
df["Churn_Risk"] = model.predict_proba(X)[:, 1]
df["Churn_Risk_%"] = (df["Churn_Risk"] * 100).round(2)
df["Expected_Loss"] = df["Churn_Risk"] * df["MonthlyCharges"]

# Sidebar filters
st.sidebar.header("Filteri")

contract_filter = st.sidebar.multiselect(
    "Tip ugovora",
    options=df["Contract"].unique(),
    default=df["Contract"].unique()
)

risk_threshold = st.sidebar.slider(
    "Minimalni churn rizik (%)",
    0,
    100,
    50
)

filtered_df = df[
    (df["Contract"].isin(contract_filter)) &
    (df["Churn_Risk_%"] >= risk_threshold)
].copy()

filtered_df = filtered_df.sort_values(by="Expected_Loss", ascending=False)

# Metrics
col1, col2, col3 = st.columns(3)

col1.metric("Ukupno korisnika", len(df))
col2.metric("Rizični korisnici", len(filtered_df))
col3.metric("Očekivani mjesečni gubitak", f"{filtered_df['Expected_Loss'].sum():.2f} €")

st.subheader("Top 10 najrizičnijih korisnika")

top10 = filtered_df.nlargest(10, "Churn_Risk_%")

fig = px.bar(
    top10,
    x="customerID",
    y="Churn_Risk_%",
    title="Top 10 korisnika sa najvećim rizikom odlaska"
)

st.plotly_chart(fig, use_container_width=True)


st.subheader("Rangirana lista rizičnih korisnika")

st.dataframe(
    filtered_df[
        [
            "customerID",
            "gender",
            "SeniorCitizen",
            "Contract",
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "Churn_Risk_%",
            "Expected_Loss"
        ]
    ],
    use_container_width=True
)

# Download CSV

csv = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Preuzmi listu rizičnih korisnika (CSV)",
    data=csv,
    file_name="high_risk_customers.csv",
    mime="text/csv"
)

st.subheader("Detalji korisnika")

if len(filtered_df) == 0:
    st.warning("Nema korisnika koji odgovaraju izabranim filterima.")
else:
    selected_customer = st.selectbox(
        "Izaberi korisnika",
        filtered_df["customerID"]
    )

    customer = filtered_df[filtered_df["customerID"] == selected_customer].iloc[0]
    customer_index = df[df["customerID"] == selected_customer].index[0]

    st.write("### Profil korisnika")
    st.write(customer)

    st.write("### Preporučena retention akcija")

    if customer["Churn_Risk_%"] >= 80:
        st.error("Visok rizik: ponuditi 20% popusta i hitan kontakt call centra.")
    elif customer["Churn_Risk_%"] >= 60:
        st.warning("Srednje visok rizik: ponuditi bolji paket ili bonus uslugu.")
    else:
        st.info("Umjeren rizik: poslati personalizovanu ponudu ili loyalty benefit.")

    st.write("### SHAP objašnjenje predikcije")
    st.write("Ovaj graf pokazuje koje karakteristike najviše povećavaju ili smanjuju rizik odlaska izabranog korisnika.")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_values[customer_index], max_display=10, show=False)
    st.pyplot(fig)

    st.subheader("Survival Analysis")

st.write(
    "Kaplan-Meier kriva prikazuje vjerovatnoću da korisnik ostane aktivan kroz vrijeme."
)

st.image("models/survival_curve.png", caption="Customer Survival Curve")


st.subheader("ROI Simulator retention kampanje")

discount_percent = st.slider("Popust za retention kampanju (%)", 0, 50, 20)

target_customers = filtered_df[filtered_df["Churn_Risk_%"] >= 60].copy()

expected_saved_rate = discount_percent / 100
campaign_cost = (target_customers["MonthlyCharges"] * expected_saved_rate).sum()
expected_revenue_saved = (target_customers["MonthlyCharges"] * target_customers["Churn_Risk"] * 6).sum()
roi = ((expected_revenue_saved - campaign_cost) / campaign_cost * 100) if campaign_cost > 0 else 0

col4, col5, col6 = st.columns(3)

col4.metric("Targetirani korisnici", len(target_customers))
col5.metric("Trošak kampanje", f"{campaign_cost:.2f} €")
col6.metric("Očekivani ROI", f"{roi:.2f}%")

st.write(
    "ROI je procijenjen na osnovu očekivanog zadržanog prihoda u narednih 6 mjeseci i troška ponuđenog popusta."
)