# Telco Customer Churn Prediction System

Ovaj projekat predstavlja web aplikaciju za predikciju odlaska korisnika telekom kompanije. Sistem koristi mašinsko učenje za identifikaciju korisnika koji su u riziku od churn-a i pomaže retention timu da donese bolje poslovne odluke.

## Cilj projekta

Cilj je razviti alat koji:
- predviđa koji korisnici su u riziku da napuste kompaniju,
- prikazuje rangiranu listu rizičnih korisnika,
- objašnjava razloge iza svake predikcije,
- predlaže retention akcije,
- računa očekivani ROI retention kampanje.

## Korišćeni dataset

Korišćen je Telco Customer Churn Dataset. Dataset sadrži podatke o korisnicima, tipu ugovora, trajanju korišćenja usluga, mjesečnim troškovima, ukupnim troškovima i informaciju da li je korisnik napustio kompaniju.

## Tehnologije

- Python
- Pandas
- Scikit-learn
- XGBoost
- SHAP
- Lifelines
- Streamlit
- Plotly
- Matplotlib

## Funkcionalnosti aplikacije

- XGBoost model za predikciju churn-a
- rangirana lista rizičnih korisnika
- prioritizacija po očekivanom finansijskom gubitku
- filteri po tipu ugovora i nivou rizika
- SHAP waterfall objašnjenje pojedinačne predikcije
- Kaplan-Meier survival analysis
- ROI simulator retention kampanje
- preporučene akcije za korisnike
- export rizičnih korisnika u CSV fajl

## Struktura projekta

```text
Churn_Projekat/
│
├── app.py
├── train.py
├── survival.py
├── requirements.txt
├── README.md
│
├── data/
│   └── telco_churn.csv
│
└── models/
    ├── churn_model.pkl
    └── survival_curve.png