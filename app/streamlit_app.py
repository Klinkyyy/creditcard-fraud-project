# Om dit script te laten werken moet je in cmd prompt dit schrijven :
# cd C:\Users\DASEE\Desktop\creditcard-fraud-project
# streamlit run app/streamlit_app.py

# ---------------------------
# 📦 BENODIGDE MODULES
# ---------------------------

import os                 # Voor het werken met paden en mappen (zoals modelpaden dynamisch ophalen)
import joblib             # Voor het laden van het opgeslagen machine learning model (.pkl-bestand)
import pandas as pd       # Voor het maken en verwerken van invoergegevens als DataFrame
import streamlit as st    # Streamlit is het framework waarmee we de webinterface bouwen

# ---------------------------
# 📁 MODEL LADEN
# ---------------------------

# Bepaal het juiste pad naar het modelbestand, ongeacht vanaf waar de app gestart wordt
# __file__ = pad naar dit script zelf
# ".." = ga naar de bovenliggende map → "models/fraud_model_xgb.pkl"
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "fraud_model_xgb.pkl")

# Laad het opgeslagen XGBoost-model
model = joblib.load(model_path)

# ---------------------------
# 🖥️ STREAMLIT INTERFACE
# ---------------------------

# Stel de paginatitel en layout in
st.set_page_config(page_title="Fraude Detector")

# Titel bovenin de app
st.title("💳 Creditcard Fraud Detector")

# ---------------------------
# 👤 GEBRUIKERSINVOER
# ---------------------------

# 1. Transactiebedrag invoeren (float, tussen 0 en 5000 euro)
amount = st.number_input("Transactiebedrag (€)", min_value=0.0, max_value=5000.0, value=50.0)

# 2. Tijd sinds eerste transactie (in seconden)
time = st.number_input("Seconden sinds eerste transactie", min_value=0, max_value=200000, value=10000)

# 3. V1 t/m V28: de 28 PCA-componenten als schuifjes (default op 0.0)
pca_inputs = [
    st.slider(f"V{i}", -30.0, 30.0, 0.0, step=0.1) for i in range(1, 29)
]

# ---------------------------
# 🧮 DATAFRAME MAKEN VOOR MODEL
# ---------------------------

# Combineer alle inputs in één rij en zet ze in een pandas DataFrame met juiste kolomnamen
# Zet kolommen in dezelfde volgorde als bij modeltraining: Time, V1–V28, Amount
all_features = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
input_df = pd.DataFrame([[time] + pca_inputs + [amount]], columns=all_features)

# ---------------------------
# 🔍 MODELVOORSPELLING
# ---------------------------

# Als gebruiker op de knop klikt, voer voorspelling uit
if st.button("Voorspel fraude"):

    # 1 = fraude, 0 = geen fraude
    prediction = model.predict(input_df)[0]

    # Verkrijg de waarschijnlijkheid dat het fraude is (tussen 0 en 1)
    proba = model.predict_proba(input_df)[0][1]

    # Toon resultaat met vertrouwen
    if prediction == 1:
        st.error(f"⚠️ Fraude gedetecteerd! Vertrouwen: {proba:.2%}")
    else:
        st.success(f"✅ Geen fraude. Vertrouwen: {proba:.2%}")
