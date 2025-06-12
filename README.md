# 💳 Creditcard Fraud Detection

Een end-to-end machine learning project voor het detecteren van frauduleuze creditcardtransacties.

## 🔍 Projectbeschrijving

In dit project wordt een machine learning-model getraind om frauduleuze transacties te herkennen. We gebruiken een ongebalanceerde dataset en passen verschillende technieken toe om dit probleem realistisch aan te pakken.

**Belangrijkste doelen:**
- Exploratory Data Analysis (EDA)
- Trainen en vergelijken van modellen (Logistic Regression vs XGBoost)
- Evaluatie met precision, recall, F1-score en ROC-curve
- Opslaan van het beste model
- (optioneel) Live demo via Streamlit

---

## 🗃️ Dataset

- 📂 Bron: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 📏 Oorspronkelijke grootte: 284.807 transacties
- 🧪 Gebruikte subset: alle frauduleuze transacties + 5000 willekeurige normale

---

## 🔧 Gebruikte technologieën

- Python (pandas, matplotlib, seaborn)
- Scikit-learn (modeltraining & evaluatie)
- XGBoost (geavanceerde classifier)
- Streamlit (webinterface voor demo)
- Joblib (model opslaan)
- Jupyter Notebook / VS Code

---

## 📈 Resultaten

- Logistic Regression:  
  - F1-score fraude: **0.90**
  - ROC-AUC: **0.98**

- XGBoost:  
  - F1-score fraude: **0.91**
  - ROC-AUC: **0.99**

> 🏆 XGBoost presteerde iets beter en is gekozen als eindmodel


with this code i upload the files to the github dictonary :
git remote add origin https://github.com/Klinkyyy/creditcard-fraud-project.git
git branch -M main
git push -u origin main
---

## 🚀 Live demo (optioneel)

Start de Streamlit-app lokaal:

```bash
streamlit run app/streamlit_app.py
