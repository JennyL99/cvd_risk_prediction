# cvd_risk_prediction.py
import streamlit as st
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="CVD Risk Prediction", layout="centered")

# -------------------------------
# Load model (if available)
# -------------------------------
MODEL_PATH = "model_LR_tuned_optuna_calibrated.pkl"
try:
    model = joblib.load(MODEL_PATH)
    st.sidebar.success("Model loaded successfully.")
except Exception:
    model = None
    st.sidebar.warning("⚠️ Model not found. Using example coefficients for demonstration.")

# Example fallback coefficients and intercept
example_coefficients = {
    'SBP': 0.1448, 'TG': 0.0315, 'WBC': 0.0659, 'BMI': 0.0256,
    'Hypertension': 0.1309, 'Dyslipidemia': 0.1399, 'Multimorbidity': 0.1841,
    'Bodily pains': 0.1569, 'Famine Exposure': 0.2030
}
intercept = -1.5

# If model exists, extract coefficients
if model is not None and hasattr(model, "coef_"):
    try:
        intercept = model.intercept_[0]
        example_coefficients = dict(zip(model.feature_names_in_, model.coef_[0]))
    except Exception:
        pass

# -------------------------------
# Sidebar Info
# -------------------------------
st.sidebar.title("About this app")
st.sidebar.info(
    "This web app estimates the risk of cardiovascular disease (CVD) "
    "based on a logistic regression model and displays the SHAP force plot."
)

# -------------------------------
# Main Layout
# -------------------------------
st.title("CVD Risk Prediction")

col1, col2 = st.columns(2)

# -------------------------------
# Column 1: Birth Year and numerical inputs
# -------------------------------
with col1:
    birth_year = st.number_input("Birth Year", min_value=1900, max_value=2025, value=1960, step=1)

    sbp = st.number_input("Systolic BP (mmHg)", min_value=80, max_value=200, value=120, step=1)
    tg = st.number_input("Triglycerides (mg/dL)", min_value=50, max_value=500, value=150, step=1)
    wbc = st.number_input("White Blood Cell (×10⁹/L)", min_value=2.0, max_value=20.0, value=6.0, step=0.1)
    bmi = st.number_input("Body Mass Index (kg/m²)", min_value=15.0, max_value=40.0, value=22.0, step=0.1)

# -------------------------------
# Determine famine exposure group automatically
# -------------------------------
def categorize_famine_exposure(year):
    if year > 1963:
        return 1, "No-exposed group (birth after 1963-01-01)"
    elif 1959 <= year <= 1962:
        return 2, "Fetal-exposed group (birth between 1959-01-01 and 1962-12-31)"
    elif 1949 <= year <= 1958:
        return 3, "Childhood-exposed group (birth between 1949-01-01 and 1958-12-31)"
    else:
        return 4, "Adolescence/Adult-exposed group (birth before 1948-12-31)"

famine_exposure, famine_text = categorize_famine_exposure(birth_year)

# -------------------------------
# Column 2: categorical inputs
# -------------------------------
with col2:
    hypertension = st.selectbox("Hypertension (0=No, 1=Yes)", [0, 1])
    dyslipidemia = st.selectbox("Dyslipidemia (0=No, 1=Yes)", [0, 1])
    multimorbidity = st.selectbox("Multimorbidity (0=No, 1=Yes)", [0, 1])
    bodily_pains = st.selectbox("Bodily pains (0=No, 1=Yes)", [0, 1])
    st.markdown(f"**Famine exposure group:**  \n{famine_text}")

# -------------------------------
# Calculate standardized and predicted risk
# -------------------------------
numerical_stats = {'SBP': {'mean': 135, 'std': 20}, 'TG': {'mean': 150, 'std': 80},
                   'WBC': {'mean': 6.5, 'std': 2.0}, 'BMI': {'mean': 24, 'std': 4}}

def standardize(feature, value):
    if feature in numerical_stats:
        mean = numerical_stats[feature]['mean']
        std = numerical_stats[feature]['std']
        return (value - mean) / std if std != 0 else value
    return value

if st.button("Calculate Risk"):
    # Input features dictionary
    inputs = {
        'SBP': sbp,
        'TG': tg,
        'WBC': wbc,
        'BMI': bmi,
        'Hypertension': hypertension,
        'Dyslipidemia': dyslipidemia,
        'Multimorbidity': multimorbidity,
        'Bodily pains': bodily_pains,
        'Famine Exposure': famine_exposure
    }

    # Standardize values
    X_std = np.array([standardize(k, v) for k, v in inputs.items()]).reshape(1, -1)
    coef_array = np.array([example_coefficients.get(k, 0) for k in inputs.keys()])
    lp = intercept + np.dot(X_std, coef_array)
    p = 1 / (1 + np.exp(-lp))
    risk = float(p[0])

    st.markdown(f"### 🩺 Estimated CVD Risk: **{risk * 100:.1f}%**")

    if risk < 0.10:
        st.success("Low risk. Keep maintaining a healthy lifestyle.")
    elif risk < 0.30:
        st.warning("Moderate risk. Consider regular cardiovascular checkups.")
    else:
        st.error("High risk. Please consult a doctor for detailed evaluation.")

    # -------------------------------
    # SHAP force plot (if model available)
    # -------------------------------
    try:
        if model is not None:
            explainer = shap.LinearExplainer(model, np.zeros((1, len(inputs))))
            shap_values = explainer(np.array(list(inputs.values())).reshape(1, -1))
            st.subheader("SHAP Force Plot")
            shap.plots.force(shap_values[0], matplotlib=True, show=False)
            st.pyplot(bbox_inches='tight')
        else:
            st.info("SHAP visualization is not available without the trained model file.")
    except Exception as e:
        st.warning(f"Unable to display SHAP plot: {e}")
