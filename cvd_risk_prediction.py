import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="CVD Risk Prediction", layout="centered")

# ----------------------------
# Language toggle (top-right)
# ----------------------------
if "lang" not in st.session_state:
    st.session_state.lang = "en"
col_left, col_right = st.columns([0.82, 0.18])
with col_right:
    if st.session_state.lang == "en":
        if st.button("中文"):
            st.session_state.lang = "cn"
    else:
        if st.button("English"):
            st.session_state.lang = "en"
lang = st.session_state.lang

# ----------------------------
# Text dictionary
# ----------------------------
TEXT = {
    "en": {
        "title": "CVD Risk Prediction",
        "intro": "This web app estimates the risk of cardiovascular disease (CVD) "
                 "based on a logistic regression model and displays the SHAP force plot.",
        "birth": "Birth Year",
        "sbp": "Systolic BP (mmHg)",
        "tg": "Triglycerides (mg/dL)",
        "wbc": "White Blood Cell (×10⁹/L)",
        "bmi": "Body Mass Index (kg/m²)",
        "htn": "Hypertension (0=No, 1=Yes)",
        "dys": "Dyslipidemia (0=No, 1=Yes)",
        "multi": "Multimorbidity (0=No, 1=Yes)",
        "pain": "Bodily pains (0=No, 1=Yes)",
        "famine": "Famine Exposure (auto detected)",
        "auto": "Automatically detected based on birth year:",
        "predict": "Predict",
        "low": "Low risk. Keep maintaining a healthy lifestyle.",
        "mid": "Moderate risk. Consider regular cardiovascular checkups.",
        "high": "High risk. Please consult a doctor for detailed evaluation.",
    },
    "cn": {
        "title": "心血管疾病风险预测",
        "intro": "本网页基于逻辑回归模型估计心血管疾病（CVD）发生风险，并显示 SHAP 力图。",
        "birth": "出生年份",
        "sbp": "收缩压（mmHg）",
        "tg": "甘油三酯（mg/dL）",
        "wbc": "白细胞（×10⁹/L）",
        "bmi": "体重指数（kg/m²）",
        "htn": "高血压（0=否，1=是）",
        "dys": "血脂异常（0=否，1=是）",
        "multi": "共病（0=否，1=是）",
        "pain": "身体疼痛（0=否，1=是）",
        "famine": "饥荒暴露（自动识别）",
        "auto": "根据出生年份自动识别：",
        "predict": "预测",
        "low": "低风险：保持健康的生活方式。",
        "mid": "中风险：建议定期进行心血管检查。",
        "high": "高风险：建议尽快就医评估。",
    }
}
T = TEXT[lang]

# ----------------------------
# Model loading
# ----------------------------
MODEL_FILE = "model_LR_tuned_optuna_calibrated.pkl"
EXAMPLE_INTERCEPT = -1.5
EXAMPLE_COEFS = {
    'SBP': 0.1448, 'TG': 0.0315, 'WBC': 0.0659, 'BMI': 0.0256,
    'Hypertension': 0.1309, 'Dyslipidemia': 0.1399,
    'Multimorbidity': 0.1841, 'Bodily pains': 0.1569,
    'Famine Exposure': 0.2030
}
NUMERICAL_FEATURES = ['SBP', 'TG', 'WBC', 'BMI']
CATEGORICAL_FEATURES = ['Hypertension', 'Dyslipidemia', 'Multimorbidity', 'Bodily pains', 'Famine Exposure']
FEATURE_ORDER = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
NUM_STATS = {'SBP': {'mean': 135.0, 'std': 20.0},
             'TG': {'mean': 150.0, 'std': 80.0},
             'WBC': {'mean': 6.5, 'std': 2.0},
             'BMI': {'mean': 24.0, 'std': 4.0}}

def load_model_params():
    if os.path.exists(MODEL_FILE):
        try:
            model = joblib.load(MODEL_FILE)
            if hasattr(model, "base_estimator_"):
                model = model.base_estimator_
            if hasattr(model, "intercept_") and hasattr(model, "coef_"):
                intercept = float(model.intercept_[0])
                features = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else FEATURE_ORDER
                coefs = {f: float(c) for f, c in zip(features, model.coef_[0])}
                return intercept, coefs
        except Exception as e:
            st.warning(f"Model load failed: {e}")
    return EXAMPLE_INTERCEPT, EXAMPLE_COEFS.copy()

intercept_val, coefs = load_model_params()

# ----------------------------
# UI layout
# ----------------------------
st.markdown(f"<h1 style='text-align:center'>{T['title']}</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align:center; color:#444'>{T['intro']}</p>", unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns(2)

def categorize_famine(year):
    if year > 1963:
        return 1, "No-exposed group (birth after 1963-01-01)"
    elif 1959 <= year <= 1962:
        return 2, "Fetal-exposed group (1959–1962)"
    elif 1949 <= year <= 1958:
        return 3, "Childhood-exposed group (1949–1958)"
    else:
        return 4, "Adolescent/Adult-exposed group (≤1948)"

with col1:
    birth_year = st.number_input(T["birth"], min_value=1900, max_value=2025, value=1960, step=1)
    sbp = st.number_input(T["sbp"], value=120.0)
    tg = st.number_input(T["tg"], value=150.0)
    wbc = st.number_input(T["wbc"], value=6.0)
    bmi = st.number_input(T["bmi"], value=22.0)

with col2:
    hypertension = st.selectbox(T["htn"], [0, 1])
    dyslipidemia = st.selectbox(T["dys"], [0, 1])
    multimorbidity = st.selectbox(T["multi"], [0, 1])
    bodily_pains = st.selectbox(T["pain"], [0, 1])
    famine_exposure, famine_text = categorize_famine(birth_year)
    famine_display = st.selectbox(T["famine"], ["1 - No-exposed", "2 - Fetal", "3 - Childhood", "4 - Adolescent/Adult"], index=famine_exposure - 1)
    st.caption(f"{T['auto']} {famine_text}")

# ----------------------------
# Prediction and SHAP
# ----------------------------
if st.button(T["predict"]):
    input_dict = {'SBP': sbp, 'TG': tg, 'WBC': wbc, 'BMI': bmi,
                  'Hypertension': hypertension, 'Dyslipidemia': dyslipidemia,
                  'Multimorbidity': multimorbidity, 'Bodily pains': bodily_pains,
                  'Famine Exposure': famine_exposure}

    def standardize_single(f, v):
        if f in NUM_STATS:
            m, s = NUM_STATS[f]['mean'], NUM_STATS[f]['std']
            return (v - m) / s
        return v

    X_std = np.array([[standardize_single(f, v) for f, v in input_dict.items()]])
    coef_array = np.array([coefs.get(f, 0) for f in input_dict.keys()])
    lp = intercept_val + np.dot(X_std, coef_array)
    p = 1 / (1 + np.exp(-lp))
    risk = float(p[0])

    st.markdown(f"<h3 style='text-align:center'>🩺 Risk Probability: <b>{risk*100:.1f}%</b></h3>", unsafe_allow_html=True)
    if risk < 0.1:
        st.success(T["low"])
    elif risk < 0.3:
        st.warning(T["mid"])
    else:
        st.error(T["high"])

# ----------------------------
# SHAP force plot generation
# ----------------------------
    st.subheader("SHAP-style force plot")
    try:
        background = np.zeros((1, len(FEATURE_ORDER)))
        class SkModel:
            def __init__(self, intercept, coef):
                self.intercept_ = np.array([intercept])
                self.coef_ = np.array([coef])
        coef_list = [coefs.get(f, 0.0) for f in FEATURE_ORDER]
        skm = SkModel(intercept_val, coef_list)
        x_standardized = np.array([[standardize_single(f, input_dict[f]) for f in FEATURE_ORDER]])
        explainer = shap.LinearExplainer(skm, background, feature_perturbation="interventional")
        shap_vals = explainer.shap_values(x_standardized)
        shap_vals_arr = np.array(shap_vals).reshape(1, -1)[0]
        plt.figure(figsize=(10, 2))
        shap.force_plot(explainer.expected_value, shap_vals_arr, x_standardized[0],
                        feature_names=FEATURE_ORDER, matplotlib=True, show=False)
        fig = plt.gcf()
        st.pyplot(fig)
    except Exception as e:
        st.write("Could not create SHAP force plot automatically. Falling back to a bar chart of contributions.")
        st.write(e)
        contribs = []
        for f in FEATURE_ORDER:
            stdv = standardize_single(f, input_dict[f])
            coefv = coefs.get(f, 0.0)
            contribs.append((f, coefv * stdv))
        contrib_df = pd.DataFrame(contribs, columns=['feature', 'contribution']).set_index('feature')
        fig, ax = plt.subplots(figsize=(8, 4))
        contrib_df.sort_values('contribution', inplace=True)
        ax.barh(contrib_df.index, contrib_df['contribution'])
        ax.set_xlabel("Contribution (coef * standardized value)")
        st.pyplot(fig)
