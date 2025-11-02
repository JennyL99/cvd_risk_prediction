# app.py
import streamlit as st
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import pandas as pd

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(page_title="CVD Risk Prediction", layout="centered")

# -------------------------------
# Language Toggle (Top-right)
# -------------------------------
if "lang" not in st.session_state:
    st.session_state.lang = "en"

# Place language button in top-right corner
col_lang, _ = st.columns([0.15, 0.85])
with col_lang:
    if st.session_state.lang == "en":
        if st.button("中文"):
            st.session_state.lang = "cn"
    else:
        if st.button("English"):
            st.session_state.lang = "en"

# -------------------------------
# Multilingual Text
# -------------------------------
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
        "shap": "SHAP Force Plot",
        "no_model": "SHAP visualization is not available without the trained model file."
    },
    "cn": {
        "title": "心血管疾病风险预测",
        "intro": "本网页基于逻辑回归模型估计心血管疾病（CVD）发生风险，并显示 SHAP 力图结果。",
        "birth": "出生年份",
        "sbp": "收缩压（mmHg）",
        "tg": "甘油三酯（mg/dL）",
        "wbc": "白细胞（×10⁹/L）",
        "bmi": "体重指数（kg/m²）",
        "htn": "高血压（0=否，1=是）",
        "dys": "血脂异常（0=否，1=是）",
        "multi": "多重疾病（0=否，1=是）",
        "pain": "身体疼痛（0=否，1=是）",
        "famine": "饥荒暴露（根据出生年份自动识别）",
        "auto": "根据出生年份自动识别：",
        "predict": "预测",
        "low": "低风险：建议继续保持健康的生活方式。",
        "mid": "中风险：建议定期进行心血管健康检查。",
        "high": "高风险：建议尽快就医进行评估。",
        "shap": "SHAP 力图",
        "no_model": "未检测到模型文件，无法显示 SHAP 可视化。"
    }
}

lang = st.session_state.lang
T = TEXT[lang]

# -------------------------------
# Load model
# -------------------------------
MODEL_PATH = "model_LR_tuned_optuna_calibrated.pkl"
try:
    model = joblib.load(MODEL_PATH)
except Exception:
    model = None

# Example fallback coefficients
example_coefficients = {
    'SBP': 0.1448, 'TG': 0.0315, 'WBC': 0.0659, 'BMI': 0.0256,
    'Hypertension': 0.1309, 'Dyslipidemia': 0.1399, 'Multimorbidity': 0.1841,
    'Bodily pains': 0.1569, 'Famine Exposure': 0.2030
}
intercept = -1.5

# Try to extract coefficients
if model is not None:
    try:
        if hasattr(model, "base_estimator_"):  # For calibrated models
            lr_model = model.base_estimator_
            intercept = lr_model.intercept_[0]
            example_coefficients = dict(zip(lr_model.feature_names_in_, lr_model.coef_[0]))
        elif hasattr(model, "coef_"):
            intercept = model.intercept_[0]
            example_coefficients = dict(zip(model.feature_names_in_, model.coef_[0]))
    except Exception:
        pass

# -------------------------------
# Title and Intro
# -------------------------------
st.markdown(f"<h1 style='text-align:center'>{T['title']}</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align:center; font-size:16px; color:#555;'>{T['intro']}</p>", unsafe_allow_html=True)
st.markdown("---")

# -------------------------------
# Layout
# -------------------------------
col1, col2 = st.columns(2)

# Column 1: numeric inputs
with col1:
    birth_year = st.number_input(T["birth"], min_value=1900, max_value=2025, value=1960, step=1)
    sbp = st.number_input(T["sbp"], min_value=80, max_value=200, value=120, step=1)
    tg = st.number_input(T["tg"], min_value=50, max_value=500, value=150, step=1)
    wbc = st.number_input(T["wbc"], min_value=2.0, max_value=20.0, value=6.0, step=0.1)
    bmi = st.number_input(T["bmi"], min_value=15.0, max_value=40.0, value=22.0, step=0.1)

# Determine famine exposure automatically
def categorize_famine_exposure(year):
    if year > 1963:
        return 1, "No-exposed group (birth after 1963-01-01)" if lang == "en" else "非暴露组（出生≥1963-01-01）"
    elif 1959 <= year <= 1962:
        return 2, "Fetal-exposed group (birth 1959–1962)" if lang == "en" else "胎儿期暴露组（1959–1962年出生）"
    elif 1949 <= year <= 1958:
        return 3, "Childhood-exposed group (birth 1949–1958)" if lang == "en" else "儿童期暴露组（1949–1958年出生）"
    else:
        return 4, "Adolescence/Adult-exposed group (birth ≤1948)" if lang == "en" else "青春期/成人暴露组（≤1948年出生）"

famine_exposure, famine_text = categorize_famine_exposure(birth_year)

# Column 2: categorical inputs
with col2:
    hypertension = st.selectbox(T["htn"], [0, 1])
    dyslipidemia = st.selectbox(T["dys"], [0, 1])
    multimorbidity = st.selectbox(T["multi"], [0, 1])
    bodily_pains = st.selectbox(T["pain"], [0, 1])
    famine_display = st.selectbox(
        T["famine"],
        ["1 - No-exposed", "2 - Fetal-exposed", "3 - Childhood-exposed", "4 - Adolescence/Adult-exposed"]
        if lang == "en"
        else ["1 - 非暴露组", "2 - 胎儿期暴露组", "3 - 儿童期暴露组", "4 - 青春期/成人暴露组"],
        index=famine_exposure - 1
    )
    st.caption(f"{T['auto']} {famine_text}")

# -------------------------------
# Calculate Risk
# -------------------------------
numerical_stats = {'SBP': {'mean': 135, 'std': 20}, 'TG': {'mean': 150, 'std': 80},
                   'WBC': {'mean': 6.5, 'std': 2.0}, 'BMI': {'mean': 24, 'std': 4}}

def standardize(feature, value):
    if feature in numerical_stats:
        mean = numerical_stats[feature]['mean']
        std = numerical_stats[feature]['std']
        return (value - mean) / std if std != 0 else value
    return value

if st.button(T["predict"]):
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

    X_std = np.array([standardize(k, v) for k, v in inputs.items()]).reshape(1, -1)
    coef_array = np.array([example_coefficients.get(k, 0) for k in inputs.keys()])
    lp = intercept + np.dot(X_std, coef_array)
    p = 1 / (1 + np.exp(-lp))
    risk = float(p[0])

    st.markdown(f"<h3 style='text-align:center'>🩺 {T['title']} Probability: <b>{risk * 100:.1f}%</b></h3>", unsafe_allow_html=True)

    if risk < 0.10:
        st.success(T["low"])
    elif risk < 0.30:
        st.warning(T["mid"])
    else:
        st.error(T["high"])

    # -------------------------------
    # SHAP Force Plot
    # -------------------------------
    try:
        if model is not None:
            inner_model = model.base_estimator_ if hasattr(model, "base_estimator_") else model
            explainer = shap.LinearExplainer(inner_model, np.zeros((1, len(inputs))))
            shap_values = explainer(np.array(list(inputs.values())).reshape(1, -1))
            st.subheader(T["shap"])
            shap.plots.force(shap_values[0], matplotlib=True, show=False)
            st.pyplot(bbox_inches='tight')
        else:
            st.info(T["no_model"])
    except Exception as e:
        st.warning(f"Unable to display SHAP plot: {e}")
