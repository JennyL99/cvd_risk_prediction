import streamlit as st
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import pandas as pd

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="CVD Risk Prediction", layout="centered")

# -------------------------------
# Language toggle in top-right
# -------------------------------
if "lang" not in st.session_state:
    st.session_state.lang = "en"

# Create two columns, put the language toggle in the right (small) column
col_left, col_right = st.columns([0.82, 0.18])
with col_right:
    # show the toggle button; when english -> button shows 中文; when chinese -> button shows English
    if st.session_state.lang == "en":
        if st.button("中文"):
            st.session_state.lang = "cn"
    else:
        if st.button("English"):
            st.session_state.lang = "en"

lang = st.session_state.lang

# -------------------------------
# Multilingual text
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
        "no_model": "SHAP visualization is not available without the trained model file.",
        "model_info_title": "Model information",
        "model_info_text": "The app tries to load a trained model file named `model_LR_tuned_optuna_calibrated.pkl`. "
                           "If not found, example coefficients are used for demonstration."
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
        "no_model": "未检测到模型文件，无法显示 SHAP 可视化。",
        "model_info_title": "模型简介",
        "model_info_text": "程序会尝试加载名为 `model_LR_tuned_optuna_calibrated.pkl` 的训练模型文件。若未找到，将使用示例系数进行演示。"
    }
}
T = TEXT[lang]

# -------------------------------
# Try to load model
# -------------------------------
MODEL_PATH = "model_LR_tuned_optuna_calibrated.pkl"
try:
    model = joblib.load(MODEL_PATH)
    model_loaded = True
except Exception as e:
    model = None
    model_loaded = False

# -------------------------------
# Default coefficients (fallback)
# -------------------------------
example_coefficients = {
    'SBP': 0.1448, 'TG': 0.0315, 'WBC': 0.0659, 'BMI': 0.0256,
    'Hypertension': 0.1309, 'Dyslipidemia': 0.1399, 'Multimorbidity': 0.1841,
    'Bodily pains': 0.1569, 'Famine Exposure': 0.2030
}
intercept = -1.5

# Try to extract real coefficients if possible
if model is not None:
    try:
        # If model is a calibrated wrapper, try to find inner estimator
        if model.__class__.__name__ == "CalibratedClassifierCV":
            # try a few ways to get the inner estimator
            inner = None
            if hasattr(model, "base_estimator") and model.base_estimator is not None:
                inner = model.base_estimator
            elif hasattr(model, "base_estimator_") and model.base_estimator_ is not None:
                inner = model.base_estimator_
            elif hasattr(model, "calibrated_classifiers_") and len(model.calibrated_classifiers_) > 0:
                # calibrated_classifiers_ may contain fitted classifiers; try first and see if it has coef_
                try:
                    inner = model.calibrated_classifiers_[0]
                except Exception:
                    inner = None
            if inner is not None and hasattr(inner, "coef_"):
                intercept = float(inner.intercept_[0]) if hasattr(inner.intercept_, "__len__") else float(inner.intercept_)
                example_coefficients = dict(zip(inner.feature_names_in_, inner.coef_[0]))
            else:
                # fallback: try the wrapper itself
                if hasattr(model, "coef_"):
                    intercept = float(model.intercept_[0]) if hasattr(model.intercept_, "__len__") else float(model.intercept_)
                    example_coefficients = dict(zip(model.feature_names_in_, model.coef_[0]))
        else:
            # not calibrated, interpret directly
            if hasattr(model, "coef_") and hasattr(model, "feature_names_in_"):
                intercept = float(model.intercept_[0]) if hasattr(model.intercept_, "__len__") else float(model.intercept_)
                example_coefficients = dict(zip(model.feature_names_in_, model.coef_[0]))
    except Exception:
        # keep fallback if extraction fails
        pass

# -------------------------------
# Title + model intro (moved under title)
# -------------------------------
st.markdown(f"<h1 style='text-align:center'>{T['title']}</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align:center; color:#444'>{T['intro']}</p>", unsafe_allow_html=True)

# Model intro area (under title, above inputs)
st.markdown("### " + T.get("model_info_title", "Model information"))
st.info(T.get("model_info_text", ""))

st.markdown("---")

# -------------------------------
# Inputs layout
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    birth_year = st.number_input(T["birth"], min_value=1900, max_value=2025, value=1960, step=1)
    sbp = st.number_input(T["sbp"], min_value=80, max_value=200, value=120, step=1)
    tg = st.number_input(T["tg"], min_value=50, max_value=500, value=150, step=1)
    wbc = st.number_input(T["wbc"], min_value=2.0, max_value=20.0, value=6.0, step=0.1)
    bmi = st.number_input(T["bmi"], min_value=15.0, max_value=40.0, value=22.0, step=0.1)

# famine categorization based on birth year (localized text)
def categorize_famine_exposure(year):
    if year > 1963:
        return 1, ("No-exposed group (birth after 1963-01-01)" if lang == "en" else "非暴露组（出生≥1963-01-01）")
    elif 1959 <= year <= 1962:
        return 2, ("Fetal-exposed group (birth 1959–1962)" if lang == "en" else "胎儿期暴露组（1959–1962年出生）")
    elif 1949 <= year <= 1958:
        return 3, ("Childhood-exposed group (birth 1949–1958)" if lang == "en" else "儿童期暴露组（1949–1958年出生）")
    else:
        return 4, ("Adolescence/Adult-exposed group (birth ≤1948)" if lang == "en" else "青春期/成人暴露组（≤1948年出生）")

famine_exposure, famine_text = categorize_famine_exposure(birth_year)

with col2:
    hypertension = st.selectbox(T["htn"], [0, 1])
    dyslipidemia = st.selectbox(T["dys"], [0, 1])
    multimorbidity = st.selectbox(T["multi"], [0, 1])
    bodily_pains = st.selectbox(T["pain"], [0, 1])
    # Famine exposure shown as dropdown (auto set but same UI)
    famine_options = (["1 - No-exposed", "2 - Fetal-exposed", "3 - Childhood-exposed", "4 - Adolescence/Adult-exposed"]
                      if lang == "en"
                      else ["1 - 非暴露组", "2 - 胎儿期暴露组", "3 - 儿童期暴露组", "4 - 青春期/成人暴露组"])
    famine_display = st.selectbox(T["famine"], famine_options, index=famine_exposure - 1)
    st.caption(f"{T['auto']} {famine_text}")

# -------------------------------
# Prediction logic
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
    # SHAP: robust handling for calibrated or unknown model types
    # -------------------------------
    def build_simple_linear_model(intercept_val, coef_dict, feature_order):
        coef_list = [coef_dict.get(f, 0.0) for f in feature_order]
        class SimpleLinear:
            def __init__(self, intercept, coefs):
                self.intercept_ = np.array([intercept])
                self.coef_ = np.array([coefs])
                self.feature_names_in_ = np.array(feature_order)
            def predict(self, X):
                X = np.asarray(X, dtype=float)
                return (X.dot(np.array(coefs)) + intercept).ravel()
            def predict_proba(self, X):
                lp = self.predict(X)
                p = 1.0 / (1.0 + np.exp(-lp))
                # return shape (n_samples, 2)
                return np.vstack([1 - p, p]).T
        # Note: we capture intercept and coef_list via closure variables
        return SimpleLinear(intercept_val, coef_list)

    try:
        if model is not None:
            # try to derive an inner linear estimator for SHAP
            inner_for_shap = None
            # Case 1: CalibratedClassifierCV (commonly causes the "unknown model type" error)
            if model.__class__.__name__ == "CalibratedClassifierCV":
                # try base_estimator or base_estimator_
                if hasattr(model, "base_estimator") and model.base_estimator is not None:
                    cand = model.base_estimator
                elif hasattr(model, "base_estimator_") and model.base_estimator_ is not None:
                    cand = model.base_estimator_
                else:
                    cand = None
                # if cand has coef_ use it
                if cand is not None and hasattr(cand, "coef_"):
                    inner_for_shap = cand
                else:
                    # fallback: try calibrated_classifiers_ list elements (sklearn stores fitted calibrated estimators)
                    try:
                        if hasattr(model, "calibrated_classifiers_") and len(model.calibrated_classifiers_) > 0:
                            candidate = model.calibrated_classifiers_[0]
                            if hasattr(candidate, "coef_"):
                                inner_for_shap = candidate
                    except Exception:
                        inner_for_shap = None
            else:
                # not calibrated: use model directly if it's linear-like
                if hasattr(model, "coef_"):
                    inner_for_shap = model

            # If still None, build a simple linear model from extracted coefficients
            if inner_for_shap is None:
                inner_for_shap = build_simple_linear_model(intercept, example_coefficients, list(inputs.keys()))

            # Prepare standardized sample for SHAP explanation (explainer expects the same space as model)
            sample_std = np.array([standardize(k, v) for k, v in inputs.items()]).reshape(1, -1)
            # Background: use zeros (meaning mean after standardization)
            background = np.zeros((1, sample_std.shape[1]))
            explainer = shap.LinearExplainer(inner_for_shap, background, feature_perturbation="interventional")
            shap_vals = explainer(sample_std)
            st.subheader(T["shap"])
            # shap_vals may be array-like; produce force plot
            try:
                shap.plots.force(shap_vals[0], matplotlib=True, show=False)
                st.pyplot(bbox_inches='tight')
            except Exception:
                # last-resort fallback: show bar chart of contributions
                contribs = []
                for f in inputs.keys():
                    stdv = standardize(f, inputs[f])
                    coef = example_coefficients.get(f, 0.0)
                    contribs.append((f, coef * stdv))
                contrib_df = pd.DataFrame(contribs, columns=["feature", "contribution"]).set_index("feature")
                fig, ax = plt.subplots(figsize=(8,4))
                contrib_df.sort_values("contribution", inplace=True)
                ax.barh(contrib_df.index, contrib_df["contribution"])
                ax.set_xlabel("Contribution (coef * standardized value)")
                st.pyplot(fig)
        else:
            st.info(T["no_model"])
    except Exception as e:
        st.warning(f"Unable to display SHAP plot: {e}")
