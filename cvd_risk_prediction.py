import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import shap
import matplotlib.pyplot as plt

# ----------------------------
# Configuration / defaults
# ----------------------------
MODEL_FILENAMES = [
    "model_LR_tuned_optuna_calibrated.pkl",
    "model.pkl",
    "model_LR_tuned.pkl",
    "model_LR_default.pkl"
]

# Example coefficients & intercept (fallback)
EXAMPLE_INTERCEPT = -1.5
EXAMPLE_COEFS = {
    'SBP': 0.1448, 'TG': 0.0315, 'WBC': 0.0659, 'BMI': 0.0256,
    'Hypertension': 0.1309, 'Dyslipidemia': 0.1399, 'Multimorbidity': 0.1841,
    'Bodily pains': 0.1569, 'Famine Exposure': 0.2030
}

NUMERICAL_FEATURES = ['SBP', 'TG', 'WBC', 'BMI']
CATEGORICAL_FEATURES = ['Hypertension', 'Dyslipidemia', 'Multimorbidity', 'Bodily pains', 'Famine Exposure']
FEATURE_ORDER = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Example standardization stats (fallback)
NUM_STATS = {
    'SBP': {'mean': 135.0, 'std': 20.0},
    'TG': {'mean': 150.0, 'std': 80.0},
    'WBC': {'mean': 6.5, 'std': 2.0},
    'BMI': {'mean': 24.0, 'std': 4.0}
}

# ----------------------------
# Helper: try to load model and extract params
# ----------------------------
def load_model_params():
    # Try to find model file in current directory
    for fname in MODEL_FILENAMES:
        if os.path.exists(fname):
            try:
                model = joblib.load(fname)
                # try extract intercept and coefs
                if hasattr(model, "intercept_") and hasattr(model, "coef_"):
                    intercept = float(model.intercept_[0]) if hasattr(model.intercept_, "__len__") else float(model.intercept_)
                    coef_arr = model.coef_[0] if hasattr(model.coef_, "__len__") else model.coef_
                    # try get feature names
                    if hasattr(model, "feature_names_in_"):
                        features = list(model.feature_names_in_)
                    else:
                        features = FEATURE_ORDER
                    coefs = {feat: float(coef) for feat, coef in zip(features, coef_arr)}
                    return intercept, coefs
            except Exception as e:
                st.warning(f"Failed to load model file {fname}: {e}")
    # fallback
    return EXAMPLE_INTERCEPT, EXAMPLE_COEFS.copy()

# ----------------------------
# A light sklearn-like wrapper for SHAP/visualization
# ----------------------------
class SimpleLogit:
    def __init__(self, intercept, coef_dict, feature_order, num_stats):
        self.intercept_ = np.array([intercept])
        self.coef_ = np.array([ [coef_dict.get(f, 0.0) for f in feature_order] ])
        self.feature_names_in_ = np.array(feature_order)
        self.num_stats = num_stats
        self.feature_order = feature_order

    def _standardize(self, X):
        # X is 2D numpy array with columns in feature_order
        X_std = X.copy().astype(float)
        for i, f in enumerate(self.feature_order):
            if f in NUMERICAL_FEATURES:
                stats = self.num_stats.get(f, {"mean":0.0,"std":1.0})
                std = stats.get("std", 1.0) or 1.0
                mean = stats.get("mean", 0.0)
                X_std[:, i] = (X_std[:, i] - mean) / std
            else:
                # categorical: assume already 0/1 or integer encoded; no standardization
                X_std[:, i] = X_std[:, i]
        return X_std

    def predict_proba(self, X):
        Xs = self._standardize(np.asarray(X))
        lp = Xs.dot(self.coef_.T) + self.intercept_
        p = 1.0 / (1.0 + np.exp(-lp))
        probs = np.hstack([1-p, p])
        return probs

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="CVD Risk Calculator (English) with SHAP", layout="centered")

st.title("Cardiovascular Disease (CVD) Risk Calculator")
st.markdown("Enter the patient parameters below. The app computes a predicted risk probability (logistic regression) and shows a SHAP-style force plot to explain the contribution of each feature.")

# load params
intercept_val, coefs = load_model_params()

st.sidebar.header("Model information")
st.sidebar.write("Intercept (loaded or fallback):", round(intercept_val, 4))
st.sidebar.write("Coefficients (loaded or fallback):")
st.sidebar.json(coefs)

st.header("Inputs")

# Input widgets
col1, col2 = st.columns(2)

with col1:
    sbp = st.number_input("Systolic blood pressure (SBP, mmHg)", value=120.0, min_value=60.0, max_value=250.0, step=1.0)
    wbc = st.number_input("White blood cell count (WBC, ×10⁹/L)", value=6.0, min_value=1.0, max_value=40.0, step=0.1)
    hypertension = st.selectbox("Hypertension (0=No, 1=Yes)", options=[0,1], index=0)
    multimorbidity = st.selectbox("Multimorbidity (0=No, 1=Yes)", options=[0,1], index=0)
    famine = st.selectbox("Famine exposure (1=non-exposed, 2=fetal, 3=childhood, 4=adolescent/adult)", options=[1,2,3,4], index=1)

with col2:
    tg = st.number_input("Triglycerides (TG, mg/dL)", value=150.0, min_value=10.0, max_value=1000.0, step=1.0)
    bmi = st.number_input("Body mass index (BMI, kg/m²)", value=22.0, min_value=10.0, max_value=60.0, step=0.1)
    dyslipidemia = st.selectbox("Dyslipidemia (0=No,1=Yes)", options=[0,1], index=0)
    bodily = st.selectbox("Bodily pains (0=No,1=Yes)", options=[0,1], index=0)

# Pack features in the same order as FEATURE_ORDER
input_dict = {
    'SBP': sbp,
    'TG': tg,
    'WBC': wbc,
    'BMI': bmi,
    'Hypertension': hypertension,
    'Dyslipidemia': dyslipidemia,
    'Multimorbidity': multimorbidity,
    'Bodily pains': bodily,
    'Famine Exposure': famine
}

# Standardization stats - if user uploaded different stats, they can edit NUM_STATS at top of this file
num_stats = NUM_STATS

# Build feature vector in numpy
x_vec = np.array([[ input_dict[f] for f in FEATURE_ORDER ]], dtype=float)

# Create model wrapper
model_wrapper = SimpleLogit(intercept_val, coefs, FEATURE_ORDER, NUM_STATS)

# Compute probability
probs = model_wrapper.predict_proba(x_vec)
prob = float(probs[0,1])
pct = prob * 100.0

st.subheader("Predicted risk probability")
st.write(f"Probability of CVD: **{pct:.1f}%**")

# Risk level
if prob < 0.10:
    st.success("Risk level: Low")
elif prob < 0.30:
    st.warning("Risk level: Medium")
else:
    st.error("Risk level: High")

# Show contributions (manual per-feature contributions = coef * standardized value)
st.subheader("Feature contributions (linear model)")
# Standardize numeric features to compute contributions
def standardize_single(f, val):
    if f in NUM_STATS:
        std = NUM_STATS[f].get("std", 1.0) or 1.0
        mean = NUM_STATS[f].get("mean", 0.0)
        return (val - mean) / std
    else:
        return val

contribs = []
for f in FEATURE_ORDER:
    val = input_dict[f]
    stdval = standardize_single(f, val)
    coef = coefs.get(f, 0.0)
    contrib = coef * stdval
    contribs.append((f, val, stdval, coef, contrib))

contrib_df = pd.DataFrame(contribs, columns=['feature','raw','standardized','coef','contribution'])
contrib_df = contrib_df.sort_values('contribution', key=lambda s: np.abs(s), ascending=False)
st.dataframe(contrib_df.style.format({"raw":"{:.3f}","standardized":"{:.3f}","coef":"{:.4f}","contribution":"{:.4f}"}), height=300)

# ----------------------------
# SHAP force plot generation
# ----------------------------
st.subheader("SHAP-style force plot")

# Prepare data for SHAP: we will create a small background (zeros) and use LinearExplainer
try:
    # Create a small background (e.g., mean values)
    background = np.zeros((1, len(FEATURE_ORDER)))
    # But put background numeric columns as zeros after standardization (i.e., mean)
    # We will pass standardized data into the explainer, so define explainer based on the linear model coefficients
    # Create a sklearn-like model for explainer
    class SkModel:
        def __init__(self, intercept, coef):
            self.intercept_ = np.array([intercept])
            self.coef_ = np.array([coef])
    coef_list = [ coefs.get(f, 0.0) for f in FEATURE_ORDER ]
    skm = SkModel(intercept_val, coef_list)

    # For shap LinearExplainer, we want to explain the prediction on standardized features.
    # So we create X_standardized for sample:
    x_standardized = np.array([[ standardize_single(f, input_dict[f]) for f in FEATURE_ORDER ]])

    # Explainer
    explainer = shap.LinearExplainer(skm, background, feature_perturbation="interventional")
    shap_vals = explainer.shap_values(x_standardized)
    # shap_vals shape (n_features,) or (1, n_features)
    shap_vals_arr = np.array(shap_vals).reshape(1, -1)[0]

    # Plot force plot using matplotlib backend
    plt.figure(figsize=(10,2))
    # Use shap.force_plot (matplotlib)
    shap.force_plot(explainer.expected_value, shap_vals_arr, x_standardized[0], feature_names=FEATURE_ORDER, matplotlib=True, show=False)
    fig = plt.gcf()
    st.pyplot(fig)
except Exception as e:
    st.write("Could not create SHAP force plot automatically. Falling back to a bar chart of contributions.")
    st.write(e)
    # fallback: simple horizontal bar chart of contributions
    fig, ax = plt.subplots(figsize=(8,4))
    contrib_df_sorted = contrib_df.set_index('feature').sort_values('contribution')
    ax.barh(contrib_df_sorted.index, contrib_df_sorted['contribution'])
    ax.set_xlabel("Contribution (coef * standardized value)")
    st.pyplot(fig)

st.markdown("---")
st.markdown("Notes:")
st.markdown("- If you want the app to use your trained model, upload the model file (e.g., `model_LR_tuned_optuna_calibrated.pkl`) into the same folder as this app before deployment. The app will try to load it automatically.")
st.markdown("- For deployment to Streamlit Cloud: include `requirements.txt` in the repo and deploy the repository on https://share.streamlit.io/")
