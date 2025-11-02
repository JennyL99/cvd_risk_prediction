# CVD Risk Calculator (Streamlit)

This repository contains a Streamlit application that reproduces a simple CVD risk calculator and shows a SHAP-style force plot.

## How it works
- The app tries to load a model file from the app folder (e.g. `model_LR_tuned_optuna_calibrated.pkl` or `model.pkl`). If not found, it will use example coefficients included in the script.
- Inputs are standardized using example mean/std values. If you have different standardization statistics, update the `NUM_STATS` dictionary in `app.py`.
- The app uses a simple linear model wrapper and SHAP's `LinearExplainer` to produce a force plot for the standardized input.

## How to deploy to Streamlit Cloud
1. Put this repository on GitHub.
2. On https://share.streamlit.io, click **New app**, connect your GitHub and select this repository and `app.py` as the main file.
3. Ensure `requirements.txt` is present (it is included) so Streamlit Cloud can install dependencies.
4. If you want to use your trained model, upload your `.pkl` model file to the repository (or to the app folder) before deploying.

## Notes
- If your model uses a Pipeline or different preprocessing, adjust `load_model_params()` in `app.py` to extract the real coefficients correctly.
- If SHAP plotting fails in the environment, the app falls back to a simple bar chart of contributions.
