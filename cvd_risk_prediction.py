import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib
import os
from supabase import create_client, Client
import bcrypt
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

# 页面配置
st.set_page_config(page_title="CVD Risk Prediction with User System", layout="centered")

# ----------------------------
# 初始化Supabase客户端
# ----------------------------
@st.cache_resource
def init_supabase():
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)

supabase = init_supabase()

# ----------------------------
# 加载模型（同你原有代码）
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

# 饥荒暴露分类函数
def categorize_famine(year):
    if year > 1963:
        return 1, "No-exposed group (birth after 1963)"
    elif 1959 <= year <= 1962:
        return 2, "Fetal-exposed group (1959–1962)"
    elif 1949 <= year <= 1958:
        return 3, "Childhood-exposed group (1949–1958)"
    else:
        return 4, "Adolescent/Adult-exposed group (≤1948)"

# ----------------------------
# 用户认证配置
# ----------------------------
def get_users():
    """从Supabase获取所有用户，返回字典格式供authenticator使用"""
    response = supabase.table("users").select("*").execute()
    data = response.data
    credentials = {"usernames": {}}
    for user in data:
        credentials["usernames"][user["username"]] = {
            "name": user.get("name", user["username"]),
            "password": user["password_hash"],  # authenticator需要的是已哈希的密码
            "email": user.get("email", "")
        }
    return credentials

# 保存新用户
def register_user(username, password, email=""):
    # 哈希密码
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    # 插入到Supabase
    try:
        supabase.table("users").insert({
            "username": username,
            "password_hash": hashed,
            "email": email
        }).execute()
        return True
    except Exception as e:
        st.error(f"注册失败: {e}")
        return False

# ----------------------------
# 历史记录保存和查询
# ----------------------------
def save_prediction(user_id, inputs, risk):
    """保存预测记录到Supabase"""
    data = {
        "user_id": user_id,
        "birth_year": inputs["birth_year"],
        "sbp": inputs["sbp"],
        "tg": inputs["tg"],
        "wbc": inputs["wbc"],
        "bmi": inputs["bmi"],
        "hypertension": inputs["hypertension"],
        "dyslipidemia": inputs["dyslipidemia"],
        "multimorbidity": inputs["multimorbidity"],
        "bodily_pains": inputs["bodily_pains"],
        "famine_exposure": inputs["famine_exposure"],
        "risk_probability": float(risk)
    }
    supabase.table("predictions").insert(data).execute()

def get_user_history(user_id):
    """查询用户的预测历史"""
    response = supabase.table("predictions")\
        .select("*")\
        .eq("user_id", user_id)\
        .order("created_at", desc=True)\
        .execute()
    return response.data

# ----------------------------
# 用户认证界面
# ----------------------------
# 先尝试从session_state获取认证状态，如果没有则初始化
if "authentication_status" not in st.session_state:
    st.session_state["authentication_status"] = None

# 侧边栏放置登录/注册/登出
with st.sidebar:
    st.title("用户中心")
    
    if st.session_state["authentication_status"]:
        st.write(f"欢迎，{st.session_state['name']}")
        if st.button("登出"):
            st.session_state["authentication_status"] = None
            st.session_state["name"] = None
            st.session_state["username"] = None
            st.rerun()
    else:
        # 简单的登录/注册选择
        mode = st.radio("选择", ["登录", "注册"])
        
        if mode == "登录":
            username = st.text_input("用户名")
            password = st.text_input("密码", type="password")
            if st.button("登录"):
                # 从Supabase查询用户
                response = supabase.table("users").select("*").eq("username", username).execute()
                if len(response.data) == 1:
                    user = response.data[0]
                    if bcrypt.checkpw(password.encode('utf-8'), user["password_hash"].encode('utf-8')):
                        st.session_state["authentication_status"] = True
                        st.session_state["name"] = user.get("name", username)
                        st.session_state["username"] = username
                        st.session_state["user_id"] = user["id"]
                        st.rerun()
                    else:
                        st.error("密码错误")
                else:
                    st.error("用户不存在")
        
        elif mode == "注册":
            new_user = st.text_input("用户名")
            new_pass = st.text_input("密码", type="password")
            new_email = st.text_input("邮箱（可选）")
            if st.button("注册"):
                if new_user and new_pass:
                    # 检查用户名是否已存在
                    exist = supabase.table("users").select("*").eq("username", new_user).execute()
                    if len(exist.data) == 0:
                        if register_user(new_user, new_pass, new_email):
                            st.success("注册成功，请登录")
                            st.rerun()
                    else:
                        st.error("用户名已存在")
                else:
                    st.warning("用户名和密码不能为空")

# ----------------------------
# 主应用：仅当登录后显示
# ----------------------------
if st.session_state.get("authentication_status"):
    # 语言切换
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

    # 文本字典
    TEXT = {
        "en": {
            "title": "CVD Risk Prediction",
            "intro": "Estimate 5-year CVD risk using a logistic model and SHAP.",
            "birth": "Birth Year",
            "sbp": "Systolic BP (mmHg)",
            "tg": "Triglycerides (mg/dL)",
            "wbc": "White Blood Cell (×10⁹/L)",
            "bmi": "Body Mass Index (kg/m²)",
            "htn": "Hypertension (0=No, 1=Yes)",
            "dys": "Dyslipidemia (0=No, 1=Yes)",
            "multi": "Multimorbidity (0=No, 1=Yes)",
            "pain": "Bodily pains (0=No, 1=Yes)",
            "famine": "Famine Exposure (auto)",
            "auto": "Automatically detected based on birth year:",
            "predict": "Predict",
            "low": "Low risk. Keep healthy.",
            "mid": "Moderate risk. Consider regular checkups.",
            "high": "High risk. Consult a doctor.",
            "history": "Prediction History",
            "no_history": "No history yet.",
        },
        "cn": {
            "title": "心血管疾病风险预测",
            "intro": "基于Logistic回归模型和SHAP可视化。",
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
            "low": "低风险：保持健康。",
            "mid": "中风险：建议定期检查。",
            "high": "高风险：请就医。",
            "history": "历史预测记录",
            "no_history": "暂无历史记录。",
        }
    }
    T = TEXT[lang]

    st.markdown(f"<h1 style='text-align:center'>{T['title']}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center; color:#444'>{T['intro']}</p>", unsafe_allow_html=True)
    st.markdown("---")

    # 输入布局
    col1, col2 = st.columns(2)

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
        st.caption(f"{T['auto']} {famine_text}")

    # 预测按钮
    if st.button(T["predict"]):
        # 用于模型预测的字典（注意键名与模型训练时一致）
        input_dict = {
            'SBP': sbp, 'TG': tg, 'WBC': wbc, 'BMI': bmi,
            'Hypertension': hypertension,
            'Dyslipidemia': dyslipidemia,
            'Multimorbidity': multimorbidity,
            'Bodily pains': bodily_pains,
            'Famine Exposure': famine_exposure
        }

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

        st.markdown(f"<h3 style='text-align:center'>Risk Probability: <b>{risk*100:.1f}%</b></h3>", unsafe_allow_html=True)
        if risk < 0.1:
            st.success(T["low"])
        elif risk < 0.3:
            st.warning(T["mid"])
        else:
            st.error(T["high"])

        # 构建用于保存的数据字典（键名与数据库列一致）
        save_data = {
            "birth_year": birth_year,
            "sbp": sbp,
            "tg": tg,
            "wbc": wbc,
            "bmi": bmi,
            "hypertension": hypertension,
            "dyslipidemia": dyslipidemia,
            "multimorbidity": multimorbidity,
            "bodily_pains": bodily_pains,
            "famine_exposure": famine_exposure
        }

        # 调用保存函数，传入新字典
        save_prediction(st.session_state["user_id"], save_data, risk)

        # SHAP力图
        st.subheader("SHAP Force Plot")
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
            st.write("SHAP plot failed, showing contribution bar chart.")
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

    # 显示历史记录
    with st.expander(T["history"]):
        history = get_user_history(st.session_state["user_id"])
        if history:
            df = pd.DataFrame(history)
            # 选择要显示的列
            cols = ['created_at', 'risk_probability'] + [c for c in df.columns if c not in ['id', 'user_id', 'created_at', 'risk_probability']]
            df = df[cols]
            df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
            df['risk_probability'] = (df['risk_probability'] * 100).round(1).astype(str) + '%'
            st.dataframe(df)
        else:
            st.info(T["no_history"])

else:
    st.info("请先在左侧登录或注册以使用预测工具。")
