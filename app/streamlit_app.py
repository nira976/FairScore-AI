import streamlit as st
import joblib
import pandas as pd
import time
import shap
import numpy as np
# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="FairScore AI",
    page_icon="💳",
    layout="wide"
)
st.markdown("""
<style>

.stApp {
    background-color: #0E1117;
}

.main-title {
    text-align: center;
    color: #4CAF50;
    font-size: 55px;
    font-weight: bold;
}

.sub-title {
    text-align: center;
    color: #B0B0B0;
    font-size: 22px;
}

.footer {
    text-align: center;
    color: grey;
    margin-top: 40px;
}

.stButton > button {
    width: 100%;
    height: 60px;
    border-radius: 15px;
    background: linear-gradient(90deg, #4CAF50, #2ECC71);
    color: white;
    font-size: 20px;
    font-weight: bold;
    border: none;
}

.stButton > button:hover {
    border: 2px solid #4CAF50;
}

</style>
""", unsafe_allow_html=True)
# ---------------------------
# Load Model
# ---------------------------
model = joblib.load("fairscore_xgb_model.pkl")
explainer = shap.TreeExplainer(model)
# ---------------------------
# Mappings
# ---------------------------
worker_mapping = {
    "Content Creator": 0,
    "Delivery Partner": 1,
    "Freelancer": 2,
    "Ride Share Driver": 3
}

risk_mapping = {
    0: "High Risk",
    1: "Low Risk",
    2: "Moderate Risk"
}

# ---------------------------
# Header
# ---------------------------
st.markdown("""
<div class='main-title'>
💳 FairScore AI
</div>

<div class='sub-title'>
Explainable Credit Risk Assessment Platform for Gig Workers
</div>

<div class='sub-title'>
Built using XGBoost and Explainable AI
</div>

<hr>
""", unsafe_allow_html=True)

# ---------------------------
# Metrics
# ---------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="📊 Model Accuracy",
        value="81%"
    )

with col2:
    st.metric(
        label="👥 Dataset Size",
        value="1000 Workers"
    )

with col3:
    st.metric(
        label="🤖 ML Model",
        value="XGBoost"
    )

st.write("")

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("📌 About FairScore AI")

st.sidebar.info("""
FairScore AI evaluates the credit risk of gig workers using machine learning.

The model considers:

• Financial stability
• Payment behaviour
• Employment characteristics
• Debt burden

Built using:

• XGBoost
• SHAP Explainability
• Streamlit
""")

# ---------------------------
# Input Sections
# ---------------------------
col1, col2 = st.columns(2)

# Financial Inputs
with col1:

    st.subheader("💰 Financial Indicators")

    monthly_income = st.number_input(
        "Monthly Income (₹)",
        min_value=0,
        value=30000
    )

    savings_ratio = st.slider(
        "Savings Ratio",
        0.0,
        1.0,
        0.30
    )

    existing_debt_ratio = st.slider(
        "Existing Debt Ratio",
        0.0,
        1.0,
        0.40
    )

    emi_defaults = st.number_input(
        "EMI Defaults",
        min_value=0,
        value=0
    )

    emergency_fund_months = st.slider(
        "Emergency Fund (Months)",
        0,
        12,
        3
    )

    spending_volatility = st.slider(
        "Spending Volatility",
        0.0,
        1.0,
        0.50
    )

# Behavioural Inputs
with col2:

    st.subheader("📱 Behavioural Indicators")

    customer_rating = st.slider(
        "Customer Rating",
        1.0,
        5.0,
        4.0
    )

    bill_payment_score = st.slider(
        "Bill Payment Score",
        0,
        100,
        75
    )

    upi_transactions = st.number_input(
        "Monthly UPI Transactions",
        min_value=0,
        value=50
    )

    employment_duration = st.slider(
        "Employment Duration (Years)",
        0,
        20,
        3
    )

    platform_dependency = st.slider(
        "Platform Dependency",
        0.0,
        1.0,
        0.50
    )

    income_stability = st.slider(
        "Income Stability",
        0.0,
        1.0,
        0.50
    )

    worker_type = st.selectbox(
        "Worker Type",
        [
            "Content Creator",
            "Delivery Partner",
            "Freelancer",
            "Ride Share Driver"
        ]
    )

# ---------------------------
# Predict Button
# ---------------------------
st.write("")
st.write("")

predict = st.button(
    "🔍 Assess Credit Risk",
    use_container_width=True
)

# ---------------------------
# Prediction Section
# ---------------------------
if predict:

    with st.spinner("🤖 AI is analyzing the financial profile..."):

        time.sleep(2)

        input_data = pd.DataFrame({

            "worker_type": [worker_mapping[worker_type]],
            "monthly_income": [monthly_income],
            "income_stability": [income_stability],
            "customer_rating": [customer_rating],
            "savings_ratio": [savings_ratio],
            "upi_transactions": [upi_transactions],
            "bill_payment_score": [bill_payment_score],
            "emi_defaults": [emi_defaults],
            "spending_volatility": [spending_volatility],
            "employment_duration": [employment_duration],
            "existing_debt_ratio": [existing_debt_ratio],
            "platform_dependency": [platform_dependency],
            "emergency_fund_months": [emergency_fund_months]

        })

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(input_data)[0]

        confidence = probability.max() * 100

        risk = risk_mapping[prediction]

        # SHAP Explanation
        shap_values = explainer.shap_values(input_data)
        #st.write("SHAP type:", type(shap_values))
        #st.write("SHAP shape:", np.array(shap_values).shape)

        feature_names = input_data.columns.tolist()

        feature_importance = np.abs(shap_values[0, :, prediction])

        top_indices = np.argsort(feature_importance)[::-1][:3]

        top_features = []

        for idx in top_indices:

         feature = feature_names[idx]

         value = input_data.iloc[0][feature]

         top_features.append((feature, value))

    # ---------------------------
    # Prediction Summary
    # ---------------------------

    st.balloons()

    st.info(f"""
Prediction Summary

Worker Type: {worker_type}

Predicted Category: {risk}

Model Confidence: {confidence:.2f}%
""")

    st.write("")

    # ---------------------------
    # Risk Cards
    # ---------------------------

    if risk == "Low Risk":

        st.markdown(f"""
        <div style="
        padding:30px;
        border-radius:15px;
        background-color:#28a745;
        text-align:center;
        color:white;
        font-size:30px;
        font-weight:bold;
        ">
        🟢 LOW RISK<br><br>
        Confidence: {confidence:.2f}%
        </div>
        """, unsafe_allow_html=True)

    elif risk == "Moderate Risk":

        st.markdown(f"""
        <div style="
        padding:30px;
        border-radius:15px;
        background-color:#f0ad4e;
        text-align:center;
        color:white;
        font-size:30px;
        font-weight:bold;
        ">
        🟡 MODERATE RISK<br><br>
        Confidence: {confidence:.2f}%
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown(f"""
        <div style="
        padding:30px;
        border-radius:15px;
        background-color:#ff4b4b;
        text-align:center;
        color:white;
        font-size:30px;
        font-weight:bold;
        ">
        🔴 HIGH RISK<br><br>
        Confidence: {confidence:.2f}%
        </div>
        """, unsafe_allow_html=True)

    # ---------------------------
    # SHAP Explanation
    # ---------------------------

    st.write("")

    st.subheader("🧠 AI Explanation")

    st.info(
        "These factors had the strongest influence on the model's prediction."
    )

    for feature, value in top_features:

        st.write(
            f"• **{feature.replace('_', ' ').title()}** : {value}"
        )

# ---------------------------
# Footer
# ---------------------------

st.markdown("""
<hr>

<div class='footer'>

Built by Niraja Manjrekar ❤️

FairScore AI © 2026

</div>
""", unsafe_allow_html=True)