import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Smart Predictive Maintenance",
    layout="wide"
)

# =========================================================
# LOAD MODELS
# =========================================================

stage1_model = joblib.load("stage1_model.pkl")
stage1_scaler = joblib.load("stage1_scaler.pkl")

stage2_model = joblib.load("stage2_model.pkl")
stage2_scaler = joblib.load("stage2_scaler.pkl")

# =========================================================
# TITLE
# =========================================================

st.title("Predictive Maintenance System for Smart Cities")

st.markdown("---")

st.write("""
This system predicts:

✅ Machine Health Status  
✅ Failure Type  
✅ Risk Level  
✅ Confidence Score
""")

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Enter Machine Parameters")

machine_type = st.sidebar.selectbox(
    "Machine Type",
    ['L', 'M', 'H']
)

air_temp = st.sidebar.number_input(
    "Air Temperature [K]",
    value=298.0
)

process_temp = st.sidebar.number_input(
    "Process Temperature [K]",
    value=308.0
)

rot_speed = st.sidebar.number_input(
    "Rotational Speed [rpm]",
    value=1500
)

torque = st.sidebar.number_input(
    "Torque [Nm]",
    value=40.0
)

tool_wear = st.sidebar.number_input(
    "Tool Wear [min]",
    value=100
)

# =========================================================
# ENCODE MACHINE TYPE
# =========================================================

type_map = {
    'L': 0,
    'M': 1,
    'H': 2
}

machine_type_encoded = type_map[machine_type]

# =========================================================
# STAGE 1 INPUT
# =========================================================

stage1_input = pd.DataFrame([{
    'Type': machine_type_encoded,
    'Air temperature [K]': air_temp,
    'Process temperature [K]': process_temp,
    'Rotational speed [rpm]': rot_speed,
    'Torque [Nm]': torque,
    'Tool wear [min]': tool_wear
}])

# =========================================================
# SCALE INPUT
# =========================================================

stage1_scaled = stage1_scaler.transform(stage1_input)

# =========================================================
# BUTTON
# =========================================================

if st.button("Predict Machine Status"):

    st.subheader("Prediction Result")

    # =====================================================
    # STAGE 1 PREDICTION
    # =====================================================

    failure_probability = stage1_model.predict_proba(stage1_scaled)[0][1]

    threshold = 0.10

    if failure_probability > threshold:
        stage1_prediction = 1
    else:
        stage1_prediction = 0

    stage1_probability = stage1_model.predict_proba(stage1_scaled)[0]

    confidence = round(np.max(stage1_probability) * 100, 2)

    # =====================================================
    # METRICS
    # =====================================================

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Failure Probability",
        f"{round(failure_probability*100,2)}%"
    )

    col2.metric(
        "Confidence Score",
        f"{confidence}%"
    )

    health_score = round(100 - (failure_probability * 100), 2)

    col3.metric(
        "Machine Health Score",
        f"{health_score}/100"
    )

    st.markdown("---")

    # =====================================================
    # HEALTHY MACHINE
    # =====================================================

    if stage1_prediction == 0:

        st.success("✅ Machine is Healthy")

    # =====================================================
    # FAILURE DETECTED
    # =====================================================

    else:

        st.error("⚠️ Machine Failure Predicted")

        # =================================================
        # FEATURE ENGINEERING
        # =================================================

        temp_difference = process_temp - air_temp

        power = (torque * rot_speed) / 1000

        # =================================================
        # STAGE 2 INPUT
        # =================================================

        stage2_input = pd.DataFrame([{
            'Type': machine_type_encoded,
            'Air temperature [K]': air_temp,
            'Process temperature [K]': process_temp,
            'Rotational speed [rpm]': rot_speed,
            'Torque [Nm]': torque,
            'Tool wear [min]': tool_wear,
            'Temp Difference': temp_difference,
            'Power': power
        }])

        # =================================================
        # SCALE STAGE 2 INPUT
        # =================================================

        stage2_scaled = stage2_scaler.transform(stage2_input)

        # =================================================
        # FAILURE TYPE PREDICTION
        # =================================================

        failure_type = stage2_model.predict(stage2_scaled)[0]

        failure_prob = stage2_model.predict_proba(stage2_scaled)[0]

        failure_confidence = round(np.max(failure_prob) * 100, 2)

        st.subheader("Failure Type Prediction")

        st.error(f"Predicted Failure Type: {failure_type}")

        st.warning(f"Prediction Confidence: {failure_confidence}%")

        # =================================================
        # RISK LEVEL
        # =================================================

        st.subheader("Risk Level")

        risk_percent = failure_probability * 100

        if risk_percent > 60:
            st.error("🔴 HIGH RISK")

        elif risk_percent > 30:
            st.warning("🟠 MEDIUM RISK")

        else:
            st.success("🟢 LOW RISK")

        # =================================================
        # FAILURE INFORMATION
        # =================================================

        st.subheader("Failure Information")

        if failure_type == "TWF":
            st.write("Tool Wear Failure")

        elif failure_type == "HDF":
            st.write("Heat Dissipation Failure")

        elif failure_type == "PWF":
            st.write("Power Failure")

        elif failure_type == "OSF":
            st.write("Overstrain Failure")

        elif failure_type == "RNF":
            st.write("Random Failure")

        # =================================================
        # FAILURE PROBABILITY PIE CHART
        # =================================================

        st.subheader("Failure Probability Distribution")

        healthy_prob = 100 - risk_percent

        fig2, ax2 = plt.subplots(figsize=(5,5))

        ax2.pie(
            [healthy_prob, risk_percent],
            labels=['Healthy', 'Failure Risk'],
            autopct='%1.1f%%',
            startangle=90
        )

        ax2.set_title("Machine Health Distribution")

        st.pyplot(fig2)

        # =================================================
        # FEATURE CONTRIBUTION CHART
        # =================================================

        st.subheader("Feature Contribution Analysis")

        feature_data = pd.DataFrame({
            'Feature': [
                'Air Temp',
                'Process Temp',
                'Rot Speed',
                'Torque',
                'Tool Wear'
            ],
            'Value': [
                air_temp,
                process_temp,
                rot_speed,
                torque,
                tool_wear
            ]
        })

        fig3, ax3 = plt.subplots(figsize=(8,4))

        ax3.barh(
            feature_data['Feature'],
            feature_data['Value']
        )

        ax3.set_title("Machine Parameter Contribution")

        st.pyplot(fig3)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

