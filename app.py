import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Page config
st.set_page_config(
    page_title="Predictive Maintenance — Machine Failure Predictor",
    page_icon="⚙️",
    layout="centered"
)

# Header
st.title("⚙️ Predictive Maintenance")
st.subheader("Machine Failure Prediction System")
st.markdown("Enter real-time sensor readings to predict whether a machine is at risk of failure.")
st.divider()

# Input form
col1, col2 = st.columns(2)

with col1:
    machine_type = st.selectbox("Machine Type", options=[0, 1, 2],
                                 format_func=lambda x: ['High (H)', 'Low (L)', 'Medium (M)'][x])
    air_temp = st.slider("Air Temperature (K)", 295.0, 305.0, 300.0, 0.1)
    process_temp = st.slider("Process Temperature (K)", 305.0, 315.0, 310.0, 0.1)

with col2:
    rot_speed = st.slider("Rotational Speed (rpm)", 1168, 2886, 1500)
    torque = st.slider("Torque (Nm)", 3.8, 76.6, 40.0, 0.1)
    tool_wear = st.slider("Tool Wear (min)", 0, 253, 100)

st.divider()

# Engineered features
temp_diff = process_temp - air_temp
power = torque * rot_speed
tool_wear_torque = tool_wear * torque

# Show engineered features
with st.expander("🔧 Engineered Features (auto-calculated)"):
    c1, c2, c3 = st.columns(3)
    c1.metric("Temp Difference", f"{temp_diff:.2f} K")
    c2.metric("Power", f"{power:.0f} W")
    c3.metric("Tool Wear × Torque", f"{tool_wear_torque:.1f}")

# Predict button
if st.button("🔍 Predict Machine Status", use_container_width=True):
    features = np.array([[machine_type, air_temp, process_temp,
                          rot_speed, torque, tool_wear,
                          temp_diff, power, tool_wear_torque]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    st.divider()
    if prediction == 1:
        st.error(f"⚠️ HIGH RISK — Machine failure predicted!")
        st.metric("Failure Probability", f"{probability*100:.1f}%")
        st.markdown("**Recommended Action:** Schedule immediate maintenance inspection.")
    else:
        st.success(f"✅ NORMAL — Machine operating within safe parameters")
        st.metric("Failure Probability", f"{probability*100:.1f}%")
        st.markdown("**Status:** Continue monitoring. Next scheduled check as planned.")

    # Probability gauge
    st.progress(float(probability))

st.divider()
st.caption("Model: Gradient Boosting | Accuracy: 99.25% | ROC-AUC: 0.967 | Dataset: AI4I 2020 Predictive Maintenance")
