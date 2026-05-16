import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date

# Page settings
st.set_page_config(
    page_title="⚡ Electricity Demand Forecasting",
    page_icon="⚡",
    layout="wide"
)

# Model load karo
@st.cache_resource
def load_model():
    model = joblib.load('electricity_model.pkl')
    return model

model = load_model()

# Title
st.title("⚡ Electricity Demand Forecasting")
st.markdown("### Predict future electricity consumption!")

# Sidebar - User Input
st.sidebar.header("📅 Enter Date Details")

selected_date = st.sidebar.date_input(
    "Select a Date",
    value=date.today()
)

lag_1 = st.sidebar.number_input(
    "Yesterday's Consumption (GWh)",
    min_value=800.0,
    max_value=1800.0,
    value=1338.0
)

lag_7 = st.sidebar.number_input(
    "Last Week Same Day (GWh)",
    min_value=800.0,
    max_value=1800.0,
    value=1338.0
)

lag_30 = st.sidebar.number_input(
    "Last Month Same Day (GWh)",
    min_value=800.0,
    max_value=1800.0,
    value=1338.0
)

# Prediction button
if st.sidebar.button("🔮 Predict!"):
    
    # Features banana
    features = np.array([[
        selected_date.year,
        selected_date.month,
        selected_date.day,
        selected_date.weekday(),
        (selected_date.month - 1) // 3 + 1,
        lag_1, lag_7, lag_30
    ]])
    
    # Prediction
    prediction = model.predict(features)[0]
    
    # Result dikhao
    st.success(f"### ⚡ Predicted Consumption: {prediction:.2f} GWh")
    
    # Gauge chart
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📅 Date", selected_date.strftime("%B %d, %Y"))
    with col2:
        st.metric("⚡ Prediction", f"{prediction:.2f} GWh")
    with col3:
        day_names = ["Monday","Tuesday","Wednesday",
                     "Thursday","Friday","Saturday","Sunday"]
        st.metric("📆 Day", day_names[selected_date.weekday()])

    # Bar chart
    fig, ax = plt.subplots(figsize=(8, 3))
    bars = ax.bar(
        ['Yesterday\n(Lag 1)', 'Last Week\n(Lag 7)', 
         'Last Month\n(Lag 30)', 'Prediction'],
        [lag_1, lag_7, lag_30, prediction],
        color=['steelblue', 'steelblue', 'steelblue', 'orange']
    )
    ax.set_ylabel('Consumption (GWh)')
    ax.set_title('Consumption Comparison')
    st.pyplot(fig)

else:
    st.info("👈 Enter values in sidebar and click Predict!")
    
    # Default info cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 Best Model Accuracy", "98.1%")
    with col2:
        st.metric("📊 Training Data", "4,383 Days")
    with col3:
        st.metric("🌍 Dataset", "Germany 2006-2017")