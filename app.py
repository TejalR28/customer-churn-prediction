import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# ===============================
# Load trained artifacts
# ===============================
artifacts = pickle.load(open("Customer_churn_model.pkl", "rb"))
model = artifacts['model']
scaler = pickle.load(open("scaler.pkl", "rb"))

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# ===============================
# Page config
# ===============================
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")
st.title("📊 Customer Churn Prediction Dashboard")

# ===============================
# Load & clean dataset
# ===============================
data = pd.read_csv("customer_churn.csv")

# Convert Churn to numeric if needed
if data['Churn'].dtype == 'object':
    data['Churn'] = data['Churn'].map({'No': 0, 'Yes': 1})

# FIX TotalCharges (string → numeric)
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['TotalCharges'].fillna(0, inplace=True)

# Remove ID column (never a feature)
if 'customerID' in data.columns:
    data.drop('customerID', axis=1, inplace=True)

# Feature columns EXACTLY like training
feature_columns = data.drop('Churn', axis=1).columns

# ===============================
# KPI Section
# ===============================
col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", len(data))
col2.metric("Churned Customers", int(data['Churn'].sum()))
col3.metric("Churn Rate", f"{data['Churn'].mean() * 100:.2f}%")

# ===============================
# Visualization
# ===============================
st.subheader("📉 Churn Distribution")

fig, ax = plt.subplots(figsize=(6, 4))
data['Churn'].value_counts().plot(kind='bar', ax=ax)
ax.set_xticklabels(['No Churn', 'Churn'], rotation=0)
ax.set_ylabel("Customers")
st.pyplot(fig)

# ===============================
# Prediction Section
# ===============================
st.subheader("🔮 Predict Customer Churn")
st.markdown("### Enter Customer Details")

input_data = {}

# Dynamic inputs for ALL features
for col in feature_columns:
    if col in encoders:  # categorical
        input_data[col] = st.selectbox(
            col,
            encoders[col].classes_
        )

    elif pd.api.types.is_numeric_dtype(data[col]):  # numeric
        min_val = float(data[col].min())
        max_val = float(data[col].max())
        default = float(data[col].median())

        input_data[col] = st.number_input(
            col,
            min_value=min_val,
            max_value=max_val,
            value=default
        )

# ===============================
# Prediction Logic
# ===============================
if st.button("Predict Churn"):
    input_df = pd.DataFrame([input_data])

    # Apply label encoding (same as training)
    for col, encoder in encoders.items():
        if col in input_df.columns:
            input_df[col] = encoder.transform(input_df[col])

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict probability
    churn_prob = model.predict_proba(input_scaled)[0][1]

    st.markdown("### Prediction Result")
    if churn_prob >= 0.5:
        st.error(f"⚠️ High Churn Risk: {churn_prob * 100:.2f}%")
    else:
        st.success(f"✅ Low Churn Risk: {(1 - churn_prob) * 100:.2f}%")
