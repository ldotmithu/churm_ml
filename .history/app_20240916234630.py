import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Load the trained model
model = joblib.load(Path('arifacts\model_training\model.joblib'))
churm_preprocess = joblib.load(Path('arifacts\model_training\preprocess.pkl'))

# Create the Streamlit UI
st.title("Churn Prediction App")

# Input fields for the features
CreditScore = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
Geography = st.selectbox("Geography", ['France', 'Germany', 'Spain'])
Gender = st.selectbox("Gender", ['Male', 'Female'])
Age = st.number_input("Age", min_value=18, max_value=100, value=30)
Tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=5)
Balance = st.number_input("Balance", value=10000.0)
NumOfProducts = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
HasCrCard = st.selectbox("Has Credit Card", [0, 1])
IsActiveMember = st.selectbox("Is Active Member", [0, 1])
EstimatedSalary = st.number_input("Estimated Salary", value=50000.0)

# Create a DataFrame from input data
input_data = pd.DataFrame({
    'CreditScore': [CreditScore],
    'Geography': [Geography],
    'Gender': [Gender],
    'Age': [Age],
    'Tenure': [Tenure],
    'Balance': [Balance],
    'NumOfProducts': [NumOfProducts],
    'HasCrCard': [HasCrCard],
    'IsActiveMember': [IsActiveMember],
    'EstimatedSalary': [EstimatedSalary]
})

# Preprocess the input data
input_data = pd.get_dummies(input_data, drop_first=True)

# Ensure all columns match the training set
required_columns = model.feature_names_in_
missing_cols = set(required_columns) - set(input_data.columns)
for col in missing_cols:
    input_data[col] = 0
input_data = input_data[required_columns]

# Predict churn
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.warning("This customer is likely to churn.")
    else:
        st.success("This customer is likely to stay.")
