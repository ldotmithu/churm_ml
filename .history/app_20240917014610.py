import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.exceptions import InconsistentVersionWarning
import time

# Load the model and preprocessing pipeline
model = joblib.load(Path('arifacts/model_training/model.joblib'))
churn_preprocess = joblib.load(Path('arifacts/model_training/preprocess.pkl'))

# Custom CSS for styling
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://example.com/your_background_image.jpg");
        background-size: cover;
    }
    .stButton > button {
        background-color: #0099ff;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton > button:hover {
        background-color: #0077cc;
    }
    .stTitle {
        color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Churn Prediction Using Machine Learning")

tabs = st.tabs(["Prediction", "Feature Explanation", "Visualizations"])

with tabs[0]:
    st.header("Predict Churn")

    with st.form("user_input_form"):
        CreditScore = st.slider("Credit Score", min_value=300, max_value=850, value=600)
        Geography = st.radio("Geography", ['France', 'Germany', 'Spain'])
        Gender = st.radio("Gender", ['Male', 'Female'])
        Age = st.slider("Age", min_value=18, max_value=100, value=30)
        Tenure = st.slider("Tenure (Years)", min_value=0, max_value=10, value=5)
        Balance = st.number_input("Balance", value=10000.0)
        NumOfProducts = st.slider("Number of Products", min_value=1, max_value=4, value=1)
        HasCrCard = st.radio("Has Credit Card", [0, 1])
        IsActiveMember = st.radio("Is Active Member", [0, 1])
        EstimatedSalary = st.number_input("Estimated Salary", value=50000.0)

        submitted = st.form_submit_button("Predict")
        if submitted:
            with st.spinner("Making Prediction..."):
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

                input_data_preprocessed = churn_preprocess.transform(input_data)
                prediction_proba = model.predict_proba(input_data_preprocessed)[0][1]
                prediction = model.predict(input_data_preprocessed)[0]

                st.success(f"Prediction completed!")
                if prediction == 1:
                    st.warning(f"This customer is likely to churn with a probability of {prediction_proba:.2f}.")
                else:
                    st.success(f"This customer is likely to stay with a probability of {1 - prediction_proba:.2f}.")

with tabs[1]:
    st.header("Feature Explanations")
    st.write("""
    ### Feature Explanations:
    - **CreditScore**: Customer's credit score, ranging from 300 to 850. Higher credit scores indicate better creditworthiness.
    - **Geography**: Customer's location, such as France, Germany, or Spain. Geography can influence customer behavior and churn.
    - **Gender**: Customer's gender (Male or Female). Behavior and churn tendencies might differ between genders.
    - **Age**: Customer's age in years. Younger customers might have higher churn rates, while older customers may be more loyal.
    - **Tenure**: Number of years the customer has been with the company. Higher tenure might indicate stronger loyalty.
    - **Balance**: Account balance. Higher balances could indicate more engagement with the service.
    - **NumOfProducts**: Number of products the customer uses. More products usually mean higher engagement and lower churn.
    - **HasCrCard**: Whether the customer has a credit card with the company (1 = Yes, 0 = No). Having a credit card might reduce churn.
    - **IsActiveMember**: Whether the customer is an active member (1 = Yes, 0 = No). Active members are generally less likely to churn.
    - **EstimatedSalary**: The customer's annual estimated salary. Higher salaries might correlate with higher engagement.
    """)

with tabs[2]:
    st.header("Visualizations")

    # Feature Importance Plot
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_  # if using a tree-based model
        features = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

        fig, ax = plt.subplots()
        ax.barh(features, importances)
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance')
        st.pyplot(fig)

    # Correlation Heatmap
    sample_data = pd.DataFrame({
        'CreditScore': [CreditScore],
        'Age': [Age],
        'Tenure': [Tenure],
        'Balance': [Balance],
        'NumOfProducts': [NumOfProducts],
        'EstimatedSalary': [EstimatedSalary]
    })
    corr_matrix = sample_data.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Feature Correlation Heatmap')
    st.pyplot(fig)

    # Animated Plot Example
    fig = px.scatter(sample_data, x="Age", y="Balance", animation_frame="Tenure", 
                     title="Age vs Balance Over Tenure")
    st.plotly_chart(fig)
