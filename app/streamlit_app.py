import streamlit as st
import pandas as pd
import joblib

import pickle
with open('models/credit_risk_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title('Credit Risk Analyzer')

age = st.number_input('Age', min_value=18, max_value=100, value=30)
income = st.number_input('Annual Income', min_value=0, value=50000)
loan_amount = st.number_input('Loan Amount', min_value=0, value=10000)

if st.button('Predict'):
    input_data = pd.DataFrame([[age, income, loan_amount]], columns=['AGE', 'INCOME', 'LOAN_AMOUNT'])
    prediction = model.predict_proba(input_data)[0][1]
    st.write(f'Probability of Default: {prediction:.2%}')
