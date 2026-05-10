import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load all models and preprocessing tools
models = {
    'Logistic Regression': joblib.load('logistic_regression.joblib'),
    'SVM': joblib.load('svm.joblib'),
    'Decision Tree': joblib.load('decision_tree.joblib'),
    'Random Forest': joblib.load('random_forest.joblib'),
}
scaler = joblib.load('scaler.joblib')
encoder = joblib.load('encoder.joblib')

st.title("🏦 Loan Approval Predictor")

# Model Selection
selected_model_name = st.selectbox("Select Model", list(models.keys()))
selected_model = models[selected_model_name]

# Input columns
col1, col2 = st.columns(2)
with col1:
    married = st.selectbox("Married", ["No", "Yes"])
    dependents = st.selectbox("Dependents", [0, 1, 2, 3])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])

with col2:
    applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
    coapplicant_income = st.number_input("Co-applicant Income", min_value=0, value=0)
    loan_amount = st.number_input("Loan Amount", min_value=0, value=120)
    term = st.number_input("Loan Term (Days)", min_value=0, value=360)

credit_history = st.selectbox("Credit History", ["No (0.0)", "Yes (1.0)"])
property_area = st.selectbox("Property Area", ["Semiurban", "Urban", "Rural"])

if st.button("Predict Loan Status"):
    # Preprocessing
    married_val = 1 if married == "Yes" else 0
    edu_val = 1 if education == "Graduate" else 0
    emp_val = 1 if self_employed == "Yes" else 0
    cred_val = 1.0 if "Yes" in credit_history else 0.0
    prop_map = {"Semiurban": 0, "Urban": 1, "Rural": 2}
    
    total_income = np.log1p(applicant_income + coapplicant_income)
    log_loan = np.log1p(loan_amount)
    
    # Create feature DataFrame
    user_data = pd.DataFrame({
        'Married': [married_val],
        'Dependents': [int(dependents)],
        'Education': [edu_val],
        'Self_Employed': [emp_val],
        'TotalIncome': [total_income],
        'LoanAmount': [log_loan],
        'Loan_Amount_Term': [term],
        'Credit_History': [cred_val]
    })
    
    # Scale numericals
    num_cols = ["TotalIncome", "LoanAmount", "Loan_Amount_Term"]
    user_data[num_cols] = scaler.transform(user_data[num_cols])
    
    # One-Hot Encode Property Area
    prop_df = pd.DataFrame({'Property_Area': [prop_map[property_area]]})
    encoded_prop = encoder.transform(prop_df)
    encoded_prop_df = pd.DataFrame(encoded_prop, columns=encoder.get_feature_names_out(['Property_Area']))
    
    # Combine user data and encoded property area
    final_input = pd.concat([user_data, encoded_prop_df], axis=1)
    
    # Reorder columns to match 'expected_columns'
    expected_columns = selected_model.feature_names_in_
    final_input = final_input[expected_columns]
    
    # Now predict
    prediction = selected_model.predict(final_input)
    
    if prediction[0] == 1:
        st.success("🎉 Loan Approved!")
        st.balloons()
    else:
        st.error("❌ Loan Rejected.")
        