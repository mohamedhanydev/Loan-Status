import streamlit as st
import joblib
import numpy as np
import pandas as pd
import sqlite3
import hashlib

# --- DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS users(username TEXT PRIMARY KEY, password TEXT)')
    conn.commit()
    conn.close()

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    return make_hashes(password) == hashed_text

def add_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('INSERT INTO users(username, password) VALUES (?,?)', (username, make_hashes(password)))
    conn.commit()
    conn.close()

def login_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username =?', (username,))
    data = c.fetchone()
    if data and check_hashes(password, data[1]):
        return True
    return False

# --- CONFIG & STYLING ---
st.set_page_config(page_title="LoanGuard AI", page_icon="🏦", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .stSelectbox, .stNumberInput { border-radius: 10px; }
    div[data-testid="stMetricValue"] { font-size: 1.5rem; }
    </style>
    """, unsafe_allow_html=True)

# --- APP LOGIC ---
def main():
    init_db()

    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if not st.session_state['logged_in']:
        show_auth_page()
    else:
        show_predictor_page()

def show_auth_page():
    st.title("🛡️ LoanGuard AI Portal")
    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        with st.form("login_form"):
            user = st.text_input("Username")
            pw = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                if login_user(user, pw):
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = user
                    st.rerun()
                else:
                    st.error("Invalid Username/Password")

    with tab2:
        with st.form("reg_form"):
            new_user = st.text_input("New Username")
            new_pw = st.text_input("New Password", type="password")
            if st.form_submit_button("Sign Up"):
                try:
                    add_user(new_user, new_pw)
                    st.success("Account created! Please login.")
                except:
                    st.warning("Username already exists.")

def show_predictor_page():
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=100)
        st.title(f"Welcome, {st.session_state['username']}")
        if st.button("Logout"):
            st.session_state['logged_in'] = False
            st.rerun()
        
        st.divider()
        st.info("This tool uses ML to predict loan eligibility based on historical data.")

    # Main Body
    st.title("🏦 Loan Approval Intelligence")
    st.markdown("---")

    # Load resources (wrapped in try-except for safety)
    try:
        models = {
            'Logistic Regression': joblib.load('logistic_regression.joblib'),
            'SVM': joblib.load('svm.joblib'),
            'Decision Tree': joblib.load('decision_tree.joblib'),
            'Random Forest': joblib.load('random_forest.joblib'),
        }
        scaler = joblib.load('scaler.joblib')
        encoder = joblib.load('encoder.joblib')
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return

    # Layout
    col_l, col_r = st.columns([1, 2], gap="large")

    with col_l:
        st.subheader("Model Settings")
        selected_model_name = st.selectbox("Intelligence Engine", list(models.keys()))
        selected_model = models[selected_model_name]
        
        st.subheader("Personal Details")
        married = st.selectbox("Marital Status", ["No", "Yes"])
        dependents = st.select_slider("Number of Dependents", options=[0, 1, 2, 3])
        education = st.radio("Education Level", ["Graduate", "Not Graduate"], horizontal=True)
        self_employed = st.radio("Self Employed", ["No", "Yes"], horizontal=True)

    with col_r:
        st.subheader("Financial Profile")
        c1, c2 = st.columns(2)
        with c1:
            applicant_income = st.number_input("Applicant Income ($)", min_value=0, value=5000, step=500)
            loan_amount = st.number_input("Loan Amount ($1000s)", min_value=0, value=120)
        with c2:
            coapplicant_income = st.number_input("Co-applicant Income ($)", min_value=0, value=0)
            term = st.number_input("Term (Days)", min_value=0, value=360)

        credit_history = st.selectbox("Credit History Clean?", ["No (0.0)", "Yes (1.0)"])
        property_area = st.segmented_control("Property Location", ["Semiurban", "Urban", "Rural"])

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Analyze Application"):
            # Preprocessing Logic
            married_val = 1 if married == "Yes" else 0
            edu_val = 1 if education == "Graduate" else 0
            emp_val = 1 if self_employed == "Yes" else 0
            cred_val = 1.0 if "Yes" in credit_history else 0.0
            prop_map = {"Semiurban": 0, "Urban": 1, "Rural": 2}
            
            total_income = np.log1p(applicant_income + coapplicant_income)
            log_loan = np.log1p(loan_amount)
            
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
            
            num_cols = ["TotalIncome", "LoanAmount", "Loan_Amount_Term"]
            user_data[num_cols] = scaler.transform(user_data[num_cols])
            
            prop_df = pd.DataFrame({'Property_Area': [prop_map[property_area]]})
            encoded_prop = encoder.transform(prop_df)
            encoded_prop_df = pd.DataFrame(encoded_prop, columns=encoder.get_feature_names_out(['Property_Area']))
            
            final_input = pd.concat([user_data, encoded_prop_df], axis=1)
            final_input = final_input[selected_model.feature_names_in_]
            
            prediction = selected_model.predict(final_input)
            
            st.divider()
            if prediction[0] == 1:
                st.balloons()
                st.success("### ✅ Application Approved!")
                st.write("The model suggests high confidence in this candidate's repayment ability.")
            else:
                st.error("### ❌ Application Rejected.")
                st.write("Based on the provided data, this application does not meet the approval criteria.")

if __name__ == "__main__":
    main()