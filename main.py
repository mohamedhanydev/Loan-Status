import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTENC
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
cat_cols = ["Married", "Dependents", "Education", "Self_Employed", "Credit_History", "Property_Area"]
num_cols = ["TotalIncome", "LoanAmount", "Loan_Amount_Term"]


# 1- Data loading
try:
    train_df = pd.read_csv("./train_data.csv")
    test_df = pd.read_csv("./test_data.csv")
    print("Data loaded successfully!\n")
except FileNotFoundError:
    print("Files not found.")
    exit(0)


# 2- EXPLORATORY DATA ANALYSIS (EDA)
def run_eda(df):
    """Generates visualizations to understand data distributions and relationships."""
    # Loan Approval
    plt.figure(figsize=(6, 4))
    df["Loan_Status"].value_counts().plot(kind="bar", color=["salmon", "steelblue"])
    plt.title("Loan Approval Distribution")
    plt.xlabel("0 = Rejected | 1 = Approved")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # Applicant Income Distribution
    plt.figure(figsize=(8, 5))
    plt.hist(df["ApplicantIncome"], bins=30, color="steelblue", edgecolor="black")
    plt.title("Applicant Income Distribution")
    plt.xlabel("Applicant Income")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # Loan Amount Distribution
    plt.figure(figsize=(8, 5))
    plt.hist(df["LoanAmount"], bins=30, color="salmon", edgecolor="black")
    plt.title("Loan Amount Distribution")
    plt.xlabel("Loan Amount")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # Correlation Heatmap
    plt.figure(figsize=(12, 9))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    # Total Income vs Loan Status
    plt.figure(figsize=(8, 6))
    sns.boxplot(
        x="Loan_Status",
        y="ApplicantIncome",
        data=df,
        hue="Loan_Status",
        palette="pastel",
        legend=False,
    )
    plt.title("Applicant Income vs Loan Status")
    plt.xlabel("Loan Status (0 = Rejected , 1 = Approved)")
    plt.ylabel("Applicant Income")
    plt.tight_layout()
    plt.show()

    # Categorical Impact Plots
    categorical_features = ['Property_Area', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History']
    for feature in categorical_features:
        plt.figure(figsize=(7, 5))
        sns.countplot(x=feature, hue='Loan_Status', data=df, palette='viridis')
        plt.title(f'Loan Status by {feature.replace("_", " ")}')
        plt.xlabel(feature.replace("_", " "))
        plt.ylabel('Count')
        plt.legend(title='Loan Status', labels=['Rejected', 'Approved'])
        plt.tight_layout()
        plt.show()

# 3- PREPROCESSING & PIPELINE
def base_cleaning(df):
    """Handles missing values, feature engineering, outliers, and basic mapping."""
    df_clean = df.copy()
    
    # Drop unnecessary features
    columns_to_drop = ["Loan_ID", "Gender"]
    for col in columns_to_drop:
        if col in df_clean.columns:
            df_clean = df_clean.drop(columns=[col])

    # Clean text anomalies
    if "Dependents" in df_clean.columns:
        df_clean["Dependents"] = df_clean["Dependents"].replace("3+", "3")


    # Impute Numerical Values
    for col in ["ApplicantIncome", "CoapplicantIncome","LoanAmount", "Loan_Amount_Term"]:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # Combine Incomes
    if 'ApplicantIncome' in df_clean.columns and 'CoapplicantIncome' in df_clean.columns:
        df_clean['TotalIncome'] = df_clean['ApplicantIncome'] + df_clean['CoapplicantIncome']
        df_clean = df_clean.drop(columns=['ApplicantIncome', 'CoapplicantIncome'])
    elif 'ApplicantIncome' in df_clean.columns:
        df_clean['TotalIncome'] = df_clean['ApplicantIncome']
        df_clean = df_clean.drop(columns=['ApplicantIncome'])
    elif 'CoapplicantIncome' in df_clean.columns:
        df_clean['TotalIncome'] = df_clean['CoapplicantIncome']
        df_clean = df_clean.drop(columns=['CoapplicantIncome'])


    # Impute Categorical Values
    for col in cat_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])


    # Outliers: Log Transformation
    outlier_cols = ["TotalIncome", "LoanAmount"]
    for col in outlier_cols:
        if col in df_clean.columns:
            df_clean[col] = np.log1p(df_clean[col])

    # Data Type Conversions
    df_clean["Dependents"] = df_clean["Dependents"].astype(int)

    # Basic Binary Label Encoding
    mapping_dict = {
        "Married": {"Yes": 1, "No": 0},
        "Education": {"Graduate": 1, "Not Graduate": 0},
        "Self_Employed": {"Yes": 1, "No": 0},
        "Property_Area": {"Semiurban": 0, "Urban": 1, "Rural": 2},
        "Loan_Status": {"Y": 1, "N": 0}
    }
    
    for col, val in mapping_dict.items():
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].map(val)

    return df_clean

# 4- MODEL EVALUATION & EXECUTION
def evaluate_results(model_name, y_true, y_pred):
    """Standardized output for model evaluation."""
    print(f"--- {model_name} Results ---")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred):.4f}\n")

def models():
    # Logistic Regression (Champion Model)
    lr_model = LogisticRegression(C=0.4, max_iter=100_000, random_state=RANDOM_STATE)
    lr_model.fit(X_train, y_train)
    lr_y_pred = lr_model.predict(X_test)
    evaluate_results("Logistic Regression", y_test, lr_y_pred)

    # Support Vector Machine (SVM)
    svm_model = svm.SVC(kernel="linear", C=1.0, random_state=RANDOM_STATE)
    svm_model.fit(X_train, y_train)
    svm_predictions = svm_model.predict(X_test)
    evaluate_results("SVM", y_test, svm_predictions)

    # Decision Tree
    dt_model = DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)
    dt_model.fit(X_train, y_train)
    dt_predictions = dt_model.predict(X_test)
    evaluate_results("Decision Tree", y_test, dt_predictions)

    # Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf_classifier.fit(X_train, y_train)
    rf_predictions = rf_classifier.predict(X_test)
    evaluate_results("Random Forest Classifier", y_test, rf_predictions)


# 1. Base Cleaning
train_clean = base_cleaning(train_df)
test_clean = base_cleaning(test_df)

# 2. Split Features and Target
X_train = train_clean.drop(columns=["Loan_Status"])
y_train = train_clean["Loan_Status"]

X_test = test_clean.drop(columns=["Loan_Status"])
y_test = test_clean["Loan_Status"]

# 3. Handle Imbalance (SMOTENC on TRAIN DATA ONLY)
actual_cat_cols = [col for col in cat_cols if col in X_train.columns]
sm = SMOTENC(random_state=RANDOM_STATE, categorical_features=actual_cat_cols)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# 4. Handle Text Categories (ONE-HOT ENCODING)
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Transform Train
train_encoded_data = encoder.fit_transform(X_train_res[['Property_Area']])
train_encoded_df = pd.DataFrame(train_encoded_data, columns=encoder.get_feature_names_out(['Property_Area']))
X_train_final = X_train_res.drop(columns=['Property_Area']).reset_index(drop=True)
X_train_final = pd.concat([X_train_final, train_encoded_df], axis=1)

# Transform Test
test_encoded_data = encoder.transform(X_test[['Property_Area']])
test_encoded_df = pd.DataFrame(test_encoded_data, columns=encoder.get_feature_names_out(['Property_Area']))
X_test_final = X_test.drop(columns=['Property_Area']).reset_index(drop=True)
X_test_final = pd.concat([X_test_final, test_encoded_df], axis=1)

# 5. Feature Scaling (Fit on Train, Transform on Both)
scaler = StandardScaler()
X_train_final[num_cols] = scaler.fit_transform(X_train_final[num_cols])
X_test_final[num_cols] = scaler.transform(X_test_final[num_cols])

X_train, y_train = X_train_final, y_train_res
X_test, y_test = X_test_final, y_test

models()
