# changes: remove gender, add applicant income and coapplicant income into one column called total income.
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.tree import DecisionTreeClassifier, plot_tree

RANDOM_STATE = 42
cat_cols = [
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Credit_History",
    "Property_Area",
]
num_cols = ["TotalIncome", "LoanAmount", "Loan_Amount_Term"]
# ==========================================
# SECTION 1: DATA LOADING
# ==========================================
try:
    train_df = pd.read_csv("train_data.csv")
    test_df = pd.read_csv("test_data.csv")
    print("Data loaded successfully!")
except FileNotFoundError:
    print("Files not found.")

# ==========================================
# SECTION 2: EDA (Owner: Omar Wael)
# ==========================================
def run_eda(df):
    # Loan Approval #
    plt.figure(figsize=(6, 4))
    df["Loan_Status"].value_counts().plot(kind="bar", color=["salmon", "steelblue"])
    plt.title("Loan Approval Distribution")
    plt.xlabel("0 = Rejected | 1 = Approved")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # Applicant Income Distribution #
    plt.figure(figsize=(8, 5))
    plt.hist(df["ApplicantIncome"], bins=30, color="steelblue", edgecolor="black")
    plt.title("Applicant Income Distribution")
    plt.xlabel("Applicant Income")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # Loan Amount Distribution #
    plt.figure(figsize=(8, 5))
    plt.hist(df["LoanAmount"], bins=30, color="salmon", edgecolor="black")
    plt.title("Loan Amount Distribution")
    plt.xlabel("Loan Amount")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # Correlation Heatmap #
    plt.figure(figsize=(12, 9))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    # Total Income vs Loan Status #
    plt.figure(figsize=(8, 6))
    sns.boxplot(
        x="Loan_Status",
        y="ApplicantIncome",
        data=df,
        hue="Loan_Status",
        palette="pastel",
        legend=False,
    )
    plt.title("Total Income vs Loan Status")
    plt.xlabel("Loan Status (0 = Rejected , 1 = Approved)")
    plt.ylabel("Total Income")
    plt.tight_layout()
    plt.show()

    # Property Area vs Loan Status #
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Property_Area', hue='Loan_Status', data=df, palette='viridis')
    plt.title('Loan Status by Property Area')
    plt.xlabel('Property Area')
    plt.ylabel('Count')
    plt.legend(title='Loan Status', labels=['Rejected', 'Approved'])
    plt.tight_layout()
    plt.show()

    # Married vs Loan Status #
    plt.figure(figsize=(7, 5))
    sns.countplot(x='Married', hue='Loan_Status', data=df, palette='viridis')
    plt.title('Loan Status by Marital Status')
    plt.xlabel('Married')
    plt.ylabel('Count')
    plt.legend(title='Loan Status', labels=['Rejected', 'Approved'])
    plt.xticks(ticks=[0, 1], labels=['No', 'Yes'])
    plt.tight_layout()
    plt.show()

    # Dependents vs Loan Status #
    plt.figure(figsize=(7, 5))
    sns.countplot(x='Dependents', hue='Loan_Status', data=df, palette='viridis')
    plt.title('Loan Status by Dependents')
    plt.xlabel('Dependents')
    plt.ylabel('Count')
    plt.legend(title='Loan Status', labels=['Rejected', 'Approved'])
    plt.tight_layout()
    plt.show()

    # Education vs Loan Status #
    plt.figure(figsize=(7, 5))
    sns.countplot(x='Education', hue='Loan_Status', data=df, palette='viridis')
    plt.title('Loan Status by Education')
    plt.xlabel('Education')
    plt.ylabel('Count')
    plt.legend(title='Loan Status', labels=['Rejected', 'Approved'])
    plt.xticks(ticks=[0, 1], labels=['Not Graduate', 'Graduate'])
    plt.tight_layout()
    plt.show()

    # Self_Employed vs Loan Status #
    plt.figure(figsize=(7, 5))
    sns.countplot(x='Self_Employed', hue='Loan_Status', data=df, palette='viridis')
    plt.title('Loan Status by Self-Employment')
    plt.xlabel('Self_Employed')
    plt.ylabel('Count')
    plt.legend(title='Loan Status', labels=['Rejected', 'Approved'])
    plt.xticks(ticks=[0, 1], labels=['No', 'Yes'])
    plt.tight_layout()
    plt.show()

    # Credit_History vs Loan Status #
    plt.figure(figsize=(7, 5))
    sns.countplot(x='Credit_History', hue='Loan_Status', data=df, palette='viridis')
    plt.title('Loan Status by Credit History')
    plt.xlabel('Credit_History')
    plt.ylabel('Count')
    plt.legend(title='Loan Status', labels=['Rejected', 'Approved'])
    plt.xticks(ticks=[0, 1], labels=['No', 'Yes'])
    plt.tight_layout()
    plt.show()

# ==========================================
# SECTION 3: PREPROCESSING & PIPELINE
# ==========================================

def base_cleaning(df):
    """Handles missing values, outliers, and basic mapping"""
    df_clean = df.copy()
    if "Loan_ID" in df_clean.columns:
        df_clean = df_clean.drop(columns=["Loan_ID"])

    if "Dependents" in df_clean.columns:
        df_clean["Dependents"] = df_clean["Dependents"].replace("3+", "3")

    # Drop rows with >3 missing values
    df_clean = df_clean[df_clean.isnull().sum(axis=1) < 3]

    # Impute Missing Values for individual income columns before creating TotalIncome
    if "ApplicantIncome" in df_clean.columns:
        df_clean["ApplicantIncome"] = df_clean["ApplicantIncome"].fillna(df_clean["ApplicantIncome"].median())
    if "CoapplicantIncome" in df_clean.columns:
        df_clean["CoapplicantIncome"] = df_clean["CoapplicantIncome"].fillna(df_clean["CoapplicantIncome"].median())

    # Create TotalIncome and drop original income columns
    if 'ApplicantIncome' in df_clean.columns and 'CoapplicantIncome' in df_clean.columns:
        df_clean['TotalIncome'] = df_clean['ApplicantIncome'] + df_clean['CoapplicantIncome']
        df_clean = df_clean.drop(columns=['ApplicantIncome', 'CoapplicantIncome'])
    elif 'ApplicantIncome' in df_clean.columns:
        df_clean['TotalIncome'] = df_clean['ApplicantIncome']
        df_clean = df_clean.drop(columns=['ApplicantIncome'])
    elif 'CoapplicantIncome' in df_clean.columns:
        df_clean['TotalIncome'] = df_clean['CoapplicantIncome']
        df_clean = df_clean.drop(columns=['CoapplicantIncome'])

    # Impute other Missing Numerical Values (LoanAmount, Loan_Amount_Term)
    for col in ["LoanAmount", "Loan_Amount_Term"]:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    for col in cat_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

    # Drop Gender column
    if "Gender" in df_clean.columns:
        df_clean = df_clean.drop(columns=["Gender"])

    df_clean = df_clean.dropna()

    # Outliers (Log Transform)
    outlier_cols = ["TotalIncome", "LoanAmount"]
    for col in outlier_cols:
        if col in df_clean.columns:
            df_clean[col] = np.log1p(df_clean[col])

    df_clean["Dependents"] = df_clean["Dependents"].astype(int)

    # Basic Label Encoding
    df_clean["Married"] = df_clean["Married"].map({"Yes": 1, "No": 0})
    df_clean["Education"] = df_clean["Education"].map({"Graduate": 1, "Not Graduate": 0})
    df_clean["Self_Employed"] = df_clean["Self_Employed"].map({"Yes": 1, "No": 0})
    df_clean["Property_Area"] = df_clean["Property_Area"].map({"Semiurban": 0, "Urban": 1, "Rural": 2})

    if "Loan_Status" in df_clean.columns:
        df_clean["Loan_Status"] = df_clean["Loan_Status"].map({"Y": 1, "N": 0})

    return df_clean

# 1. Base Cleaning
train_clean = base_cleaning(train_df)
test_clean = base_cleaning(test_df)

# 2. Split Features and Target
X_train = train_clean.drop(columns=["Loan_Status"])
y_train = train_clean["Loan_Status"]

X_test = test_clean.drop(columns=["Loan_Status"])
y_test = test_clean["Loan_Status"]

# 3. SMOTENC (TRAIN DATA ONLY)
actual_cat_cols = [col for col in cat_cols if col in X_train.columns]
sm = SMOTENC(random_state=RANDOM_STATE, categorical_features=actual_cat_cols)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# 4. ONE HOT ENCODING (Fit on Train, Transform on Both)
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Transform Train
train_encoded_data = encoder.fit_transform(X_train_res[['Property_Area']])
train_encoded_df = pd.DataFrame(train_encoded_data, columns=encoder.get_feature_names_out(['Property_Area']))
X_train_final = X_train_res.drop(columns=['Property_Area']).reset_index(drop=True)
X_train_final = pd.concat([X_train_final, train_encoded_df], axis=1)

# Transform Test (NO FIT, NO SMOTE)
test_encoded_data = encoder.transform(X_test[['Property_Area']])
test_encoded_df = pd.DataFrame(test_encoded_data, columns=encoder.get_feature_names_out(['Property_Area']))
X_test_final = X_test.drop(columns=['Property_Area']).reset_index(drop=True)
X_test_final = pd.concat([X_test_final, test_encoded_df], axis=1)

# 5. SCALING (Fit on Train, Transform on Both)
scaler = StandardScaler()
X_train_final[num_cols] = scaler.fit_transform(X_train_final[num_cols])
X_test_final[num_cols] = scaler.transform(X_test_final[num_cols])

X_train, y_train = X_train_final, y_train_res
X_test, y_test = X_test_final, y_test

# ==========================================
# UTILITY: Standard Evaluation Function
# ==========================================
def evaluate_results(model_name, y_true, y_pred):
    print(f"--- {model_name} Results ---")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred):.4f}")
    # print("\nConfusion Matrix:")
    # sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues")
    # plt.show()

    # print(classification_report(y_true, y_pred))


# ==========================================
# SECTION 4: LOGISTIC REGRESSION
# ==========================================
lr_model = LogisticRegression(C=0.4, max_iter=100_000)
lr_model.fit(X_train, y_train)
lr_y_pred = lr_model.predict(X_test)
evaluate_results("Logistic Regression", y_test, lr_y_pred)

# ==========================================
# SECTION 5: SVM
# ==========================================

# create the svm model and fit the data
svm_model = svm.SVC(kernel="linear", C=1.0)
svm_model.fit(X_train, y_train)

# Make Predictions
predictions = svm_model.predict(X_test)
# Evaluate Accuracy
evaluate_results("SVM", y_test, predictions)

# ==========================================
# SECTION 6: DECISION TREE
# ==========================================

dt_model = DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)
evaluate_results("Decision Tree", y_test, dt_predictions)

# plt.figure(figsize=(20, 10))

# plot_tree(
#     dt_model,
#     filled=True,
#     feature_names=X_train.columns,
#     class_names=["Reject", "Approve"],
#     rounded=True,
#     max_depth=3,
#     fontsize=10,
# )

# plt.title("Decision Tree: Loan Approval Logic")
# plt.show()

# ==========================================
# SECTION 7: RandomForestClassifier
# ==========================================

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)
evaluate_results("RandomForestClassifier", y_test, y_pred)

# Save all models and the preprocessing objects
joblib.dump(lr_model, 'logistic_regression.joblib')
joblib.dump(svm_model, 'svm.joblib')
joblib.dump(dt_model, 'decision_tree.joblib')
joblib.dump(rf_classifier, 'random_forest.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(encoder, 'encoder.joblib')