import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score
import datetime
import io

# Set page config
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# Title
st.title("Customer Churn Prediction Dashboard")

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select a page:", 
                          ["Data Overview", "Data Exploration", "Model Training", "Make Predictions"])

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv('Model_Data.csv')
    return data

data = load_data()

# Data preprocessing function
def preprocess_data(data):
    # Add dummy customer_id if missing
    if 'customer_id' not in data.columns:
        data['customer_id'] = 9999  # Dummy ID for prediction

    # Convert to datetime, handle errors if any
    data['TheLastTransDate'] = pd.to_datetime(data['TheLastTransDate'], errors='coerce')
    data['Days_Since_Last_Transaction'] = (pd.Timestamp.now() - data['TheLastTransDate']).dt.days

    # Drop original datetime column to avoid dtype errors
    if 'TheLastTransDate' in data.columns:
        data.drop(columns=['TheLastTransDate'], inplace=True)

    # Encoding categorical variables for initial columns
    data = pd.get_dummies(data, columns=['interaction_type', 'payment_method', 'payment_status'], drop_first=True)

    # Aggregate transaction data if multiple customers exist
    if data['customer_id'].nunique() > 1:
        user_agg = data.groupby('customer_id').agg({
            'Transaction_type': lambda x: (x == 'Purchase').sum(),
            'interaction_outcome': lambda x: (x == 'Completed').sum(),
            'Months_Since_Last_Transaction': 'mean',
            'Days_Since_Last_Transaction': 'mean'
        }).reset_index()

        user_agg.rename(columns={
            'Transaction_type': 'Total_Purchases',
            'interaction_outcome': 'Successful_Interactions'
        }, inplace=True)

        data = data.merge(user_agg, on='customer_id', how='left')
        data.rename(columns={
            'Months_Since_Last_Transaction_x': 'Months_Since_Last_Transaction',
            'Days_Since_Last_Transaction_x': 'Days_Since_Last_Transaction'
        }, inplace=True)
    else:
        # For single customer input, create the columns manually
        data['Total_Purchases'] = (data['Transaction_type'] == 'Purchase').astype(int)
        data['Successful_Interactions'] = (data['interaction_outcome'] == 'Completed').astype(int)

    # Ensure numeric columns exist before scaling
    numeric_columns = ['Months_Since_Last_Transaction', 'Days_Since_Last_Transaction',
                       'Total_Purchases', 'Successful_Interactions']
    for col in numeric_columns:
        if col not in data.columns:
            data[col] = 0

    # Scale numeric columns
    scaler = StandardScaler()
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

    # Encode more categorical columns
    data = pd.get_dummies(data, columns=['interaction_outcome', 'Transaction_type'], drop_first=True)

    return data

# Data Overview Page
if options == "Data Overview":
    st.header("Data Overview")

    st.subheader("First 5 rows of the dataset")
    st.write(data.head())

    st.subheader("Dataset Information")
    buffer = io.StringIO()
    data.info(buf=buffer)
    st.text(buffer.getvalue())

    st.subheader("Descriptive Statistics")
    st.write(data.describe())

    st.subheader("Categorical Variables Summary")
    st.write(data.select_dtypes(include=['object']).describe())

    st.subheader("Target Variable Distribution")
    st.bar_chart(data['Target'].value_counts())

# Data Exploration Page
elif options == "Data Exploration":
    st.header("Data Exploration")

    st.subheader("Univariate Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Numerical Variables Histograms")
        numerical_data = data.select_dtypes(include='number')
        fig, ax = plt.subplots(figsize=(10, 8))
        numerical_data.hist(color='b', ax=ax)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.write("Numerical Variables Boxplots")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.boxplot(data=numerical_data, ax=ax)
        st.pyplot(fig)

    st.subheader("Categorical Variables Countplots")
    categorical_data = data.select_dtypes(include='object')
    for column in categorical_data.columns:
        fig, ax = plt.subplots()
        sns.countplot(data=categorical_data, x=column, palette="Set1", ax=ax)
        plt.title(f"Countplot of {column}")
        st.pyplot(fig)

    st.subheader("Multivariate Analysis")

    st.write("Pairplot of Numerical Variables")
    fig = sns.pairplot(data.select_dtypes(include='number'))
    st.pyplot(fig)

    st.write("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numerical_data.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Model Training Page
elif options == "Model Training":
    st.header("Model Training")

    # Preprocess data
    processed_data = preprocess_data(data)
    X = processed_data.drop(columns=['Target', 'customer_id'])
    y = processed_data['Target']

    test_size = st.slider("Select test size ratio:", 0.1, 0.5, 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    model_name = st.selectbox("Select a model:", 
                            ["Logistic Regression", "Random Forest", 
                             "Support Vector Machine", "K-Nearest Neighbors"])

    if st.button("Train Model"):
        st.write(f"Training {model_name}...")

        if model_name == "Logistic Regression":
            model = LogisticRegression(random_state=42, class_weight='balanced')
        elif model_name == "Random Forest":
            model = RandomForestClassifier(random_state=42, class_weight='balanced')
        elif model_name == "Support Vector Machine":
            model = SVC(random_state=42, class_weight='balanced', probability=True)
        elif model_name == "K-Nearest Neighbors":
            model = KNeighborsClassifier()

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)

        st.success("Model trained successfully!")
        st.subheader("Model Performance")
        st.write(f"F1 Score: {f1*100:.2f}%")

        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.table(pd.DataFrame(report).transpose())

        if model_name == "Random Forest":
            st.subheader("Feature Importance")
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=feature_importance.head(10), x='Importance', y='Feature', ax=ax)
            plt.title("Top 10 Important Features")
            st.pyplot(fig)

# Make Predictions Page
elif options == "Make Predictions":
    st.header("Make Predictions on New Data")

    processed_data = preprocess_data(data)
    X = processed_data.drop(columns=['Target', 'customer_id'])

    st.subheader("Enter Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        last_trans_date = st.date_input("Last Transaction Date", datetime.date.today())
        months_since_last = st.number_input("Months Since Last Transaction", min_value=0, max_value=120, value=1)
        total_purchases = st.number_input("Total Purchases", min_value=0, value=1)

    with col2:
        successful_interactions = st.number_input("Successful Interactions", min_value=0, value=1)
        payment_method = st.selectbox("Payment Method", ["Credit Card", "Bank Transfer", "PayPal"])
        payment_status = st.selectbox("Payment Status", ["Completed", "Pending", "Failed"])

    interaction_type = st.selectbox("Interaction Type", ["Email", "Call", "Chat"])
    transaction_type = st.selectbox("Transaction Type", ["Purchase", "Refund", "Query"])
    interaction_outcome = st.selectbox("Interaction Outcome", ["Completed", "Failed"])

    model_name = st.selectbox("Select Model for Prediction:", 
                            ["Logistic Regression", "Random Forest", 
                             "Support Vector Machine", "K-Nearest Neighbors"])

    if st.button("Predict Churn"):
        input_data = {
            'TheLastTransDate': [last_trans_date],
            'Months_Since_Last_Transaction': [months_since_last],
            'Total_Purchases': [total_purchases],
            'Successful_Interactions': [successful_interactions],
            'payment_method': [payment_method],
            'payment_status': [payment_status],
            'interaction_type': [interaction_type],
            'Transaction_type': [transaction_type],
            'interaction_outcome': [interaction_outcome]
        }

        input_df = pd.DataFrame(input_data)

        # Add dummy customer_id
        input_df['customer_id'] = 9999

        processed_input = preprocess_data(input_df)

        # Align input columns with model expected columns
        expected_columns = X.columns
        for col in expected_columns:
            if col not in processed_input.columns:
                processed_input[col] = 0

        processed_input = processed_input[expected_columns]

        # Prepare training set for model training before prediction
        X_train, X_test, y_train, y_test = train_test_split(X, processed_data['Target'], test_size=0.2, random_state=42)

        if model_name == "Logistic Regression":
            model = LogisticRegression(random_state=42, class_weight='balanced')
        elif model_name == "Random Forest":
            model = RandomForestClassifier(random_state=42, class_weight='balanced')
        elif model_name == "Support Vector Machine":
            model = SVC(random_state=42, class_weight='balanced', probability=True)
        elif model_name == "K-Nearest Neighbors":
            model = KNeighborsClassifier()

        model.fit(X_train, y_train)

        prediction = model.predict(processed_input)
        prediction_proba = model.predict_proba(processed_input)

        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.error("Prediction: Customer is likely to churn")
        else:
            st.success("Prediction: Customer is not likely to churn")

        st.write(f"Probability of churn: {prediction_proba[0][1]*100:.2f}%")
