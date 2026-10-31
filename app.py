import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.joblib"
CONFIG_PATH = BASE_DIR / "models" / "model_config.json"
DATA_PATH = BASE_DIR / "data" / "telco_churn_cleaned.csv"

FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges"
]

NUMERIC_COLUMNS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

CATEGORICAL_COLUMNS = [
    col for col in FEATURE_COLUMNS if col not in NUMERIC_COLUMNS
]

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide"
)

st.title("Customer Churn & Retention Intelligence")
st.caption("An ML-powered platform for customer churn prediction and retention analysis")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_config():

    defaults = {
        "threshold": 0.50,
        "risk_low_max": 0.30,
        "risk_medium_max": 0.60,
        "model_name": "Tuned Random Forest"
    }

    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as file:
            saved_config = json.load(file)

        defaults.update(saved_config)

    return defaults

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

def prepare_data(data):
    data = data.copy()

    data = data.drop(
        columns=["Churn", "customerID"],
        errors="ignore"
    )

    missing_columns = [
        col for col in FEATURE_COLUMNS
        if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    data = data[FEATURE_COLUMNS]

    for col in NUMERIC_COLUMNS:
        data[col] = pd.to_numeric(
            data[col], errors="coerce"
        ).fillna(0)

    for col in CATEGORICAL_COLUMNS:
        data[col] = data[col].fillna("Unknown").astype(str)

    return data

def assign_risk(probability, config):
    if probability < config["risk_low_max"]:
        return "Low"

    if probability < config["risk_medium_max"]:
        return "Medium"

    return "High"

def recommend_action(row):
    actions = []

    if row["Risk_Category"] == "High":
        actions.append("Prioritize retention outreach")

    if row["Contract"] == "Month-to-month":
        actions.append("Offer suitable long-term plan options")

    if row["OnlineSecurity"] == "No":
        actions.append("Offer online security guidance")

    if row["TechSupport"] == "No":
        actions.append("Offer technical support assistance")

    if row["tenure"] < 12:
        actions.append("Provide onboarding support")

    if row["MonthlyCharges"] > 75:
        actions.append("Review plan pricing and value")

    if not actions:
        actions.append("Continue regular customer engagement")

    return "; ".join(actions)

def score_customers(data, model, config):
    prepared = prepare_data(data)

    probabilities = model.predict_proba(prepared)[:, 1]

    result = data.copy()

    result["Churn_Probability"] = probabilities
    result["Predicted_Churn"] = np.where(
        probabilities >= config["threshold"],
        "Yes",
        "No"
    )

    result["Risk_Category"] = [
        assign_risk(p, config) for p in probabilities
    ]

    result["Recommended_Action"] = result.apply(
        recommend_action, axis=1
    )

    return result

try:
    model = load_model()
    config = load_config()
    df = load_data()

    df = score_customers(df, model, config)

except Exception as e:
    st.error(f"Application setup error: {e}")
    st.stop()

overview, individual, bulk, insights = st.tabs([
    "Business Overview",
    "Individual Prediction",
    "Bulk Prediction",
    "Retention Insights"
])

with overview:
    st.header("Business Overview")

    total_customers = len(df)
    predicted_churners = (
        df["Predicted_Churn"] == "Yes"
    ).sum()

    high_risk = (
        df["Risk_Category"] == "High"
    ).sum()

    predicted_rate = (
        predicted_churners / total_customers * 100
        if total_customers else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Predicted Churners", f"{predicted_churners:,}")
    col3.metric("Predicted Churn Rate", f"{predicted_rate:.2f}%")
    col4.metric("High-Risk Customers", f"{high_risk:,}")

    st.subheader("Customer Risk Distribution")

    risk_counts = (
        df["Risk_Category"]
        .value_counts()
        .reindex(["Low", "Medium", "High"], fill_value=0)
        .reset_index()
    )

    risk_counts.columns = ["Risk Category", "Customers"]

    fig = px.pie(
        risk_counts,
        names="Risk Category",
        values="Customers",
        title="Distribution by Risk Category",
        hole=0.4
    )

    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        contract_counts = df["Contract"].value_counts().reset_index()
        contract_counts.columns = ["Contract", "Customers"]

        fig = px.bar(
            contract_counts,
            x="Contract",
            y="Customers",
            title="Customers by Contract Type"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        risk_contract = pd.crosstab(
            df["Contract"],
            df["Risk_Category"]
        ).reset_index()

        fig = px.bar(
            risk_contract,
            x="Contract",
            y=["Low", "Medium", "High"],
            title="Risk Distribution by Contract",
            barmode="stack"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("High-Risk Customer List")

    high_risk_customers = df[
        df["Risk_Category"] == "High"
    ].sort_values(
        "Churn_Probability", ascending=False
    )

    display_columns = [
        col for col in [
            "customerID",
            "Contract",
            "tenure",
            "MonthlyCharges",
            "Churn_Probability",
            "Risk_Category",
            "Recommended_Action"
        ]
        if col in high_risk_customers.columns
    ]

    st.dataframe(
        high_risk_customers[display_columns].head(20),
        use_container_width=True
    )

with individual:
    st.header("Individual Customer Prediction")

    with st.form("customer_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox(
                "Gender", ["Male", "Female"]
            )
            senior = st.selectbox(
                "Senior Citizen", [0, 1]
            )
            partner = st.selectbox(
                "Partner", ["Yes", "No"]
            )
            dependents = st.selectbox(
                "Dependents", ["Yes", "No"]
            )
            tenure = st.number_input(
                "Tenure (months)", 0, 72, 12
            )
            phone = st.selectbox(
                "Phone Service", ["Yes", "No"]
            )
            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["Yes", "No", "No phone service"]
            )

        with col2:
            internet = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
            )
            security = st.selectbox(
                "Online Security",
                ["Yes", "No", "No internet service"]
            )
            backup = st.selectbox(
                "Online Backup",
                ["Yes", "No", "No internet service"]
            )
            device = st.selectbox(
                "Device Protection",
                ["Yes", "No", "No internet service"]
            )
            support = st.selectbox(
                "Tech Support",
                ["Yes", "No", "No internet service"]
            )
            tv = st.selectbox(
                "Streaming TV",
                ["Yes", "No", "No internet service"]
            )
            movies = st.selectbox(
                "Streaming Movies",
                ["Yes", "No", "No internet service"]
            )

        with col3:
            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"]
            )
            paperless = st.selectbox(
                "Paperless Billing", ["Yes", "No"]
            )
            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )
            monthly = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                value=70.0
            )
            total = st.number_input(
                "Total Charges",
                min_value=0.0,
                value=840.0
            )

        submitted = st.form_submit_button(
            "Predict Churn",
            use_container_width=True
        )

    if submitted:
        customer = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": device,
            "TechSupport": support,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total
        }])

        probability = model.predict_proba(
            prepare_data(customer)
        )[0, 1]

        risk = assign_risk(probability, config)

        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        col1.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%"
        )

        col2.metric("Risk Category", risk)

        if probability >= config["threshold"]:
            st.error("This customer is predicted to churn.")
        else:
            st.success("This customer is predicted to stay.")

        recommendation_row = customer.iloc[0].copy()
        recommendation_row["Risk_Category"] = risk
        recommendation_row["MonthlyCharges"] = monthly

        action = recommend_action(recommendation_row)

        st.info(f"Recommended Action: {action}")

with bulk:
    st.header("Bulk Customer Prediction")

    uploaded_file = st.file_uploader(
        "Upload customer CSV",
        type=["csv"]
    )

    if uploaded_file is not None:
        uploaded_df = pd.read_csv(uploaded_file)

        st.write("Uploaded Data Preview")
        st.dataframe(uploaded_df.head())

        if st.button("Generate Predictions"):
            try:
                result = score_customers(
                    uploaded_df, model, config
                )

                st.success("Predictions generated successfully!")

                col1, col2, col3 = st.columns(3)

                col1.metric("Customers", len(result))
                col2.metric(
                    "Predicted Churners",
                    (result["Predicted_Churn"] == "Yes").sum()
                )
                col3.metric(
                    "High-Risk Customers",
                    (result["Risk_Category"] == "High").sum()
                )

                st.dataframe(result, use_container_width=True)

                csv_data = result.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Download Predictions CSV",
                    data=csv_data,
                    file_name="churn_predictions.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"Prediction error: {e}")

with insights:
    st.header("Retention Intelligence")

    st.write(
        f"Model: {config.get('model_name', 'Churn Model')}"
    )

    st.write(
        f"Classification threshold: {config['threshold']:.2f}"
    )

    st.subheader("Churn Risk Insights")

    risk_summary = df.groupby(
        "Risk_Category"
    ).agg(
        Customers=("Risk_Category", "count"),
        Average_Probability=("Churn_Probability", "mean")
    ).reset_index()

    st.dataframe(risk_summary, use_container_width=True)

    st.subheader("Model Feature Importance")

    try:
        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]

        feature_names = preprocessor.get_feature_names_out()

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": classifier.feature_importances_
        }).sort_values(
            "Importance", ascending=False
        ).head(15)

        fig = px.bar(
            importance_df.sort_values("Importance"),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 15 Important Features"
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.warning(f"Feature importance unavailable: {e}")

    st.subheader("Retention Strategy")

    st.write(
        "Prioritize high-risk customers, investigate billing "
        "concerns, improve onboarding, and provide relevant "
        "support or service options."
    )

st.divider()

st.caption(
    "Customer Churn & Retention Intelligence | "
    "Machine Learning powered decision-support system"
)