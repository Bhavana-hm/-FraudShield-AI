
from pathlib import Path

import joblib
import streamlit as st
import pandas as pd
import plotly.express as px


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide"
)


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "random_forest_model.pkl"
FEATURE_PATH = BASE_DIR / "models" / "feature_columns.pkl"
THRESHOLD_PATH = BASE_DIR / "models" / "best_threshold.pkl"
DATASET_PATH = BASE_DIR / "dataset" / "creditcard.csv"
REPORT_PATH = BASE_DIR / "reports" / "model_comparison.csv"


# --------------------------------------------------
# LOAD MODEL FILES
# --------------------------------------------------

@st.cache_resource
def load_model_files():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURE_PATH)
    threshold = joblib.load(THRESHOLD_PATH)

    return model, features, threshold


@st.cache_data
def load_dataset():
    if DATASET_PATH.exists():
        return pd.read_csv(DATASET_PATH)

    return None


try:
    model, feature_columns, threshold = load_model_files()

except Exception as error:
    st.error(f"Model loading error: {error}")
    st.stop()


# --------------------------------------------------
# CUSTOM DESIGN
# --------------------------------------------------

st.markdown("""
<style>

[data-testid="stAppViewContainer"] {
    background-color: #f4f7fc;
}

[data-testid="stAppViewContainer"] .stMarkdown,
[data-testid="stAppViewContainer"] label {
    color: #172554 !important;
}

[data-testid="stAppViewContainer"] h1 {
    color: #12346b !important;
    font-weight: 800;
}

[data-testid="stMetric"] {
    background-color: #ffffff !important;
    padding: 20px !important;
    border-radius: 12px !important;
    border: 1px solid #dbe3f0 !important;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"] {
    color: #172554 !important;
}

[data-testid="stNumberInput"] input {
    color: #172554 !important;
    background-color: #ffffff !important;
}

[data-testid="stSidebar"] {
    background-color: #202536 !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: #ffffff !important;
}

.stButton > button,
.stFormSubmitButton > button {
    background-color: #1746a2 !important;
    color: white !important;
    border-radius: 8px;
    border: none;
    font-weight: 600;
    padding: 10px 20px;
}

[data-testid="stAlert"] {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def get_risk_level(probability):

    if probability >= threshold:
        return "High Risk"

    elif probability >= threshold * 0.5:
        return "Medium Risk"

    return "Low Risk"


def show_prediction(probability):

    st.metric(
        "Fraud Probability",
        f"{probability * 100:.2f}%"
    )

    risk = get_risk_level(probability)

    if risk == "High Risk":
        st.error("🚨 Potential Fraud Detected")

    elif risk == "Medium Risk":
        st.warning("⚠️ Transaction Requires Review")

    else:
        st.success("✅ Transaction Classified as Genuine")

    st.write("Risk Level:", risk)


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("🛡️ FraudShield AI")

st.sidebar.caption(
    "Intelligent Credit Card Fraud Detection"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Fraud Detection",
        "Bulk Detection",
        "Analytics",
        "Model Performance"
    ]
)

st.sidebar.divider()

st.sidebar.write("Model: Random Forest")
st.sidebar.write(f"Threshold: {threshold:.2f}")
st.sidebar.caption("Academic Project")


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

if page == "Home":

    st.title("🛡️ FraudShield AI")

    st.subheader(
        "Intelligent Credit Card Fraud Detection System"
    )

    st.write(
        "An AI-powered system that analyses transaction "
        "features and predicts the probability of credit card fraud."
    )

    st.info(
        "This is an academic demonstration using an anonymized "
        "transaction dataset. It is not connected to real banks."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Selected Model", "Random Forest")
    c2.metric("Precision", "91.36%")
    c3.metric("Recall", "77.89%")
    c4.metric("F1 Score", "84.09%")

    st.divider()

    st.subheader("System Features")

    c1, c2 = st.columns(2)

    with c1:

        st.markdown("### 🔍 Single Transaction Detection")
        st.write("Analyse individual transaction features.")

        st.markdown("### 📁 Bulk Detection")
        st.write("Upload a CSV file to analyse multiple transactions.")

    with c2:

        st.markdown("### 📊 Fraud Analytics")
        st.write("Explore fraud statistics and transaction patterns.")

        st.markdown("### 📈 Model Performance")
        st.write("Compare five machine-learning algorithms.")


# --------------------------------------------------
# SINGLE TRANSACTION DETECTION
# --------------------------------------------------

elif page == "Fraud Detection":

    st.title("🔍 Fraud Detection")

    st.write(
        "Load a sample transaction or enter transaction features "
        "to estimate fraud probability."
    )

    st.warning(
        "V1–V28 are anonymized PCA features, not ordinary card details."
    )

    # Initialize session state
    if "sample_transaction" not in st.session_state:
        st.session_state.sample_transaction = None

    if "sample_version" not in st.session_state:
        st.session_state.sample_version = 0

    # Load sample transaction button
    sample_df = load_dataset()

    if sample_df is not None:

        if st.button("📥 Load Sample Transaction"):

            st.session_state.sample_transaction = (
                sample_df.iloc[0].to_dict()
            )

            # New widget keys make the sample values appear
            st.session_state.sample_version += 1

            st.success("Sample transaction loaded successfully!")

    else:

        st.warning(
            "Dataset not found. Please check the dataset folder."
        )

    # Get sample values
    sample = st.session_state.sample_transaction or {}

    version = st.session_state.sample_version

    # Transaction input form
    with st.form("transaction_form"):

        st.subheader("Transaction Information")

        time_value = st.number_input(
            "Transaction Time",
            min_value=0.0,
            value=float(sample.get("Time", 0.0)),
            key=f"time_{version}"
        )

        amount_value = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=float(sample.get("Amount", 0.0)),
            key=f"amount_{version}"
        )

        st.subheader("Anonymized Transaction Features")

        feature_values = {}

        v_features = [
            col for col in feature_columns
            if col.startswith("V")
        ]

        columns = st.columns(3)

        for index, feature in enumerate(v_features):

            with columns[index % 3]:

                feature_values[feature] = st.number_input(
                    feature,
                    value=float(sample.get(feature, 0.0)),
                    format="%.6f",
                    key=f"{feature}_{version}"
                )

        submitted = st.form_submit_button(
            "🔍 Analyse Transaction",
            use_container_width=True
        )

    # Prediction
    if submitted:

        values = {
            "Time": time_value,
            "Amount": amount_value,
            **feature_values
        }

        try:

            input_df = pd.DataFrame(
                [[values[col] for col in feature_columns]],
                columns=feature_columns
            )

            probability = float(
                model.predict_proba(input_df)[0][1]
            )

            st.divider()

            st.subheader("Prediction Result")

            show_prediction(probability)

        except Exception as error:

            st.error(f"Prediction error: {error}")


# --------------------------------------------------
# BULK CSV DETECTION
# --------------------------------------------------

elif page == "Bulk Detection":

    st.title("📁 Bulk Fraud Detection")

    st.write(
        "Upload a CSV file to analyse multiple transactions."
    )

    uploaded_file = st.file_uploader(
        "Upload transaction CSV",
        type=["csv"]
    )

    st.info(
        "CSV must contain Time, V1–V28 and Amount columns. "
        "An optional Class column will be excluded from prediction."
    )

    if uploaded_file is not None:

        try:

            uploaded_df = pd.read_csv(uploaded_file)

            # Remove actual labels before prediction
            if "Class" in uploaded_df.columns:
                uploaded_df = uploaded_df.drop(columns=["Class"])

            # Check required columns
            missing = [
                col for col in feature_columns
                if col not in uploaded_df.columns
            ]

            if missing:

                st.error(
                    "Missing columns: " + ", ".join(missing)
                )

            elif uploaded_df.empty:

                st.error("The CSV contains no transactions.")

            else:

                input_df = uploaded_df[feature_columns].apply(
                    pd.to_numeric,
                    errors="coerce"
                )

                if input_df.isnull().any().any():

                    st.error(
                        "Please remove missing or non-numeric values."
                    )

                elif st.button("Analyse Transactions"):

                    probabilities = model.predict_proba(
                        input_df
                    )[:, 1]

                    results = uploaded_df.copy()

                    results["Fraud_Probability"] = probabilities

                    results["Predicted_Class"] = (
                        probabilities >= threshold
                    ).astype(int)

                    results["Risk_Level"] = [
                        get_risk_level(p)
                        for p in probabilities
                    ]

                    fraud_count = int(
                        results["Predicted_Class"].sum()
                    )

                    c1, c2, c3 = st.columns(3)

                    c1.metric("Total Transactions", len(results))

                    c2.metric(
                        "Potential Fraud",
                        fraud_count
                    )

                    c3.metric(
                        "Likely Genuine",
                        len(results) - fraud_count
                    )

                    st.subheader("Detection Results")

                    st.dataframe(
                        results,
                        use_container_width=True
                    )

                    st.download_button(
                        "⬇️ Download Results",
                        results.to_csv(index=False).encode("utf-8"),
                        "fraud_detection_results.csv",
                        "text/csv"
                    )

        except Exception as error:

            st.error(f"CSV processing error: {error}")


# --------------------------------------------------
# ANALYTICS
# --------------------------------------------------

elif page == "Analytics":

    st.title("📊 Fraud Analytics")

    df = load_dataset()

    if df is None:

        st.warning(
            "creditcard.csv was not found in the dataset folder."
        )

    else:

        total = len(df)

        genuine = int((df["Class"] == 0).sum())

        fraud = int((df["Class"] == 1).sum())

        c1, c2, c3 = st.columns(3)

        c1.metric("Total Transactions", f"{total:,}")

        c2.metric("Genuine Transactions", f"{genuine:,}")

        c3.metric("Fraud Transactions", f"{fraud:,}")

        st.divider()

        class_df = pd.DataFrame({
            "Type": ["Genuine", "Fraud"],
            "Count": [genuine, fraud]
        })

        fig = px.bar(
            class_df,
            x="Type",
            y="Count",
            color="Type",
            text="Count",
            title="Genuine vs Fraud Transactions"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        amount_fig = px.histogram(
            df,
            x="Amount",
            color="Class",
            nbins=50,
            title="Transaction Amount Distribution"
        )

        st.plotly_chart(
            amount_fig,
            use_container_width=True
        )


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

elif page == "Model Performance":

    st.title("📈 Model Performance")

    st.subheader("Final Random Forest Test Results")

    c1, c2, c3 = st.columns(3)

    c1.metric("Precision", "91.36%")
    c2.metric("Recall", "77.89%")
    c3.metric("F1 Score", "84.09%")

    st.divider()

    st.subheader("Confusion Matrix")

    confusion_df = pd.DataFrame(
        [[56644, 7], [21, 74]],
        index=["Actual Genuine", "Actual Fraud"],
        columns=["Predicted Genuine", "Predicted Fraud"]
    )

    st.dataframe(confusion_df)

    # Model comparison
    if REPORT_PATH.exists():

        results_df = pd.read_csv(REPORT_PATH)

        st.subheader("Five Model Comparison")

        st.dataframe(
            results_df,
            use_container_width=True
        )

        metrics = [
            col for col in [
                "Precision",
                "Recall",
                "F1 Score",
                "ROC-AUC",
                "PR-AUC"
            ]
            if col in results_df.columns
        ]

        if metrics and "Model" in results_df.columns:

            selected_metric = st.selectbox(
                "Select Metric",
                metrics
            )

            fig = px.bar(
                results_df,
                x="Model",
                y=selected_metric,
                color="Model",
                title=f"Model Comparison: {selected_metric}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    else:

        st.warning(
            "Model comparison CSV was not found."
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.sidebar.divider()

st.sidebar.caption(
    "FraudShield AI | Major Project"
)
