import streamlit as st
import pandas as pd
import pickle

# ---------------------------------------------------------
# Load product-specific model and encoder
# ---------------------------------------------------------

@st.cache_resource
def load_model(product):
    name = product.lower().replace(" ", "_")

    with open(f"churn_model_{name}.pkl", "rb") as f:
        model = pickle.load(f)

    with open(f"churn_encoder_{name}.pkl", "rb") as f:
        encoder = pickle.load(f)

    return model, encoder


# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------

st.set_page_config(
    page_title="Optima Life Retention Dashboard",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

[data-testid="stMetric"] {
    background-color: #1f1f1f;
    border: 1px solid #333;
    border-radius: 12px;
    padding: 16px;
}

div.stButton > button {
    background-color: #2563eb;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}

div.stButton > button:hover {
    background-color: #1d4ed8;
    color: white;
}

.action-box {
    background-color: #1f1f1f;
    border: 1px solid #333;
    border-left: 5px solid #2563eb;
    border-radius: 10px;
    padding: 16px;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 Optima Life Retention Dashboard")
st.caption("Estimate churn risk and recommend the best retention action.")


# ---------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------

st.sidebar.header("Customer Information")

product = st.sidebar.selectbox(
    "Product",
    [
        "Healthy Meals",
        "Daily Fitness",
        "Wellness Tracker",
        "Mindful Living",
        "Premium Health"
    ]
)

age = st.sidebar.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=40
)

income = st.sidebar.selectbox(
    "Income Level",
    ["Low", "Medium", "High", "Very High"]
)

education = st.sidebar.selectbox(
    "Education",
    ["High School", "Graduate", "Post-Graduate", "Other"]
)

device = st.sidebar.selectbox(
    "Device Type",
    ["Mobile-only", "Desktop-only", "Multi-device"]
)

tech_score = st.sidebar.slider(
    "Tech Comfort Score",
    1, 5, 3
)

sessions = st.sidebar.number_input(
    "Total Sessions",
    min_value=0,
    value=50
)

session_length = st.sidebar.number_input(
    "Total Session Length",
    min_value=0.0,
    value=1000.0
)

active_days = st.sidebar.number_input(
    "Active Days",
    min_value=0,
    value=20
)

active_quarters = st.sidebar.number_input(
    "Active Quarters",
    min_value=1,
    max_value=4,
    value=4
)


# ---------------------------------------------------------
# Engineered behavioral features
# ---------------------------------------------------------

avg_sessions_per_quarter = sessions / active_quarters

avg_session_length_per_day = (
    session_length / active_days
    if active_days > 0 else 0
)

sessions_per_active_day = (
    sessions / active_days
    if active_days > 0 else 0
)


# ---------------------------------------------------------
# Product-level CLTR
# ---------------------------------------------------------

cltr_values = {
    "Healthy Meals": 368.63,
    "Premium Health": 303.36,
    "Daily Fitness": 294.16,
    "Mindful Living": 271.20,
    "Wellness Tracker": 134.22
}

cltr = cltr_values[product]


# ---------------------------------------------------------
# Prediction button
# ---------------------------------------------------------

predict = st.sidebar.button(
    "Assess Churn Risk",
    type="primary",
    use_container_width=True
)


# ---------------------------------------------------------
# Customer overview
# ---------------------------------------------------------

st.subheader("Customer Overview")

col1, col2, col3 = st.columns(3)

col1.markdown("**Product**")
col1.markdown(f"### {product}")

col2.metric(
    "Total Sessions",
    sessions
)

col3.metric(
    "CLTR",
    f"${cltr:,.0f}"
)


# ---------------------------------------------------------
# Run churn model
# ---------------------------------------------------------

if predict:

    model, encoder = load_model(product)

    categorical_cols = [
        "INCOME_LEVEL",
        "EDUCATION",
        "DEVICE_TYPE"
    ]

    categorical_data = pd.DataFrame([{
        "INCOME_LEVEL": income,
        "EDUCATION": education,
        "DEVICE_TYPE": device
    }])

    encoded = encoder.transform(
        categorical_data[categorical_cols]
    )

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(categorical_cols)
    )

    numeric_data = pd.DataFrame([{
        "TOTAL_NUM_SESSIONS": sessions,
        "GROSS_SESSION_LENGTH": session_length,
        "ACTIVE_DAYS": active_days,
        "ACTIVE_QUARTERS": active_quarters,
        "AVG_SESSIONS_PER_QUARTER": avg_sessions_per_quarter,
        "AVG_SESSION_LENGTH_PER_DAY": avg_session_length_per_day,
        "SESSIONS_PER_ACTIVE_DAY": sessions_per_active_day,
        "AGE": age,
        "TECH_COMFORT_SCORE": tech_score
    }])

    model_input = pd.concat(
        [numeric_data, encoded_df],
        axis=1
    )

    # Match the exact feature order used during model training
    if hasattr(model, "feature_names_in_"):
        model_input = model_input.reindex(
            columns=model.feature_names_in_,
            fill_value=0
        )

    # Model predicts renewal probability.
    # Churn probability = 1 - renewal probability.
    renewal_probability = model.predict_proba(
        model_input
    )[0, 1]

    churn_probability = 1 - renewal_probability


    # -----------------------------------------------------
    # Risk tier and recommendation
    # -----------------------------------------------------

    if churn_probability >= 0.60:
        risk_level = "🔴 Critical Risk"
        recommendation = (
            "Immediate personal outreach and a personalized "
            "retention offer."
        )

    elif churn_probability >= 0.30:
        risk_level = "🟠 High Risk"
        recommendation = (
            "Prioritize this customer for a targeted "
            "retention campaign."
        )

    elif churn_probability >= 0.10:
        risk_level = "🟡 Medium Risk"
        recommendation = (
            "Continue monitoring and use a low-cost "
            "re-engagement campaign."
        )

    else:
        risk_level = "🟢 Low Risk"
        recommendation = (
            "No immediate action is needed. "
            "Continue regular engagement."
        )


    # -----------------------------------------------------
    # Customer value at risk
    # -----------------------------------------------------

    weighted_risk = churn_probability * cltr


    # -----------------------------------------------------
    # Assessment results
    # -----------------------------------------------------

    st.divider()

    st.subheader("Retention Assessment")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Churn Probability",
        f"{churn_probability:.0%}"
    )

    col2.metric(
        "Risk Level",
        risk_level
    )

    col3.metric(
        "Customer Value at Risk",
        f"${weighted_risk:,.0f}"
    )

    st.progress(float(churn_probability))


    # -----------------------------------------------------
    # Recommended action
    # -----------------------------------------------------

    st.subheader("Recommended Action")

    st.markdown(
        f"""
        <div class="action-box">
            {recommendation}
        </div>
        """,
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # Observed risk factors
    # -----------------------------------------------------

    st.subheader("Observed Risk Factors")

    drivers = []

    if sessions < 25:
        drivers.append("Low session activity")

    if active_days < 10:
        drivers.append("Low number of active days")

    if tech_score <= 2:
        drivers.append("Low technology comfort")

    if device == "Mobile-only":
        drivers.append("Mobile-only customer")

    if income == "Low":
        drivers.append("Lower income segment")

    if not drivers:
        drivers.append("No major risk factors detected")

    for driver in drivers:
        st.write(f"• {driver}")


else:

    st.info(
        "Enter the customer information in the sidebar, "
        "then select Assess Churn Risk."
    )


st.caption(
    "Predictions use product-specific Random Forest churn models."
)
