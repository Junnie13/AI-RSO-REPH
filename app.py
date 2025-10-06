import streamlit as st
import pandas as pd
import plotly.express as px
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from pygwalker.api.streamlit import StreamlitRenderer
import numpy as np

# -----------------------------
# 1. Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Employee Retention Dashboard",
    page_icon="👩‍💼",
    layout="wide"
)

st.title("Employee Retention Analysis Dashboard")

# -----------------------------
# 2. Sidebar Upload
# -----------------------------
st.sidebar.header("📂 Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload Excel or CSV file", type=["xlsx", "csv"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
    st.sidebar.success(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
else:
    df = None

# -----------------------------
# 3. Tabs
# -----------------------------
eda_tab, modeling_tab, custom_tab = st.tabs([
    "📊 Exploratory Data Analysis",
    "🤖 Modeling & SHAP Analysis",
    "🧠 Custom Data Exploration"
])

# =============================================================================
# TAB 1 — EXPLORATORY DATA ANALYSIS
# =============================================================================
with eda_tab:
    if df is None:
        st.info("👆 Please upload a dataset in the sidebar to begin exploring.")
    else:
        st.header("📊 Exploratory Data Analysis")

        # --- Attrition Count ---
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Attrition Count")
            fig = px.histogram(df, x="Attrition", color="Attrition", color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Attrition by Gender")
            fig = px.histogram(df, x="Gender", color="Attrition", barmode="group",
                               color_discrete_sequence=px.colors.qualitative.Safe)
            st.plotly_chart(fig, use_container_width=True)

        # --- Department-wise Attrition Rate ---
        st.subheader("Department-wise Attrition Rate")
        dept_attrition = df.groupby("Department")["Attrition"].value_counts(normalize=True).unstack().fillna(0)
        dept_attrition = dept_attrition.reset_index().melt(id_vars="Department", var_name="Attrition", value_name="Rate")
        fig = px.bar(dept_attrition, x="Department", y="Rate", color="Attrition", barmode="stack",
                     color_discrete_sequence=px.colors.sequential.Viridis)
        st.plotly_chart(fig, use_container_width=True)

        # --- Monthly Income Distribution ---
        st.subheader("Distribution of Monthly Income")
        fig = px.histogram(df, x="MonthlyIncome", nbins=30, color_discrete_sequence=["teal"], marginal="box")
        st.plotly_chart(fig, use_container_width=True)

        # --- Correlation Heatmap ---
        st.subheader("Correlation Heatmap")
        numeric_df = df.select_dtypes(include=["int64", "float64"])
        corr = numeric_df.corr()
        fig = px.imshow(corr, text_auto=False, color_continuous_scale="RdBu_r", title="Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)

        # --- Job Satisfaction by Job Role ---
        st.subheader("Average Job Satisfaction by Role")
        if "JobRole" in df.columns and "JobSatisfaction" in df.columns:
            fig = px.bar(df, x="JobRole", y="JobSatisfaction", color="JobRole",
                         color_discrete_sequence=px.colors.qualitative.Prism)
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TAB 2 — MODELING & SHAP ANALYSIS
# =============================================================================
with modeling_tab:
    if df is None:
        st.info("👆 Please upload a dataset to perform modeling and SHAP analysis.")
    else:
        st.header("🤖 Employee Attrition Model and SHAP Analysis")

        target_col = st.selectbox(
            "Select Target Column",
            options=df.columns,
            index=list(df.columns).index("Attrition") if "Attrition" in df.columns else 0
        )

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Encode categorical variables
        X = X.apply(lambda x: LabelEncoder().fit_transform(x) if x.dtype == "object" else x)
        if y.dtype == "object":
            y = LabelEncoder().fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train Model
        model = GradientBoostingClassifier(random_state=42)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Classification Report")
            st.text(classification_report(y_test, y_pred))

        with col2:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
            fig = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues", title="Confusion Matrix")
            st.plotly_chart(fig, use_container_width=True)

        # --- ROC Curve ---
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr})
        fig = px.area(roc_df, x="FPR", y="TPR", title=f"ROC Curve (AUC = {auc:.4f})", labels={"FPR": "False Positive Rate", "TPR": "True Positive Rate"})
        fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
        st.plotly_chart(fig, use_container_width=True)

        # --- SHAP Analysis ---
        st.subheader("Feature Importance (SHAP Summary)")
        explainer = shap.Explainer(model, X_train_scaled)
        shap_values = explainer(X_test_scaled, check_additivity=False)
        shap.summary_plot(shap_values, X_test, show=False)
        st.pyplot(bbox_inches='tight')

# =============================================================================
# TAB 3 — CUSTOM DATA EXPLORATION
# =============================================================================
with custom_tab:
    if df is None:
        st.info("👆 Please upload a dataset to explore it interactively.")
    else:
        st.header("🧠 Custom Data Exploration")
        st.markdown("Use **PyGWalker** to explore your dataset freely — drag, drop, and visualize interactively.")
        pyg_app = StreamlitRenderer(df)
        pyg_app.explorer()
