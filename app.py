# --------------------------------------------------------
# EMPLOYEE ATTRITION DASHBOARD (XGBoost + Pickle Version)
# --------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from io import BytesIO
from pygwalker.api.streamlit import StreamlitRenderer

# --------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------
st.set_page_config(page_title="Employee Attrition Dashboard", layout="wide")
st.title("📊 Employee Attrition Prediction Dashboard")

# --------------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------------
page = st.sidebar.radio("Navigation", ["Dashboard", "Sandbox"])

# --------------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------------
@st.cache_resource
def load_model():
    with open("employee_attrition_model.pkl", "rb") as f:
        model = pickle.load(f)
    expected_cols = model.named_steps['preprocessor'].feature_names_in_
    return model, expected_cols

model, expected_cols = load_model()

# --------------------------------------------------------
# PAGE 1: DASHBOARD
# --------------------------------------------------------
if page == "Dashboard":
    tab1, tab2, tab3 = st.tabs(["Predict Employee Attrition", "Data Exploration", "Data Analysis"])

    # --------------------------------------------------------
    # TAB 1: PREDICT EMPLOYEE ATTRITION
    # --------------------------------------------------------
    with tab1:

        analysis_mode = st.radio("Choose Analysis Mode:", ["Single Analysis", "Batch Analysis"])

        # ---------------- SINGLE ANALYSIS ----------------
        if analysis_mode == "Single Analysis":
            st.markdown("Provide employee details below to predict attrition.")

            # Split input form into columns
            col1, col2, col3 = st.columns(3)

            with col1:
                Age = st.number_input("Age", 18, 60, 35)
                Gender = st.selectbox("Gender", ["Female", "Male"])
                OverTime = st.selectbox("OverTime", ["No", "Yes"])
                Department = st.selectbox("Department", ["Human Resources", "Research & Development", "Sales"])
                BusinessTravel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])

            with col2:
                EducationField = st.selectbox("Education Field", ["Human Resources", "Life Sciences", "Marketing", "Medical", "Technical Degree", "Other"])
                JobRole = st.selectbox("JobRole", [
                    "Healthcare Representative", "Human Resources", "Laboratory Technician", "Manager",
                    "Manufacturing Director", "Research Director", "Research Scientist", "Sales Executive", "Sales Representative"
                ])
                MaritalStatus = st.selectbox("Marital Status", ["Divorced", "Married", "Single"])
                MonthlyIncome = st.number_input("Monthly Income", 1000, 20000, 5000)
                Education = st.slider("Education (1–5)", 1, 5, 3)

            with col3:
                TotalWorkingYears = st.number_input("Total Working Years", 0, 40, 10)
                YearsAtCompany = st.number_input("Years at Company", 0, 40, 5)
                WorkLifeBalance = st.slider("Work-Life Balance (1–4)", 1, 4, 3)
                JobSatisfaction = st.slider("Job Satisfaction (1–4)", 1, 4, 3)
                PerformanceRating = st.slider("Performance Rating (1–4)", 1, 4, 3)

            # Construct DataFrame
            input_dict = {
                "Age": Age,
                "Gender": Gender,
                "OverTime": OverTime,
                "Department": Department,
                "BusinessTravel": BusinessTravel,
                "EducationField": EducationField,
                "JobRole": JobRole,
                "MaritalStatus": MaritalStatus,
                "MonthlyIncome": MonthlyIncome,
                "Education": Education,
                "TotalWorkingYears": TotalWorkingYears,
                "YearsAtCompany": YearsAtCompany,
                "WorkLifeBalance": WorkLifeBalance,
                "JobSatisfaction": JobSatisfaction,
                "PerformanceRating": PerformanceRating
            }

            input_df = pd.DataFrame([input_dict])

            # Align columns with training features
            for col in expected_cols:
                if col not in input_df.columns:
                    input_df[col] = np.nan
            input_df = input_df.reindex(columns=expected_cols)

            if st.button("Predict Attrition"):
                # Predict
                pred_prob = model.predict_proba(input_df)[0][1]
                pred_class = "Yes" if pred_prob > 0.5 else "No"
                st.success(f"Predicted Attrition: **{pred_class}** (Probability: {pred_prob:.2f})")

        # ---------------- BATCH ANALYSIS ----------------
        else:
            st.markdown("Upload a dataset (without the Attrition column). The system will predict and append results.")

            uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])
            if uploaded_file:
                df_input_original = pd.read_excel(uploaded_file)
                st.session_state["original_data"] = df_input_original.copy()

                progress = st.progress(0)
                st.info("Processing data...")
                progress.progress(25)

                # Handle missing columns
                df_input = df_input_original.copy()
                for col in expected_cols:
                    if col not in df_input.columns:
                        df_input[col] = np.nan
                df_input = df_input.reindex(columns=expected_cols)
                progress.progress(50)

                st.info("Running model predictions...")
                preds = model.predict(df_input)
                probs = model.predict_proba(df_input)[:, 1]
                progress.progress(75)

                df_output = df_input_original.copy()
                df_output["Attrition Prediction"] = np.where(preds == 1, "Yes", "No")
                df_output["Attrition Probability"] = probs.round(2)
                progress.progress(100)
                st.success("✅ Prediction Complete!")

                buffer = BytesIO()
                df_output.to_excel(buffer, index=False)
                buffer.seek(0)
                st.download_button(
                    label="📥 Download Predicted File",
                    data=buffer,
                    file_name="predicted_attrition.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

                st.session_state["predicted_data"] = df_output

    # --------------------------------------------------------
    # TAB 2: DATA EXPLORATION (DARK THEME COLORS)
    # --------------------------------------------------------
    with tab2:

        if "predicted_data" in st.session_state:
            import plotly.express as px
            import plotly.graph_objects as go

            df_pred = st.session_state["predicted_data"].copy()

            # --------------------------------------
            # 🧮 SAFE NUMERIC CONVERSIONS
            # --------------------------------------
            df_pred["OverTime_num"] = df_pred["OverTime"].map({"Yes": 1, "No": 0})
            df_pred["Attrition_num"] = df_pred["Attrition Prediction"].map({"Yes": 1, "No": 0})
            get_mean = lambda col: df_pred[col].mean() if col in df_pred.columns else 0

            # --------------------------------------
            # 🧮 CLEAN KEY METRICS OVERVIEW
            # --------------------------------------
            st.markdown("### 🧮 Key Metrics Overview")

            total_employees = len(df_pred)
            churned = df_pred["Attrition Prediction"].value_counts().get("Yes", 0)
            retained = df_pred["Attrition Prediction"].value_counts().get("No", 0)
            churn_rate = (churned / total_employees) * 100 if total_employees else 0
            avg_income = get_mean("MonthlyIncome")
            avg_age = get_mean("Age")
            overtime_rate = get_mean("OverTime_num")

            colA, colB, colC = st.columns(3)
            with colA:
                st.metric("👥 Total Employees", f"{total_employees:,}")
            with colB:
                st.metric("💔 Predicted to Leave", f"{churned:,}")
            with colC:
                st.metric("💼 Retained", f"{retained:,}")

            colD, colE, colF = st.columns(3)
            with colD:
                st.metric("💰 Avg. Monthly Income", f"{avg_income:,.0f}")
            with colE:
                st.metric("🎂 Avg. Age", f"{avg_age:.1f}")
            with colF:
                st.metric("⏰ Overtime Rate", f"{overtime_rate * 100:.1f}%")

            st.markdown("---")

            # --------------------------------------
            # 🌙 DARK THEME VISUALS
            # --------------------------------------
            st.markdown("### 🌌 Visual Trends")

            dark_palette = px.colors.qualitative.Set2
            dark_palette_alt = px.colors.qualitative.Safe

            col1, col2 = st.columns(2)

            # 1️⃣ Attrition Breakdown
            with col1:
                fig1 = px.pie(
                    df_pred,
                    names="Attrition Prediction",
                    title="Attrition Prediction Breakdown",
                    color_discrete_sequence=dark_palette,
                    hole=0.45
                )
                fig1.update_traces(textinfo="percent+label")
                fig1.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white")
                )
                st.plotly_chart(fig1, use_container_width=True)

            # 2️⃣ Probability Distribution
            with col2:
                fig2 = px.histogram(
                    df_pred,
                    x="Attrition Probability",
                    nbins=25,
                    color="Attrition Prediction",
                    color_discrete_sequence=dark_palette_alt,
                    title="Attrition Probability Distribution"
                )
                fig2.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"),
                    yaxis_title="Count",
                    xaxis_title="Probability"
                )
                st.plotly_chart(fig2, use_container_width=True)

            col3, col4 = st.columns(2)

            # 3️⃣ Department vs Attrition Rate
            with col3:
                if {"Department", "Attrition Prediction"}.issubset(df_pred.columns):
                    df_dept = (
                        df_pred.groupby("Department")["Attrition Prediction"]
                        .value_counts(normalize=True)
                        .rename("Rate")
                        .reset_index()
                    )
                    fig3 = px.bar(
                        df_dept,
                        x="Department",
                        y="Rate",
                        color="Attrition Prediction",
                        barmode="group",
                        title="Attrition Rate by Department",
                        color_discrete_sequence=dark_palette
                    )
                    fig3.update_yaxes(tickformat=".0%")
                    fig3.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="white")
                    )
                    st.plotly_chart(fig3, use_container_width=True)

            # 4️⃣ Income vs Probability
            with col4:
                if {"MonthlyIncome", "Attrition Probability"}.issubset(df_pred.columns):
                    fig4 = px.scatter(
                        df_pred,
                        x="MonthlyIncome",
                        y="Attrition Probability",
                        color="Attrition Prediction",
                        title="Monthly Income vs Attrition Probability",
                        color_discrete_sequence=dark_palette_alt,
                        opacity=0.8
                    )
                    fig4.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="white")
                    )
                    st.plotly_chart(fig4, use_container_width=True)

            col5, col6 = st.columns(2)

            # 5️⃣ Job Satisfaction
            with col5:
                if {"JobSatisfaction", "Attrition Probability"}.issubset(df_pred.columns):
                    fig5 = px.box(
                        df_pred,
                        x="JobSatisfaction",
                        y="Attrition Probability",
                        color="Attrition Prediction",
                        title="Job Satisfaction vs Attrition Probability",
                        color_discrete_sequence=dark_palette
                    )
                    fig5.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="white")
                    )
                    st.plotly_chart(fig5, use_container_width=True)

            # 6️⃣ Work-Life Balance
            with col6:
                if {"WorkLifeBalance", "Attrition Prediction"}.issubset(df_pred.columns):
                    df_wlb = (
                        df_pred.groupby("WorkLifeBalance")["Attrition Prediction"]
                        .value_counts(normalize=True)
                        .rename("Rate")
                        .reset_index()
                    )
                    fig6 = px.bar(
                        df_wlb,
                        x="WorkLifeBalance",
                        y="Rate",
                        color="Attrition Prediction",
                        barmode="group",
                        title="Work-Life Balance vs Attrition Rate",
                        color_discrete_sequence=dark_palette_alt
                    )
                    fig6.update_yaxes(tickformat=".0%")
                    fig6.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="white")
                    )
                    st.plotly_chart(fig6, use_container_width=True)

            # 7️⃣ Role Tenure vs Probability
            st.markdown("### 🕒 Role Tenure Trends")
            df_pred.columns = df_pred.columns.str.strip()

            if {"YearsInCurrentRole", "Attrition Probability"}.issubset(df_pred.columns):
                if not df_pred.empty and df_pred["YearsInCurrentRole"].notnull().any():
                    fig7 = px.scatter(
                        df_pred,
                        x="YearsInCurrentRole",
                        y="Attrition Probability",
                        color="Attrition Prediction",
                        title="Years in Current Role vs Attrition Probability",
                        color_discrete_sequence=px.colors.qualitative.Dark24,  # 💫 visible on dark bg
                        opacity=0.85,
                        hover_data=["MonthlyIncome", "JobRole", "Department"]
                    )
                    fig7.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="white"),
                        title_font=dict(size=16),
                        xaxis_title="Years in Current Role",
                        yaxis_title="Attrition Probability",
                        legend_title_text="Attrition Prediction"
                    )
                    st.plotly_chart(fig7, use_container_width=True)
                else:
                    st.info("⚠️ No valid data available for 'YearsInCurrentRole' or 'Attrition Probability'.")
            else:
                st.warning("Columns 'YearsInCurrentRole' or 'Attrition Probability' not found in dataset.")

    # --------------------------------------------------------
    # TAB 3: DATA ANALYSIS (SHAP)
    # --------------------------------------------------------
    with tab3:


        if "predicted_data" in st.session_state:
            df_pred = st.session_state["predicted_data"].copy()

            # Drop prediction-related columns
            X = df_pred.drop(columns=[col for col in df_pred.columns if "Attrition" in col])

            # Align columns to model expectations
            for col in expected_cols:
                if col not in X.columns:
                    X[col] = np.nan
            X = X.reindex(columns=expected_cols)



            # Transform data
            preprocessor = model.named_steps["preprocessor"]
            X_transformed = preprocessor.transform(X)

            # ✅ Get the actual feature names after preprocessing
            try:
                feature_names = preprocessor.get_feature_names_out()
            except:
                feature_names = [f"Feature {i}" for i in range(X_transformed.shape[1])]

            # Compute SHAP
            explainer = shap.Explainer(model.named_steps["classifier"])
            shap_values = explainer(X_transformed)

            # Create two columns for side-by-side visuals
            col1, col2 = st.columns(2, gap="medium")

            # 🎯 SHAP Summary Plot
            with col1:
                st.markdown("### 🎯 SHAP Summary Plot")
                fig, ax = plt.subplots(figsize=(6, 8))
                shap.summary_plot(
                    shap_values.values,
                    features=X_transformed,
                    feature_names=feature_names,
                    plot_type="dot",
                    color_bar=True,
                    cmap=plt.cm.coolwarm,
                    show=False
                )
                st.pyplot(fig, bbox_inches="tight")

            # 📊 Top 10 Features
            with col2:
                st.markdown("### 📊 Top Features Influencing Attrition")
                shap_sum = np.abs(shap_values.values).mean(axis=0)
                shap_importance = (
                    pd.DataFrame({
                        "Feature": feature_names,
                        "Mean |SHAP|": shap_sum
                    })
                    .sort_values("Mean |SHAP|", ascending=False)
                    .head(10)
                )

                fig2 = px.bar(
                    shap_importance,
                    x="Mean |SHAP|",
                    y="Feature",
                    orientation="h",
                    color="Mean |SHAP|",
                    color_continuous_scale="tealrose",
                    title="Top 10 Most Influential Features",
                )
                fig2.update_layout(
                    yaxis=dict(categoryorder="total ascending"),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#E0E0E0"),
                    height=500,
                    margin=dict(l=0, r=0, t=40, b=40)
                )
                st.plotly_chart(fig2, use_container_width=True)

        else:
            st.warning("Please process a batch file first to analyze SHAP values.")



# --------------------------------------------------------
# PAGE 2: SANDBOX
# --------------------------------------------------------
else:
    st.title("🧪 Sandbox Environment")
    st.write("Explore the **predicted dataset** here (with predictions and probabilities).")

    # Check if predicted dataset is stored in session_state
    if "predicted_data" in st.session_state:
        df_sandbox = st.session_state["predicted_data"]

        # PyGWalker Explorer
        pyg_app = StreamlitRenderer(df_sandbox)
        pyg_app.explorer()

    else:
        st.info("⚠️ Please perform a prediction first in the 'Predict' tab to enable Sandbox exploration.")

