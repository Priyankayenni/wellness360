import os
import shutil
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import io
import zipfile

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------
st.set_page_config(
    page_title="Wellness360 - AI Wellness Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# LOGIN CREDENTIALS
# ---------------------------------------------------
VALID_USERS = {
    "admin": {"password": "admin123", "role": "Admin"},
    "hr": {"password": "hr123", "role": "HR"}
}

# ---------------------------------------------------
# REQUIRED COLUMNS
# ---------------------------------------------------
REQUIRED_COLS = [
    "Employee_ID",
    "Age",
    "Gender",
    "Stress_Level",
    "Sleep_Quality",
    "Mental_Health_Condition",
    "Physical_Activity",
    "Industry",
    "Job_Role"
]

# ---------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------
def authenticate(username, password):
    username = username.strip()
    password = password.strip()
    if username in VALID_USERS and VALID_USERS[username]["password"] == password:
        return True, VALID_USERS[username]["role"]
    return False, None

def check_columns(df):
    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    return missing

def create_wellness_scores(df):
    stress_map = {"Low": 1, "Medium": 2, "High": 3}
    sleep_map = {"Poor": 1, "Average": 2, "Good": 3}
    mental_map = {"None": 1, "Burnout": 2, "Anxiety": 3, "Depression": 4}

    df["Stress_Score"] = df["Stress_Level"].map(stress_map)
    df["Sleep_Score"] = df["Sleep_Quality"].map(sleep_map)
    df["Mental_Score"] = df["Mental_Health_Condition"].map(mental_map)

    df["Physical_Score"] = df["Physical_Activity"].apply(
        lambda x: 3 if "daily" in str(x).lower()
        else (2 if "weekly" in str(x).lower() else 1)
    )

    for c in ["Stress_Score", "Sleep_Score", "Mental_Score", "Physical_Score"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())

    df["Physical_Health_Score"] = (df["Sleep_Score"] + df["Physical_Score"]) / 2
    df["Mental_Health_Score"] = (df["Stress_Score"] + df["Mental_Score"]) / 2

    def classify(score):
        if score >= 3:
            return "High Risk"
        elif score >= 2:
            return "Moderate Risk"
        else:
            return "Low Risk"

    df["Risk_Level"] = df["Mental_Health_Score"].apply(classify)

    def recommend(row):
        rec = []
        if row["Stress_Score"] >= 3:
            rec.append("Counseling / reduce workload")
        if row["Sleep_Score"] <= 1:
            rec.append("Sleep hygiene program")
        if row["Physical_Score"] <= 1:
            rec.append("Encourage daily exercise")
        if row["Mental_Score"] >= 3:
            rec.append("Therapy / support groups")
        return ", ".join(rec) if rec else "Maintain regular wellness check-ins"

    df["Recommendations"] = df.apply(recommend, axis=1)
    return df

def train_model(df):
    df_ml = df.copy()
    for col in df_ml.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df_ml[col] = le.fit_transform(df_ml[col].astype(str))

    X = df_ml.drop(columns=["Stress_Level"])
    y = df_ml["Stress_Level"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=120, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = round(accuracy_score(y_test, y_pred), 3)
    return model, acc, y_test, y_pred

def create_plots(df, y_test, y_pred):
    plots = {}

    # 1. Confusion Matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix - Stress Prediction")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plots["confusion_matrix"] = fig

    # 2. Overall Risk Distribution
    fig, ax = plt.subplots(figsize=(6, 3.5))
    sns.countplot(data=df, x="Risk_Level", order=["Low Risk", "Moderate Risk", "High Risk"], ax=ax)
    ax.set_title("Overall Risk Distribution")
    ax.set_xlabel("Risk Level")
    ax.set_ylabel("Employees")
    plots["risk_distribution"] = fig

    # 3. Industry-wise Risk
    if "Industry" in df.columns:
        top_ind = df["Industry"].value_counts().head(8).index
        df_ind = df[df["Industry"].isin(top_ind)]
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.countplot(data=df_ind, x="Industry", hue="Risk_Level", hue_order=["Low Risk", "Moderate Risk", "High Risk"], ax=ax)
        ax.set_title("Industry-wise Risk (Top 8)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_xlabel("Industry")
        ax.set_ylabel("Employees")
        ax.legend(title="Risk Level")
        plots["industry_risk"] = fig

    # 4. Job Role-wise Risk
    if "Job_Role" in df.columns:
        top_jobs = df["Job_Role"].value_counts().head(10).index
        df_job = df[df["Job_Role"].isin(top_jobs)]
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.countplot(data=df_job, x="Job_Role", hue="Risk_Level", hue_order=["Low Risk", "Moderate Risk", "High Risk"], ax=ax)
        ax.set_title("Job Role-wise Risk (Top 10)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_xlabel("Job Role")
        ax.set_ylabel("Employees")
        ax.legend(title="Risk Level")
        plots["jobrole_risk"] = fig

    # 5. Industry Avg Scores
    if "Industry" in df.columns:
        top_ind = df["Industry"].value_counts().head(8).index
        industry_scores = df.groupby("Industry")[["Mental_Health_Score", "Physical_Health_Score"]].mean()
        available_ind = [i for i in top_ind if i in industry_scores.index]
        industry_scores = industry_scores.loc[available_ind].sort_values("Mental_Health_Score")
        fig, ax = plt.subplots(figsize=(10, 4))
        industry_scores.plot(kind="bar", ax=ax)
        ax.set_title("Industry-wise Average Health Scores")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_xlabel("Industry")
        ax.set_ylabel("Average Score")
        ax.legend(["Mental Health Score", "Physical Health Score"])
        plots["industry_scores"] = fig

    # 6. Job Role Avg Scores
    if "Job_Role" in df.columns:
        top_jobs = df["Job_Role"].value_counts().head(10).index
        job_scores = df.groupby("Job_Role")[["Mental_Health_Score", "Physical_Health_Score"]].mean()
        available_jobs = [j for j in top_jobs if j in job_scores.index]
        job_scores = job_scores.loc[available_jobs].sort_values("Mental_Health_Score")
        fig, ax = plt.subplots(figsize=(10, 4))
        job_scores.plot(kind="bar", ax=ax)
        ax.set_title("Job Role-wise Average Health Scores")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_xlabel("Job Role")
        ax.set_ylabel("Average Score")
        ax.legend(["Mental Health Score", "Physical Health Score"])
        plots["jobrole_scores"] = fig

    return plots

def create_zip_file(df, plots):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        final_cols = [c for c in ["Employee_ID", "Age", "Gender", "Industry", "Job_Role",
                                  "Physical_Health_Score", "Mental_Health_Score",
                                  "Risk_Level", "Recommendations"] if c in df.columns]
        final_table = df[final_cols].copy()
        csv_buffer = io.StringIO()
        final_table.to_csv(csv_buffer, index=False)
        zipf.writestr("final_results_table.csv", csv_buffer.getvalue())

        for name, fig in plots.items():
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format="png", dpi=300, bbox_inches="tight")
            img_buffer.seek(0)
            zipf.writestr(f"{name}.png", img_buffer.read())

    buffer.seek(0)
    return buffer

# ---------------------------------------------------
# MAIN APP
# ---------------------------------------------------
def main():
    st.title("🏥 Wellness360 - AI Employee Wellness Prediction System")
    st.markdown("### Secure Login Required")

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.role = None

    if not st.session_state.authenticated:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login")
                if submitted:
                    ok, role = authenticate(username, password)
                    if ok:
                        st.session_state.authenticated = True
                        st.session_state.role = role
                        st.success(f"✅ Login successful as {role}")
                        st.rerun()
                    else:
                        st.error("❌ Invalid login. Use admin/admin123 or hr/hr123")
        st.markdown("---")
        st.info("📌 **Demo Credentials:**\n- **Admin:** admin / admin123\n- **HR:** hr / hr123")
        return

    st.sidebar.success(f"Logged in as: {st.session_state.role}")
    st.sidebar.markdown("---")

    tab1, tab2 = st.tabs(["🤖 ML Analytics", "✨ Interactive Glow UI"])

    with tab1:
        st.header("📊 Employee Wellness Dashboard")
        # Upload CSV
        uploaded_file = st.file_uploader("Upload Employee Wellness CSV", type=["csv"])

    # Display Existing Results
    st.subheader("📈 Research Results (Pre-loaded)")
    if os.path.exists("results"):
        result_files = os.listdir("results")
        if result_files:
            st.write(f"**Found {len(result_files)} result files**")
            for file in result_files:
                if file.endswith(".png"):
                    st.image(f"results/{file}", caption=file, use_column_width=True)
                elif file.endswith(".csv"):
                    st.dataframe(pd.read_csv(f"results/{file}").head(10))
    else:
        st.info("📁 No pre-loaded results found. Upload CSV to generate.")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            missing = check_columns(df)

            if missing:
                st.error(f"❌ Missing required columns: {missing}")
                st.info(f"CSV must include: {REQUIRED_COLS}")
                return

            st.success("✅ Data loaded successfully!")
            st.write(f"**Total Records:** {df.shape[0]}")
            st.write(f"**Columns:** {list(df.columns)}")

            with st.spinner("Processing data..."):
                df = create_wellness_scores(df)
                model, acc, y_test, y_pred = train_model(df)
                plots = create_plots(df, y_test, y_pred)

            st.success("✅ Processing complete!")

            # Model Performance
            st.subheader("🤖 Model Performance")
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{acc}")
            col2.metric("Total Employees", f"{df.shape[0]}")
            col3.metric("High Risk", f"{len(df[df['Risk_Level'] == 'High Risk'])}")

            st.write(classification_report(y_test, y_pred))

            # Plots
            st.subheader("📈 Visualizations")
            for name, fig in plots.items():
                st.pyplot(fig)

            # Download
            st.subheader("📥 Download Results")
            zip_buffer = create_zip_file(df, plots)
            st.download_button(
                label="📦 Download ZIP Report",
                data=zip_buffer,
                file_name="wellness360_report.zip",
                mime="application/zip"
            )

            # Final Table
            st.subheader("📋 Final Results Table")
            final_cols = [c for c in ["Employee_ID", "Age", "Gender", "Industry", "Job_Role",
                                      "Physical_Health_Score", "Mental_Health_Score",
                                      "Risk_Level", "Recommendations"] if c in df.columns]
            st.dataframe(df[final_cols].head(10))

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.exception(e)

    else:
        st.info("👆 Please upload a CSV file to begin analysis")

    st.sidebar.markdown("---")
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.role = None
        st.rerun()

if __name__ == "__main__":
    main()
