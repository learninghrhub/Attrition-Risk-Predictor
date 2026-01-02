import streamlit as st
import pandas as pd
import os
import plotly.express as px
from io import BytesIO


# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Retention Intelligence Simulator",
    layout="wide"
)
# -------------------------------------------------
# Global UI Styling — Tabs & Sections
# -------------------------------------------------
st.markdown("""
<style>

/* ---- Main background ---- */
.stApp {
    background-color: #020617;
    color: #e5e7eb;
}

/* ---- Tabs container ---- */
div[data-testid="stTabs"] {
    background-color: #020617;
}

/* ---- Individual tabs ---- */
button[data-baseweb="tab"] {
    background-color: #020617;
    color: #94a3b8;
    border-radius: 12px 12px 0 0;
    font-weight: 600;
    padding: 12px 18px;
    border: 1px solid #1e293b;
    margin-right: 6px;
}

/* ---- Active tab ---- */
button[data-baseweb="tab"][aria-selected="true"] {
    background-color: #0f172a;
    color: #e5e7eb;
    border-bottom: 2px solid #38bdf8;
}

/* ---- Tab content panel ---- */
div[data-testid="stTabsContent"] {
    background-color: #0f172a;
    border-radius: 0 16px 16px 16px;
    padding: 24px;
    border: 1px solid #1e293b;
    box-shadow: 0 10px 30px rgba(0,0,0,0.35);
}

/* ---- Section cards ---- */
.section-card {
    background-color: #020617;
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 22px;
    margin-bottom: 24px;
}

/* ---- Section titles ---- */
.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    margin-bottom: 8px;
    color: #e5e7eb;
}

/* ---- Section subtitle ---- */
.section-subtitle {
    font-size: 0.85rem;
    color: #94a3b8;
    margin-bottom: 18px;
}

/* ---- Divider cleanup ---- */
hr {
    border: none;
    height: 1px;
    background-color: #1e293b;
    margin: 28px 0;
}

/* ---- Dataframe polish ---- */
div[data-testid="stDataFrame"] {
    border-radius: 14px;
    border: 1px solid #1e293b;
    overflow: hidden;
}

</style>
""", unsafe_allow_html=True)
# -------------------------------------------------
# KPI Cards Styling (Streamlit Compatible)
# -------------------------------------------------
st.markdown("""
<style>

/* ---- KPI card wrapper ---- */
div[data-testid="stMetric"] {
    background: linear-gradient(
        135deg,
        #0f172a,
        #020617
    );
    border: 1px solid #1e293b;
    border-radius: 18px;
    padding: 20px 18px;
    box-shadow: 0 12px 28px rgba(0,0,0,0.45);
    transition: all 0.25s ease-in-out;
}

/* ---- KPI hover ---- */
div[data-testid="stMetric"]:hover {
    transform: translateY(-5px);
    box-shadow: 0 20px 45px rgba(0,0,0,0.65);
}

/* ---- KPI label ---- */
div[data-testid="stMetric"] label {
    color: #94a3b8;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}

/* ---- KPI value ---- */
div[data-testid="stMetric"] div {
    color: #e5e7eb;
    font-size: 1.65rem;
    font-weight: 800;
    margin-top: 6px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Chart Layout Polish
# -------------------------------------------------
st.markdown("""
<style>

/* ---- Chart container spacing ---- */
div[data-testid="stPlotlyChart"] {
    background-color: #020617;
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 16px;
    margin-bottom: 28px;
}

/* ---- Remove extra iframe padding ---- */
iframe {
    border-radius: 12px;
}

/* ---- Improve chart titles spacing ---- */
.js-plotly-plot .plotly .gtitle {
    font-size: 18px !important;
    font-weight: 700 !important;
}

/* ---- Legend polish ---- */
.js-plotly-plot .legend text {
    font-size: 12px !important;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Login Screen Styling
# -------------------------------------------------
st.markdown("""
<style>

/* ---- Login card ---- */
.login-card {
    background: linear-gradient(
        145deg,
        rgba(15, 23, 42, 0.95),
        rgba(2, 6, 23, 0.95)
    );
    border: 1px solid #1e293b;
    border-radius: 22px;
    padding: 36px 34px;
    box-shadow: 0 25px 60px rgba(0,0,0,0.65);
}

/* ---- Login title ---- */
.login-title {
    font-size: 1.6rem;
    font-weight: 800;
    text-align: center;
    margin-bottom: 6px;
    color: #e5e7eb;
}

/* ---- Login subtitle ---- */
.login-subtitle {
    font-size: 0.85rem;
    text-align: center;
    color: #94a3b8;
    margin-bottom: 26px;
}

/* ---- Button polish ---- */
button[kind="primary"] {
    background: linear-gradient(135deg, #38bdf8, #2563eb);
    border-radius: 14px;
    font-weight: 700;
    border: none;
    height: 44px;
}

/* ---- Inputs ---- */
input, select {
    border-radius: 12px !important;
}

</style>
""", unsafe_allow_html=True)
# -------------------------------------------------
# Executive Header Styling
# -------------------------------------------------
st.markdown("""
<style>

/* ---- Header container ---- */
.exec-header {
    background: linear-gradient(
        135deg,
        rgba(15, 23, 42, 0.95),
        rgba(2, 6, 23, 0.95)
    );
    border: 1px solid #1e293b;
    border-radius: 18px;
    padding: 20px 26px;
    margin-bottom: 26px;
}

/* ---- Product title ---- */
.exec-title {
    font-size: 1.6rem;
    font-weight: 800;
    color: #e5e7eb;
    margin-bottom: 4px;
}

/* ---- Product subtitle ---- */
.exec-subtitle {
    font-size: 0.85rem;
    color: #94a3b8;
}

/* ---- User context ---- */
.exec-user {
    font-size: 0.8rem;
    color: #38bdf8;
    margin-top: 6px;
}

</style>
""", unsafe_allow_html=True)
# -------------------------------------------------
# Risk-Colored KPI Cards
# -------------------------------------------------
st.markdown("""
<style>

/* ---- Default KPI (neutral) ---- */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0f172a, #020617);
}

/* ---- High Risk KPIs ---- */
div[data-testid="stMetric"]:has(label:contains("High")) {
    background: linear-gradient(135deg, #7f1d1d, #450a0a);
}

/* ---- Medium Risk KPIs ---- */
div[data-testid="stMetric"]:has(label:contains("Medium")) {
    background: linear-gradient(135deg, #78350f, #451a03);
}

/* ---- Low / Positive KPIs ---- */
div[data-testid="stMetric"]:has(label:contains("Low")) {
    background: linear-gradient(135deg, #14532d, #052e16);
}

/* ---- Cost / Negative KPIs ---- */
div[data-testid="stMetric"]:has(label:contains("Cost")) {
    background: linear-gradient(135deg, #4c0519, #1f0208);
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# CENTER SCREEN LOGIN (PHASE 3 FINAL)
# -------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:

    st.markdown("<br><br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("""
        <div class="login-card">
            <div class="login-title">Access Portal</div>
            <div class="login-subtitle">
                Secure role-based access to Retention Intelligence
            </div>
        """, unsafe_allow_html=True)

        user_role = st.selectbox(
            "Login As",
            ["Admin_1=Vivek", "Admin_2=Tabish"]
        )

        user_name = st.text_input(
            "Your Name",
            placeholder="e.g. Amit Sharma"
        )

        user_department = st.selectbox(
            "Department Scope",
            ["All", "HR", "Sales", "IT", "Finance", "Operations", "Marketing", "Admin"]
        )

        if st.button("Enter Platform"):
            if user_name.strip() == "":
                st.warning("Please enter your name.")
            else:
                st.session_state.logged_in = True
                st.session_state.user_role = user_role
                st.session_state.user_name = user_name
                st.session_state.user_department = user_department
                st.rerun()

        # ✅ CLOSE LOGIN CARD
        st.markdown("</div>", unsafe_allow_html=True)

    # ⛔ Stop app until login completes
    st.stop()
    
st.markdown("""
<div class="exec-header">
""", unsafe_allow_html=True)

# FIX: Unpack the columns into 3 variables (c1, c2, c3)
c1, c2, c3 = st.columns([1, 2, 1]) 

# Use the first column (Left) for the Title
with c1:
    st.title("HR Attrition Intelligence")

# Use the second column (Middle/Wide) for the HTML Block
with c2:
    st.markdown("""
    <div class="exec-title">Retention Intelligence</div>
    <div class="exec-subtitle">
        Enterprise Workforce Risk Intelligence Platform
    </div>
    <div class="exec-user">
        Logged in as: {user} · Role: {role} · Scope: {dept}
    </div>
    """.format(
        user=st.session_state.user_name,
        role=st.session_state.user_role,
        dept=st.session_state.user_department
    ), unsafe_allow_html=True)

# The third column (c3) is left empty for spacing, or you can add a logo here
st.markdown("---")

# -------------------------------------------------
# Persistent Storage
# -------------------------------------------------
DATA_FILE = "employee_data.csv"

if os.path.exists(DATA_FILE):
    employee_df = pd.read_csv(DATA_FILE)
else:
    employee_df = pd.DataFrame(columns=[
        "EmployeeID", "Name", "Department", "Role", "Tenure",
        "JobSatisfaction", "WorkLifeBalance", "ManagerSupport",
        "CareerGrowth", "StressLevel", "AttritionRisk", "RiskBand"
    ])
ACTIONS_FILE = "attrition_actions.csv"

if os.path.exists(ACTIONS_FILE):
    actions_df = pd.read_csv(ACTIONS_FILE)
else:
    actions_df = pd.DataFrame(columns=[
        "EmployeeID",
        "EmployeeName",
        "Department",
        "Manager",
        "RiskScore",
        "RiskBand",
        "SelectedAction",
        "ActionStatus",
        "ManagerComment"
    ])
# Ensure outcome columns exist
if "OutcomeStatus" not in actions_df.columns:
    actions_df["OutcomeStatus"] = "Pending"

if "OutcomeDate" not in actions_df.columns:
    actions_df["OutcomeDate"] = ""

# -------------------------------------------------
# Risk Logic (UPDATED – 0 to 100% SCALE)
# -------------------------------------------------
def calculate_attrition_risk(js, wl, ms, cg, stress):
    """
    js, wl, ms, cg: 1 (Very Low) → 5 (Very High)
    stress: 1 (Very Low) → 5 (Very High)
    """

    raw_score = (
        (6 - js) * 0.25 +     # Job dissatisfaction
        (6 - wl) * 0.20 +     # Work-life imbalance
        (6 - ms) * 0.20 +     # Poor manager support
        (6 - cg) * 0.15 +     # Low career growth
        stress * 0.30         # High stress impact
    )

    # Max possible raw_score = 6
    attrition_risk_pct = (raw_score / 6) * 100

    return round(min(attrition_risk_pct, 100), 2)

def risk_band(score):
    if score >= 70:
        return "High"
    elif score >= 40:
        return "Medium"
    else:
        return "Low"
def risk_color(val):
    if val == "High":
        return "background-color: #7f1d1d; color: white;"
    elif val == "Medium":
        return "background-color: #78350f; color: white;"
    else:
        return "background-color: #14532d; color: white;"

def risk_arrow(change):
    if change > 0:
        return "↓ Risk Reduced"
    elif change < 0:
        return "↑ Risk Increased"
    else:
        return "→ No Change"

        
def style_risk_rows(row):
    if "RiskBand" not in row:
        return [""] * len(row)

    if row["RiskBand"] == "High":
        return ["background-color: #7f1d1d; color: white"] * len(row)
    elif row["RiskBand"] == "Medium":
        return ["background-color: #78350f; color: white"] * len(row)
    elif row["RiskBand"] == "Low":
        return ["background-color: #14532d; color: white"] * len(row)
    else:
        return [""] * len(row)

# -------------------------------------------------
# Tabs
# -------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Employee Entry & Risk Prediction",
    "Executive Overview",
    "Prescriptive Actions",
    "Reports & Downloads",
    "Action Effectiveness",
    "Outcome Tracking"
])

# =================================================
# TAB 1 — DATA ENTRY + CSV UPLOAD
# =================================================
with tab1:

    st.markdown("### 1. Upload Employee Master (Bulk Scoring)")

    uploaded_file = st.file_uploader(
        "Upload Employee CSV",
        type=["csv"]
    )

    if uploaded_file:
        upload_df = pd.read_csv(uploaded_file)

        required_cols = [
            "EmployeeID", "Name", "Department", "Role", "Tenure",
            "JobSatisfaction", "WorkLifeBalance",
            "ManagerSupport", "CareerGrowth", "StressLevel"
        ]

        if all(col in upload_df.columns for col in required_cols):
            upload_df["AttritionRisk"] = upload_df.apply(
                lambda r: calculate_attrition_risk(
                    r["JobSatisfaction"],
                    r["WorkLifeBalance"],
                    r["ManagerSupport"],
                    r["CareerGrowth"],
                    r["StressLevel"]
                ),
                axis=1
            )

            upload_df["RiskBand"] = upload_df["AttritionRisk"].apply(risk_band)

            employee_df = pd.concat([employee_df, upload_df], ignore_index=True)
            employee_df.to_csv(DATA_FILE, index=False)

            st.success("File uploaded, scored, and saved successfully.")
            st.dataframe(upload_df, use_container_width=True)

        else:
            st.error("Uploaded file does not match required format.")

    st.markdown("---")
    st.markdown("### 2. Manual Employee Entry (Single Record)")

    col1, col2, col3 = st.columns(3)

    with col1:
        emp_id = st.text_input("Employee ID")
        name = st.text_input("Employee Name")

    with col2:
        department = st.selectbox("Department", ["HR","Sales","IT","Finance","Operations","Marketing"])

    with col3:
        role = st.selectbox("Role Level", ["Executive","Manager","Senior Staff","Staff","Entry Level"])

    tenure = st.select_slider("Tenure (Years)", [0,1,2,3,4,5,"5+"])

    st.caption("Scale: 1 = Very Low | 5 = Very High")

    js = st.select_slider("Job Satisfaction", [1,2,3,4,5])
    wl = st.select_slider("Work-Life Balance", [1,2,3,4,5])
    ms = st.select_slider("Manager Support", [1,2,3,4,5])
    cg = st.select_slider("Career Growth", [1,2,3,4,5])
    stress = st.select_slider("Stress Level", [1,2,3,4,5])

    if st.button("Save & Predict Risk"):
        risk = calculate_attrition_risk(js, wl, ms, cg, stress)

        new_row = {
            "EmployeeID": emp_id,
            "Name": name,
            "Department": department,
            "Role": role,
            "Tenure": tenure,
            "JobSatisfaction": js,
            "WorkLifeBalance": wl,
            "ManagerSupport": ms,
            "CareerGrowth": cg,
            "StressLevel": stress,
            "AttritionRisk": risk,
            "RiskBand": risk_band(risk)
        }
    if emp_id in employee_df["EmployeeID"].astype(str).values:
        st.error("Employee ID already exists. Duplicate entries are not allowed.")
        st.stop()

        employee_df = pd.concat([employee_df, pd.DataFrame([new_row])], ignore_index=True)
        employee_df.to_csv(DATA_FILE, index=False)

        st.success(f"Predicted Attrition Risk: {risk}%")

    st.markdown("### Stored Employee Data")
    st.dataframe(
    employee_df.style.apply(style_risk_rows, axis=1),
    use_container_width=True
)

    st.download_button(
        "Download Employee Data (CSV)",
        employee_df.to_csv(index=False),
        "orgaknow_employee_attrition_data.csv",
        "text/csv"
    )

if os.path.exists(ACTIONS_FILE):
    actions_df = pd.read_csv(ACTIONS_FILE)
else:
    actions_df = pd.DataFrame(columns=[
        "EmployeeID",
        "EmployeeName",
        "Department",
        "Manager",
        "RiskScore",
        "RiskBand",
        "SelectedAction",
        "ActionStatus",
        "ManagerComment"
    ])
# ------------------------------------
# Ensure Outcome Tracking Columns Exist
# ------------------------------------
if "OutcomeStatus" not in actions_df.columns:
    actions_df["OutcomeStatus"] = "Pending"

if "OutcomeDate" not in actions_df.columns:
    actions_df["OutcomeDate"] = ""

    st.markdown("## Data Management Controls")
    st.caption("Administrative controls for resetting and replacing employee data")

    col_reset1, col_reset2 = st.columns(2)

    # -----------------------------
    # OPTION 1 — ERASE ALL DATA
    # -----------------------------
    with col_reset1:
        st.markdown("### ⚠️ Erase All Employee Data")

        confirm_delete = st.checkbox(
            "I understand this will permanently delete all employee data"
        )

        if st.button("Erase Employee Master Data", type="secondary"):
            if not confirm_delete:
                st.warning("Please confirm before deleting data.")
            else:
                if os.path.exists(DATA_FILE):
                    os.remove(DATA_FILE)

                employee_df = pd.DataFrame(columns=[
                    "EmployeeID", "Name", "Department", "Role", "Tenure",
                    "JobSatisfaction", "WorkLifeBalance", "ManagerSupport",
                    "CareerGrowth", "StressLevel", "AttritionRisk", "RiskBand"
                ])

                employee_df.to_csv(DATA_FILE, index=False)
                st.success("All employee data erased successfully.")
                st.rerun()

    # -----------------------------
    # OPTION 2 — REPLACE ON UPLOAD
    # -----------------------------
    with col_reset2:
        st.markdown("### 🔄 Upload Fresh Data")

        replace_existing = st.checkbox(
            "Replace existing employee data with uploaded file"
        )

# =================================================
# TAB 2 — EXECUTIVE DASHBOARD
# =================================================
# -----------------------------
# ONLY GRAPH LABEL ENHANCEMENTS ADDED
# -----------------------------

# =============================
# TAB 2 — EXECUTIVE DASHBOARD
# =============================
with tab2:

    st.markdown("## CHRO Executive Dashboard")
    st.caption("Workforce Attrition Risk · Financial Impact · Risk Drivers")

    if employee_df.empty:
        st.warning("No employee data available.")
    else:
        total_emp = len(employee_df)

        high_df = employee_df[employee_df["RiskBand"] == "High"]
        med_df = employee_df[employee_df["RiskBand"] == "Medium"]
        low_df = employee_df[employee_df["RiskBand"] == "Low"]

        high_cnt = len(high_df)
        med_cnt = len(med_df)
        low_cnt = len(low_df)

        high_pct = round((high_cnt / total_emp) * 100, 1)
        expected_leavers = round(employee_df["AttritionRisk"].sum() / 100, 2)
        est_cost = int(expected_leavers * 500000)

        avg_risk = round(employee_df["AttritionRisk"].mean(), 2)
        risk_std = round(employee_df["AttritionRisk"].std(), 2)

        critical_roles = employee_df[
            employee_df["Role"].isin(["Executive", "Manager"])
        ]
        critical_high = len(critical_roles[critical_roles["RiskBand"] == "High"])

        dept_avg = employee_df.groupby("Department")["AttritionRisk"].mean()
        top_risk_dept = dept_avg.idxmax()
        low_risk_dept = dept_avg.idxmin()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Employees", total_emp)
        col2.metric("High Risk Employees", high_cnt)
        col3.metric("Medium Risk Employees", med_cnt)
        col4.metric("Low Risk Employees", low_cnt)

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("% Workforce High Risk", f"{high_pct}%")
        col6.metric("Expected Leavers", expected_leavers)
        col7.metric("Estimated Attrition Cost", f"${est_cost:,}")
        col8.metric("Average Attrition Risk", f"{avg_risk}%")

        col9, col10, col11, col12 = st.columns(4)
        col9.metric("High Risk Critical Roles", critical_high)
        col10.metric("Highest Risk Dept", top_risk_dept)
        col11.metric("Lowest Risk Dept", low_risk_dept)
        col12.metric("Risk Volatility Index", risk_std)

        st.markdown("---")

        # ---- Chart 1 — Risk Distribution
        fig1 = px.pie(employee_df, names="RiskBand", title="Workforce Risk Distribution")
        st.plotly_chart(fig1, use_container_width=True)

        # ---- Chart 2 — Department vs Avg Risk (LABELS ADDED)
        dept_df = dept_avg.reset_index()
        fig2 = px.bar(
            dept_df,
            x="Department",
            y="AttritionRisk",
            title="Average Attrition Risk by Department",
            text_auto=True
        )
        fig2.update_traces(textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)

        # ---- Chart 3 — Role vs Avg Risk (LABELS ADDED)
        role_df = employee_df.groupby("Role")["AttritionRisk"].mean().reset_index()
        fig3 = px.bar(
            role_df,
            x="Role",
            y="AttritionRisk",
            title="Average Attrition Risk by Role Level",
            text_auto=True
        )
        fig3.update_traces(textposition="outside")
        st.plotly_chart(fig3, use_container_width=True)

        # ---- Chart 4 — Risk Pyramid (LABELS ADDED)
        pyramid_df = pd.DataFrame({
            "RiskBand": ["Low", "Medium", "High"],
            "Headcount": [low_cnt, med_cnt, high_cnt]
        })

        fig4 = px.bar(
            pyramid_df,
            x="RiskBand",
            y="Headcount",
            title="Attrition Risk Pyramid",
            text_auto=True
        )
        fig4.update_traces(textposition="outside")
        st.plotly_chart(fig4, use_container_width=True)

        # ---- Chart 5 — Job Satisfaction vs Risk
        fig5 = px.scatter(
            employee_df,
            x="JobSatisfaction",
            y="AttritionRisk",
            color="RiskBand",
            title="Job Satisfaction vs Attrition Risk"
        )
        st.plotly_chart(fig5, use_container_width=True)

        # ---- Chart 6 — Stress vs Risk
        fig6 = px.box(
            employee_df,
            x="StressLevel",
            y="AttritionRisk",
            title="Stress Level vs Attrition Risk"
        )
        st.plotly_chart(fig6, use_container_width=True)

        # ---- Chart 7 — Heatmap
        heatmap_df = employee_df.pivot_table(
            values="AttritionRisk",
            index="Department",
            columns="Role",
            aggfunc="mean"
        )

        fig7 = px.imshow(
            heatmap_df,
            title="Attrition Risk Heatmap: Department × Role",
            aspect="auto"
        )
        st.plotly_chart(fig7, use_container_width=True)

        # ---- Chart 8 — Treemap (VALUES SHOWN)
        treemap_df = employee_df.groupby(
            ["Department", "RiskBand"]
        ).size().reset_index(name="Headcount")

        fig8 = px.treemap(
            treemap_df,
            path=["Department", "RiskBand"],
            values="Headcount",
            title="Workforce Risk Composition Tree Map"
        )
        fig8.update_traces(textinfo="label+value")
        st.plotly_chart(fig8, use_container_width=True)

with tab3:

    st.markdown("## Prescriptive Actions Engine")
    st.caption("From Risk Identification → Manager Action → Retention Outcome")

    # ------------------------------------
    # Simulated Role Selection (Phase-3.1)
    # ------------------------------------
    role_view = st.selectbox(
        "View As",
        ["CHRO", "HRBP", "Manager"]
    )

    manager_name = st.text_input(
        "Manager Name (for Manager / HRBP view)",
        value="Manager_A"
    )

    # ------------------------------------
    # Data Scope Logic (Hierarchical)
    # ------------------------------------
    scoped_df = employee_df.copy()

    if role_view == "Manager":
        scoped_df = scoped_df[scoped_df["Name"].notna()]  # placeholder for team logic

    if role_view == "HRBP":
        scoped_df = scoped_df  # future: restrict by BU / Dept

    # ------------------------------------
    # Focus on Medium + High Risk
    # ------------------------------------
    scoped_df = scoped_df[scoped_df["RiskBand"].isin(["High", "Medium"])]

    st.markdown("### At-Risk Employees Requiring Action")
    st.dataframe(
        scoped_df[[
            "EmployeeID", "Name", "Department",
            "Role", "AttritionRisk", "RiskBand"
        ]],
        use_container_width=True
    )

    st.markdown("---")

    # ------------------------------------
    # Action Selection
    # ------------------------------------
    st.markdown("### Action Planning for Selected Employee")

    emp_id = st.selectbox(
        "Select Employee ID",
        scoped_df["EmployeeID"].unique() if not scoped_df.empty else []
    )

    selected_emp = scoped_df[scoped_df["EmployeeID"] == emp_id]

    if not selected_emp.empty:

        emp = selected_emp.iloc[0]

        st.info(
            f"**{emp['Name']}** | Dept: {emp['Department']} | "
            f"Risk: {emp['AttritionRisk']}% ({emp['RiskBand']})"
        )

        recommended_action = st.selectbox(
            "Recommended Action",
            [
                "Career Path Discussion",
                "Compensation Review",
                "Manager Coaching / 1:1",
                "Internal Role Movement",
                "Workload Rebalancing",
                "Training / Upskilling",
                "Engagement Survey Follow-up",
                "No Action – Monitor"
            ]
        )

        action_status = st.selectbox(
            "Action Status",
            ["Planned", "In Progress", "Completed"]
        )

        manager_comment = st.text_area(
            "Manager / HRBP Comment",
            placeholder="Describe the action taken or planned..."
        )

        if st.button("Save Action Decision"):

            new_action = {
                "EmployeeID": emp["EmployeeID"],
                "EmployeeName": emp["Name"],
                "Department": emp["Department"],
                "Manager": manager_name,
                "RiskScore": emp["AttritionRisk"],
                "RiskBand": emp["RiskBand"],
                "SelectedAction": recommended_action,
                "ActionStatus": action_status,
                "ManagerComment": manager_comment
            }

            actions_df = pd.concat(
                [actions_df, pd.DataFrame([new_action])],
                ignore_index=True
            )

            actions_df.to_csv(ACTIONS_FILE, index=False)
            st.success("Action recorded successfully.")

    st.markdown("---")

    # ------------------------------------
    # ACTION MONITORING DASHBOARD
    # ------------------------------------
    st.markdown("### Action Monitoring & Effectiveness")

    if actions_df.empty:
        st.warning("No actions recorded yet.")
    else:
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Actions Logged", len(actions_df))
        col2.metric(
            "High Risk Covered",
            len(actions_df[actions_df["RiskBand"] == "High"])
        )
        col3.metric(
            "Actions In Progress",
            len(actions_df[actions_df["ActionStatus"] == "In Progress"])
        )
        col4.metric(
            "Completed Actions",
            len(actions_df[actions_df["ActionStatus"] == "Completed"])
        )

        # --------------------------------
        # VISUAL 1 — Action Status
        # --------------------------------
        fig1 = px.pie(
            actions_df,
            names="ActionStatus",
            title="Action Status Distribution"
        )
        st.plotly_chart(fig1, use_container_width=True)

        # --------------------------------
        # VISUAL 2 — Action Types
        # --------------------------------
        fig2 = px.bar(
            actions_df,
            x="SelectedAction",
            title="Types of Retention Actions Taken"
        )
        st.plotly_chart(fig2, use_container_width=True)

        # --------------------------------
        # VISUAL 3 — Risk vs Action Coverage
        # --------------------------------
        fig3 = px.histogram(
            actions_df,
            x="RiskScore",
            color="ActionStatus",
            title="Risk Score Coverage by Action Status"
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("### Detailed Action Log")
        st.dataframe(actions_df, use_container_width=True)
with tab4:

    st.markdown("## Reports & Downloads")
    st.caption("Executive-ready reports generated from live retention data")

    if employee_df.empty:
        st.warning("No employee data available to generate reports.")
    else:
        # ---------------------------------------
        # EXECUTIVE SUMMARY CALCULATIONS
        # ---------------------------------------
        total_emp = len(employee_df)
        high = len(employee_df[employee_df["RiskBand"] == "High"])
        medium = len(employee_df[employee_df["RiskBand"] == "Medium"])
        low = len(employee_df[employee_df["RiskBand"] == "Low"])

        expected_leavers = round(employee_df["AttritionRisk"].sum() / 100, 2)
        est_cost = int(expected_leavers * 500000)

        exec_summary = pd.DataFrame({
            "Metric": [
                "Total Employees",
                "High Risk Employees",
                "Medium Risk Employees",
                "Low Risk Employees",
                "Expected Leavers",
                "Estimated Attrition Cost ($)"
            ],
            "Value": [
                total_emp, high, medium, low,
                expected_leavers, est_cost
            ]
        })

        # ---------------------------------------
        # HIGH-RISK EMPLOYEE LIST
        # ---------------------------------------
        high_risk_emps = employee_df[
            employee_df["RiskBand"] == "High"
        ].sort_values("AttritionRisk", ascending=False)

        # ---------------------------------------
        # ACTION SUMMARY (FROM ACTIONS ENGINE)
        # ---------------------------------------
        if "actions_df" in globals() and not actions_df.empty:
            action_summary = actions_df
        else:
            action_summary = pd.DataFrame(
                {"Info": ["No actions logged yet"]}
            )

        # ---------------------------------------
        # SEGMENT STATS
        # ---------------------------------------
        segment_stats = employee_df.groupby(
            "Department"
        ).agg(
            Headcount=("EmployeeID", "count"),
            Avg_Risk=("AttritionRisk", "mean"),
            High_Risk_Count=("RiskBand", lambda x: (x == "High").sum())
        ).reset_index()

        # ---------------------------------------
        # CREATE EXCEL REPORT
        # ---------------------------------------
        output = BytesIO()

        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            exec_summary.to_excel(
                writer, sheet_name="Executive Summary", index=False
            )
            high_risk_emps.to_excel(
                writer, sheet_name="High Risk Employees", index=False
            )
            segment_stats.to_excel(
                writer, sheet_name="Department Risk Stats", index=False
            )
            action_summary.to_excel(
                writer, sheet_name="Retention Actions Log", index=False
            )

        output.seek(0)

        # ---------------------------------------
        # DOWNLOAD BUTTONS
        # ---------------------------------------
        st.download_button(
            label="Download Executive & HRBP Report (Excel)",
            data=output,
            file_name="OrgaKnow_Retention_Intelligence_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.markdown("---")

        st.markdown("### What This Report Includes")
        st.markdown("""
        - Executive Summary KPIs  
        - Prioritized High-Risk Employees  
        - Department-wise Risk Statistics  
        - Manager / HRBP Action Tracking  
        """)

        st.success("Report ready for CHRO / HRBP review.")
with tab5:

    st.markdown("## Action Effectiveness Analytics")
    st.caption("Evaluating which retention actions reduce attrition risk")

    if actions_df.empty or employee_df.empty:
        st.warning("Insufficient data to evaluate action effectiveness.")
    else:
        # -----------------------------------------
        # MERGE ACTIONS WITH CURRENT RISK
        # -----------------------------------------
        merged_df = actions_df.merge(
            employee_df[["EmployeeID", "AttritionRisk"]],
            on="EmployeeID",
            how="left",
            suffixes=("_AtAction", "_Current")
        )

        # Rename for clarity
        merged_df.rename(columns={
            "RiskScore": "Risk_At_Action",
            "AttritionRisk": "Risk_Current"
        }, inplace=True)

        merged_df["Risk_Change"] = (
            merged_df["Risk_At_Action"] - merged_df["Risk_Current"]
        )
        merged_df["RiskMovement"] = merged_df["Risk_Change"].apply(risk_arrow)


        # -----------------------------------------
        # KPIs — ACTION EFFECTIVENESS
        # -----------------------------------------
        total_actions = len(merged_df)
        improved_cases = len(merged_df[merged_df["Risk_Change"] > 0])
        worsened_cases = len(merged_df[merged_df["Risk_Change"] < 0])
        avg_risk_reduction = round(merged_df["Risk_Change"].mean(), 2)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Actions Evaluated", total_actions)
        col2.metric("Risk Reduced Cases", improved_cases)
        col3.metric("Risk Increased Cases", worsened_cases)
        col4.metric("Avg Risk Change (%)", avg_risk_reduction)

        st.markdown("---")

        # -----------------------------------------
        # CHART 1 — ACTION EFFECTIVENESS BAR
        # -----------------------------------------
        action_effect = merged_df.groupby(
            "SelectedAction"
        )["Risk_Change"].mean().reset_index()

        fig1 = px.bar(
            action_effect,
            x="SelectedAction",
            y="Risk_Change",
            title="Average Risk Reduction by Action Type"
        )
        st.plotly_chart(fig1, use_container_width=True)

        # -----------------------------------------
        # CHART 2 — ACTION SUCCESS RATE
        # -----------------------------------------
        merged_df["ActionSuccess"] = merged_df["Risk_Change"].apply(
            lambda x: "Effective" if x > 0 else "Not Effective"
        )

        fig2 = px.histogram(
            merged_df,
            x="SelectedAction",
            color="ActionSuccess",
            title="Action Effectiveness Distribution"
        )
        st.plotly_chart(fig2, use_container_width=True)

        # -----------------------------------------
        # CHART 3 — RISK CHANGE DISTRIBUTION
        # -----------------------------------------
        fig3 = px.box(
            merged_df,
            x="SelectedAction",
            y="Risk_Change",
            title="Risk Change Distribution by Action"
        )
        st.plotly_chart(fig3, use_container_width=True)

        # -----------------------------------------
        # CHART 4 — HIGH-RISK FOCUS
        # -----------------------------------------
        high_risk_actions = merged_df[
            merged_df["RiskBand"] == "High"
        ]

        fig4 = px.bar(
            high_risk_actions.groupby("SelectedAction")["Risk_Change"].mean().reset_index(),
            x="SelectedAction",
            y="Risk_Change",
            title="Action Effectiveness for High-Risk Employees"
        )
        st.plotly_chart(fig4, use_container_width=True)

        # -----------------------------------------
        # CHART 5 — MANAGER VIEW (OPTIONAL FILTER)
        # -----------------------------------------
        st.markdown("### Drill-Down: Manager View")

        manager_filter = st.selectbox(
            "Select Manager",
            merged_df["Manager"].unique()
        )

        mgr_df = merged_df[merged_df["Manager"] == manager_filter]

        fig5 = px.scatter(
            mgr_df,
            x="Risk_At_Action",
            y="Risk_Current",
            color="SelectedAction",
            title=f"Risk Movement for {manager_filter}'s Team"
        )
        st.plotly_chart(fig5, use_container_width=True)

        st.markdown("---")
        st.markdown("### Action Effectiveness Table")
        st.dataframe(
            merged_df[[
                "EmployeeID",
                "EmployeeName",
                "SelectedAction",
                "Risk_At_Action",
                "Risk_Current",
                "Risk_Change",
                "ActionStatus"
            ]],
            use_container_width=True
        )
with tab6:

    st.markdown("## Outcome Tracking")
    st.caption("Tracking retention outcomes to measure real business impact")

    if actions_df.empty:
        st.warning("No action records available for outcome tracking.")
    else:
        # ------------------------------------
        # OUTCOME UPDATE SECTION
        # ------------------------------------
        st.markdown("### Update Employee Outcome")

        emp_id = st.selectbox(
            "Select Employee ID",
            actions_df["EmployeeID"].unique()
        )

        emp_action = actions_df[actions_df["EmployeeID"] == emp_id].iloc[-1]

        st.info(
            f"**{emp_action['EmployeeName']}** | "
            f"Risk at Action: {emp_action['RiskScore']}% | "
            f"Action: {emp_action['SelectedAction']}"
        )

        outcome = st.selectbox(
            "Outcome Status",
            ["Pending", "Stayed", "Left"]
        )

        outcome_date = st.date_input("Outcome Date")

        if st.button("Save Outcome"):
            idx = actions_df[actions_df["EmployeeID"] == emp_id].index[-1]
            actions_df.loc[idx, "OutcomeStatus"] = outcome
            actions_df.loc[idx, "OutcomeDate"] = str(outcome_date)

            actions_df.to_csv(ACTIONS_FILE, index=False)
            st.success("Outcome updated successfully.")

        st.markdown("---")

        # ------------------------------------
        # OUTCOME KPIs
        # ------------------------------------
        total_tracked = len(actions_df)
        stayed = len(actions_df[actions_df["OutcomeStatus"] == "Stayed"])
        left = len(actions_df[actions_df["OutcomeStatus"] == "Left"])
        pending = len(actions_df[actions_df["OutcomeStatus"] == "Pending"])

        retention_rate = round(
            (stayed / (stayed + left)) * 100, 1
        ) if (stayed + left) > 0 else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Employees Tracked", total_tracked)
        col2.metric("Retained (Stayed)", stayed)
        col3.metric("Attrited (Left)", left)
        col4.metric("Retention Success Rate", f"{retention_rate}%")

        st.markdown("---")

        # ------------------------------------
        # VISUAL 1 — OUTCOME DISTRIBUTION
        # ------------------------------------
        fig1 = px.pie(
            actions_df,
            names="OutcomeStatus",
            title="Retention Outcomes Distribution"
        )
        st.plotly_chart(fig1, use_container_width=True)

        # ------------------------------------
        # VISUAL 2 — OUTCOME BY ACTION TYPE
        # ------------------------------------
        fig2 = px.histogram(
            actions_df,
            x="SelectedAction",
            color="OutcomeStatus",
            title="Outcome by Action Type"
        )
        st.plotly_chart(fig2, use_container_width=True)

        # ------------------------------------
        # VISUAL 3 — HIGH-RISK RETENTION
        # ------------------------------------
        high_risk_outcomes = actions_df[
            actions_df["RiskBand"] == "High"
        ]

        fig3 = px.pie(
            high_risk_outcomes,
            names="OutcomeStatus",
            title="High-Risk Employee Outcomes"
        )
        st.plotly_chart(fig3, use_container_width=True)

        # ------------------------------------
        # VISUAL 4 — RISK VS OUTCOME
        # ------------------------------------
        fig4 = px.box(
            actions_df,
            x="OutcomeStatus",
            y="RiskScore",
            title="Risk Score vs Outcome"
        )
        st.plotly_chart(fig4, use_container_width=True)

        st.markdown("---")

        st.markdown("### Outcome Tracking Log")
        st.dataframe(
            actions_df[[
                "EmployeeID",
                "EmployeeName",
                "SelectedAction",
                "RiskScore",
                "OutcomeStatus",
                "OutcomeDate",
                "Manager"
            ]],
            use_container_width=True
        )
st.markdown("---")
st.caption(
    "Retention Intelligence · Decision-support analytics. "
    "Predictions are probabilistic and should be combined with HR judgment."
)







