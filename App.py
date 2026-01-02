import streamlit as st
import pandas as pd
import time
import os
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ---------------- CONFIGURATION ----------------
st.set_page_config(page_title="OrgaKnow | AI Retention Intelligence", layout="wide")

# ---------------- GLOBAL CONSTANTS ----------------
EMPLOYEE_FILE = "employee_turnover_with_ids.csv"
EXIT_FILE = "exit_intelligence.csv"
ACTIONS_FILE = "attrition_actions.csv"
SESSION_TIMEOUT = 3600



# ---------------- HELPER FUNCTIONS ----------------

def explain_risk(emp_row, full_df):
    """Generates plain-text explanations for why an employee is at risk."""
    reasons = []
    # Check bounds to ensure we don't crash on missing data
    if "satisfaction" in emp_row and "satisfaction" in full_df.columns:
        if emp_row["satisfaction"] < full_df["satisfaction"].mean():
            reasons.append("Low job satisfaction")

    if "average_monthly_hours" in emp_row and "average_monthly_hours" in full_df.columns:
        if emp_row["average_monthly_hours"] > full_df["average_monthly_hours"].mean():
            reasons.append("Very high working hours")

    if "time_spend_company" in emp_row and "time_spend_company" in full_df.columns:
        if emp_row["time_spend_company"] > full_df["time_spend_company"].mean():
            reasons.append("Long time in same role")

    if "promotion" in emp_row and emp_row["promotion"] == 0:
        reasons.append("No recent promotion")

    if not reasons:
        reasons.append("Risk driven by complex combined factors")
    return reasons

def safe_encode(le, series):
    """Optimized: Encodes the entire column instantly using a dictionary map."""
    # Create a fast lookup dictionary
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    # Map everything at once (Vectorized)
    return series.map(mapping).fillna(-1).astype(int)

@st.cache_resource
def train_risk_model(data):
    """
    Trains the model ONCE and caches it in memory. 
    This prevents the 'loading too much' issue and eliminates missing file errors.
    """
    df = data.copy()
    
    # 1. Encoders
    le_dept = LabelEncoder()
    le_salary = LabelEncoder()
    
    if 'department' in df.columns:
        df['dept_encoded'] = le_dept.fit_transform(df['department'].astype(str))
    else:
        df['dept_encoded'] = 0 # Fallback
        
    if 'salary' in df.columns:
        df['salary_encoded'] = le_salary.fit_transform(df['salary'].astype(str))
    else:
        df['salary_encoded'] = 0 # Fallback
        
    # 2. Define Features
    feature_cols = [
        'satisfaction', 'evaluation', 'number_of_projects', 
        'average_monthly_hours', 'time_spend_company', 
        'work_accident', 'promotion', 'dept_encoded', 'salary_encoded'
    ]
    
    # Filter only columns that actually exist
    features_present = [c for c in feature_cols if c in df.columns]
    
    if not features_present:
        return None, [], None, None
        
    X = df[features_present]
    
    # Check if 'churn' exists
    if 'churn' not in df.columns:
        return None, [], None, None

    y = df['churn']
    
    # 3. Train
    model = LogisticRegression(max_iter=1000, solver='lbfgs')
    model.fit(X, y)
    
    return model, features_present, le_dept, le_salary

def get_risk_band(score):
    if score >= 70: return "High"
    elif score >= 40: return "Medium"
    else: return "Low"

def kpi_card(title, value, tone="neutral"):
    colors = {"high": "#7f1d1d", "medium": "#78350f", "low": "#14532d", "neutral": "#0f172a"}
    bg = colors.get(tone, "#0f172a")
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {bg}, #020617); border: 1px solid #1e293b; border-radius: 12px; padding: 15px; text-align: center; margin-bottom: 10px;">
        <div style="color: #94a3b8; font-size: 0.8rem; font-weight: 600;">{title}</div>
        <div style="color: #e5e7eb; font-size: 1.6rem; font-weight: 800;">{value}</div>
    </div>""", unsafe_allow_html=True)

def style_risk_rows(row):
    if "RiskBand" not in row: return [""] * len(row)
    color_map = {"High": "#450a0a", "Medium": "#451a03", "Low": "#052e16"}
    bg = color_map.get(row["RiskBand"], "")
    return [f"background-color: {bg}; color: white" if bg else ""] * len(row)

# ---------------- AUTH & SESSION ----------------
if "login_time" not in st.session_state:
    st.session_state.login_time = time.time()
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Login Logic
if not st.session_state.logged_in:
    st.markdown("<h1 style='text-align: center;'>OrgaKnow Login</h1>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login", use_container_width=True):
            # Simple auth for demo
            if username: 
                st.session_state.logged_in = True
                st.session_state.user = {"username": username, "role": "CHRO", "department": "All"}
                st.rerun()
            else:
                st.error("Please enter a username.")
    st.stop()

# Logout Logic
with st.sidebar:
    st.success(f"User: {st.session_state.user['username']}")
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

st.sidebar.markdown("### 📂 Data Source")
uploaded_file = st.sidebar.file_uploader("Upload Current Employee List (CSV)", type="csv")

if uploaded_file:
    # Use uploaded file
    raw_df = pd.read_csv(uploaded_file)
else:
    # Use default/demo file if nothing uploaded
    if os.path.exists(EMPLOYEE_FILE):
        raw_df = pd.read_csv(EMPLOYEE_FILE)
    else:
        st.error("Please upload an employee CSV.")
        st.stop()       

# ---------------- MAIN APP LOGIC ----------------
# 1. Load Data
employee_df_full = pd.DataFrame() # Holds Active employees for KPIs
employee_df_view = pd.DataFrame() # Holds the list to show in the UI
ml_model = None

if os.path.exists(EMPLOYEE_FILE):
    raw_df = pd.read_csv(EMPLOYEE_FILE)
    
    # --- FIX TYPOS ---
    if 'average_montly_hours' in raw_df.columns:
        raw_df = raw_df.rename(columns={'average_montly_hours': 'average_monthly_hours'})
    
    # Clean duplicates
    raw_df = raw_df.drop_duplicates(subset=["EmployeeID"], keep="last")

    # --- ADD STATUS COLUMN ---
    # 0 = Active, 1 = Left
    raw_df['Status'] = raw_df['churn'].apply(lambda x: 'Active' if x == 0 else 'Inactive')

    # Separate into Active and Inactive
    active_df = raw_df[raw_df['churn'] == 0].copy()
    inactive_df = raw_df[raw_df['churn'] == 1].copy()

    # --- TRAIN & PREDICT (ONLY FOR ACTIVE EMPLOYEES) ---
    ml_model, ml_features, ml_le_dept, ml_le_salary = train_risk_model(raw_df)
    
    if ml_model and not active_df.empty:
        # Encode
        if 'department' in active_df.columns:
            active_df['dept_encoded'] = safe_encode(ml_le_dept, active_df['department'].astype(str))
        if 'salary' in active_df.columns:
            active_df['salary_encoded'] = safe_encode(ml_le_salary, active_df['salary'].astype(str))

        # Predict
        pred_df = active_df.copy()
        missing_cols = [c for c in ml_features if c not in pred_df.columns]
        
        if not missing_cols:
            X_pred = pred_df[ml_features]
            probs = ml_model.predict_proba(X_pred)[:, 1]
            active_df['AttritionRisk'] = (probs * 100).round(2)
            active_df['RiskBand'] = active_df['AttritionRisk'].apply(get_risk_band)
        else:
             active_df['AttritionRisk'] = 0.0
             active_df['RiskBand'] = "Unknown"
    
    # --- HANDLE INACTIVE EMPLOYEES ---
    # We do NOT calculate risk for them. We just set placeholders.
    if not inactive_df.empty:
        inactive_df['dept_encoded'] = 0
        inactive_df['salary_encoded'] = 0
        inactive_df['AttritionRisk'] = 0.0  # Risk is 0 because they are already gone
        inactive_df['RiskBand'] = "Inactive" # Special label

    # --- MERGE BACK TOGETHER ---
    employee_df_full = active_df.copy() # Keeps full active list for KPI calculations
    
    # Create name if missing (for both)
    for df in [active_df, inactive_df]:
        if 'Name' not in df.columns:
            df['Name'] = "Employee " + df['EmployeeID'].astype(str)

    # --- SIDEBAR FILTER ---
    st.sidebar.markdown("---")
    view_option = st.sidebar.radio("📋 View Status", ["Active Only", "Inactive (Left)", "Show All"], index=0)

    if view_option == "Active Only":
        employee_df_view = active_df
    elif view_option == "Inactive (Left)":
        employee_df_view = inactive_df
    else:
        # Combine both for the list view
        employee_df_view = pd.concat([active_df, inactive_df], ignore_index=True)

else:
    st.error(f"Data file '{EMPLOYEE_FILE}' not found.")

# 2. Load Actions
if os.path.exists(ACTIONS_FILE):
    actions_df = pd.read_csv(ACTIONS_FILE)
else:
    actions_df = pd.DataFrame(columns=[
        "EmployeeID", "EmployeeName", "Department", "Manager",
        "RiskScore", "RiskBand", "SelectedAction", "ActionStatus",
        "ManagerComment", "OutcomeStatus", "OutcomeDate"
    ])

# 3. Load Exits (Historical)
if os.path.exists(EXIT_FILE):
    exit_df = pd.read_csv(EXIT_FILE)
else:
    exit_df = pd.DataFrame(columns=["EmployeeID", "ExitDate", "ExitType", "PrimaryExitReason", "ActionTaken"])

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "AI Risk Predictions", 
    "Executive Overview", 
    "Prescriptive Actions", 
    "Reports", 
    "Action Effectiveness", 
    "Outcome Tracking"
])

# =================================================
# TAB 1 — AI RISK PREDICTIONS
# =================================================
with tab1:
    # TAB 1: PREDICTIONS
    if not employee_df_view.empty and 'AttritionRisk' in employee_df_view.columns:
        k1, k2, k3 = st.columns(3)
        with k1: kpi_card("High Risk", len(employee_df_view[employee_df_view["RiskBand"]=="High"]), "high")
        with k2: kpi_card("Medium Risk", len(employee_df_view[employee_df_view["RiskBand"]=="Medium"]), "medium")
        with k3: kpi_card("Low Risk", len(employee_df_view[employee_df_view["RiskBand"]=="Low"]), "low")
        
        st.markdown("### 🧠 AI Risk Assessment")
        
        col1, col2 = st.columns([3, 1])
        with col1: search = st.text_input("Search Employee ID or Dept")
        with col2: filter_band = st.selectbox("Filter Risk Band", ["All", "High", "Medium", "Low"])
            
        view_df = employee_df_view.copy()
        if filter_band != "All": view_df = view_df[view_df['RiskBand'] == filter_band]
        if search:
            view_df = view_df[view_df['EmployeeID'].astype(str).str.contains(search) | 
                              view_df['department'].astype(str).str.contains(search, case=False)]

        # --- FIX: Reset Index to prevent sorting crash ---
        view_df = view_df.reset_index(drop=True)
        # -------------------------------------------------

        st.dataframe(
            view_df[['EmployeeID', 'department', 'AttritionRisk', 'RiskBand']].style.apply(style_risk_rows, axis=1), 
            use_container_width=True,
            column_config={
                "AttritionRisk": st.column_config.ProgressColumn("Risk Score", format="%.2f%%", min_value=0, max_value=100)
            }
        )
        
        risk_dist = view_df["RiskBand"].value_counts().reset_index()
        risk_dist.columns = ["RiskBand", "Count"]
        st.plotly_chart(px.pie(risk_dist, names="RiskBand", values="Count", title="Risk Distribution"), use_container_width=True)
    else:
        st.info("No risk data available yet.")

# =================================================
# TAB 2 — EXECUTIVE OVERVIEW (ALL CHARTS HERE)
# =================================================
with tab2:
    if not employee_df_full.empty and 'AttritionRisk' in employee_df_full.columns:
        
        # --- 1. Calculate Advanced KPIs ---
        total_emp = len(employee_df_full)
        total_active = len(active_df)
        total_inactive = len(inactive_df)
        total_records = len(raw_df)
        total_records = total_active + total_inactive

        # Risk Counts (Active Only)
        high_risk = len(active_df[active_df["RiskBand"]=="High"]) if not active_df.empty else 0
        med_risk = len(active_df[active_df["RiskBand"]=="Medium"]) if not active_df.empty else 0
        low_risk = len(active_df[active_df["RiskBand"]=="Low"]) if not active_df.empty else 0
        
        # A. Expected Leavers
        expected_leavers = int(active_df["AttritionRisk"].sum() / 100)
        
        # B. Projected Attrition Rate
        proj_attrition = (expected_leavers / total_active) * 100
        
        # C. Financial Risk
        salary_est = {"low": 50000, "medium": 90000, "high": 150000}
        
        if "salary" in employee_df_full.columns:
            temp_salary = employee_df_full["salary"].map(salary_est).fillna(60000)
            cost_series = temp_salary * 0.5 * (employee_df_full["AttritionRisk"] / 100)
            total_risk_cost = cost_series.sum()
        else:
            total_risk_cost = 0

        # --- 2. Display Top Row KPIs (Counts) ---
        st.markdown("### 🚦 Headcount Risk")
        c1, c2, c3, = st.columns(3)
        
# Total Database
        with c1: kpi_card("Total Employees (All)", total_records)
        
        # Active
        with c2: kpi_card("✅ Active Employees", total_active, "low")
        
        # Inactive
        with c3: kpi_card("❌ Inactive (Left)", total_inactive, "medium")
        
 
  # --- 3. ROW 2: ACTIVE RISK BREAKDOWN ---
        st.markdown("### ⚠️ Risk Breakdown (Active Only)")
        r1, r2, r3 = st.columns(3)
        with r1: kpi_card("High Risk", high_risk, "high")      # Red background
        with r2: kpi_card("Medium Risk", med_risk, "medium")   # Orange/Brown background
        with r3: kpi_card("Low Risk", low_risk, "low")         # Green background      
        st.markdown("---")

        # --- 3. Display Second Row KPIs (Business Impact) ---
        st.markdown("### 💰 Projected Business Impact (Next 12 Months)")
        k1, k2, k3 = st.columns(3)
        
        with k1: 
            kpi_card(
                "📉 Expected Exits", 
                f"{expected_leavers} People", 
                "high" if expected_leavers > (total_active * 0.1) else "medium"
            )
            st.caption(f"Based on sum of retention probabilities.")
            
        with k2: 
            kpi_card(
                "📊 Projected Rate", 
                f"{proj_attrition:.1f}%", 
                "high" if proj_attrition > 15 else "low"
            )
            st.caption("Industry average is ~10-15%.")

        with k3: 
            kpi_card(
                "💸 Est. Attrition Cost", 
                f"₹{total_risk_cost/1000000:.1f} M", 
                "high"
            )
            st.caption("Calculated as: Salary × 0.5 (Replacement Cost) × Risk Probability")

        st.markdown("---")

        # --- 4. Charts (Combined) ---
        c_chart1, c_chart2, c_chart3 = st.columns(3)
        with c_chart1:
            if 'department' in employee_df_full.columns:
                st.plotly_chart(px.bar(employee_df_full.groupby("department")["AttritionRisk"].mean().reset_index(), 
                                       x="department", y="AttritionRisk", title="Average Risk by Department", color="AttritionRisk"), use_container_width=True)
        with c_chart2:
            if 'satisfaction' in employee_df_full.columns:
                st.plotly_chart(px.scatter(employee_df_full, x="satisfaction", y="AttritionRisk", color="RiskBand", 
                                           title="Satisfaction Impact on Risk"), use_container_width=True)
                
        with c_chart3:
            # NEW: Active vs Inactive Chart
            status_counts = pd.DataFrame({
                "Status": ["Active", "Inactive"], 
                "Count": [total_active, total_inactive]
            })
            fig_status = px.pie(status_counts, names="Status", values="Count", title="Active vs. Inactive Employees", hole=0.4, color_discrete_sequence=["#10b981", "#ef4444"])
            st.plotly_chart(fig_status, use_container_width=True)
                                           
        st.markdown("---")
        
        # --- 5. Advanced Charts (Moved from other tabs) ---
        c_adv1, c_adv2 = st.columns(2)
        
        with c_adv1:
            # Heatmap (Moved from Tab 1)
            if 'salary' in employee_df_full.columns and 'department' in employee_df_full.columns:
                st.markdown("### 🔥 Risk Heatmap")
                heatmap_data = employee_df_full.groupby(['department', 'salary'])['AttritionRisk'].mean().reset_index()
                fig_heat = px.density_heatmap(
                    heatmap_data, 
                    x="department", 
                    y="salary", 
                    z="AttritionRisk", 
                    text_auto=True,
                    title="Risk Heatmap (Dept vs Salary)",
                    color_continuous_scale="RdYlGn_r"
                )
                st.plotly_chart(fig_heat, use_container_width=True)
                
        with c_adv2:
            # Flight Risk Matrix (Moved from Tab 4)
            if 'evaluation' in employee_df_full.columns:
                st.markdown("### ✈️ Flight Risk Matrix")
                fig_quad = px.scatter(
                    employee_df_full, 
                    x="evaluation", 
                    y="AttritionRisk", 
                    color="RiskBand",
                    hover_data=["EmployeeID", "department"],
                    title="Performance vs. Risk"
                )
                fig_quad.add_hline(y=70, line_dash="dash", annotation_text="High Risk")
                fig_quad.add_vline(x=0.7, line_dash="dash", annotation_text="High Perf")
                st.plotly_chart(fig_quad, use_container_width=True)

    else:
        st.warning("Dashboard waiting for data...")

# =================================================
# TAB 3 — PRESCRIPTIVE ACTIONS
# =================================================
with tab3:
    st.markdown("### 🔮 Simulator")
    if not employee_df_view.empty and ml_model:
        sim_emp = st.selectbox("Select Employee for Simulation", employee_df_view["EmployeeID"].unique())
        
        if sim_emp:
            curr_row = employee_df_view[employee_df_view["EmployeeID"]==sim_emp].iloc[0]
            st.write(f"Current Risk: **{curr_row['AttritionRisk']}%**")
            
            c1, c2 = st.columns(2)
            with c1:
                new_hours = st.slider("New Monthly Hours", 100, 350, int(curr_row.get("average_monthly_hours", 200)))
            with c2:
                new_promo = st.selectbox("Promote?", [0, 1], index=int(curr_row.get("promotion", 0)))
                
            # Simulate Button
            if st.button("Run Simulation"):
                sim_data = pd.DataFrame([curr_row.to_dict()])
                sim_data["average_monthly_hours"] = new_hours
                sim_data["promotion"] = new_promo
                
                # Re-encode is NOT needed here because we used 'curr_row.to_dict()' 
                # which already contains the 'dept_encoded' we saved in step 1.
                
                # Predict
                if set(ml_features).issubset(sim_data.columns):
                    X_sim = sim_data[ml_features]
                    new_prob = ml_model.predict_proba(X_sim)[0][1] * 100
                    delta = new_prob - curr_row['AttritionRisk']
                    
                    st.metric("New Risk Score", f"{new_prob:.2f}%", f"{delta:.2f}%", delta_color="inverse")
                else:
                    st.error("Simulator Error: Missing encoded columns.")

    st.markdown("---")
    st.markdown("### 📝 Action Planning")
    
    if not employee_df_view.empty:
        # Check if RiskBand exists before filtering
        if "RiskBand" in employee_df_view.columns:
            risk_employees = employee_df_view[employee_df_view["RiskBand"].isin(["High", "Medium"])]
        else:
            risk_employees = pd.DataFrame()
            
        action_emp_id = st.selectbox("Select At-Risk Employee", risk_employees["EmployeeID"].unique() if not risk_employees.empty else [])
        
        if action_emp_id:
            emp = risk_employees[risk_employees["EmployeeID"] == action_emp_id].iloc[0]
            reasons = explain_risk(emp, employee_df_full)
            for r in reasons: st.warning(r)
            
            c1, c2 = st.columns(2)
            with c1: action = st.selectbox("Action", ["1:1 Coaching", "Workload Adjustment", "Salary Review", "Training"])
            with c2: status = st.selectbox("Status", ["Planned", "In Progress", "Completed"])
            comment = st.text_area("Notes")
            
            if st.button("Save Action Plan"):
                new_action = {
                    "EmployeeID": emp["EmployeeID"],
                    "EmployeeName": emp.get("Name", f"Emp {emp['EmployeeID']}"),
                    "Department": emp["department"],
                    "Manager": st.session_state.user["username"],
                    "RiskScore": emp["AttritionRisk"],
                    "RiskBand": emp["RiskBand"],
                    "SelectedAction": action,
                    "ActionStatus": status,
                    "ManagerComment": comment,
                    "OutcomeStatus": "Pending",
                    "OutcomeDate": ""
                }
                actions_df = pd.concat([actions_df, pd.DataFrame([new_action])], ignore_index=True)
                actions_df.to_csv(ACTIONS_FILE, index=False)
                st.success("Action Saved!")

# =================================================
# TAB 4 — REPORTS
# =================================================
# =================================================
# TAB 4 — REPORTS
# =================================================
with tab4:
    st.markdown("## Reports Download")
    
    # Option to download FULL database (Active + Inactive)
    if not employee_df_view.empty: # This takes whatever is in view or raw
        csv = raw_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Complete Database (CSV)", csv, "all_employees.csv", "text/csv")
        
    report_type = st.selectbox("Select Specific Report", ["High Risk (Active)", "Inactive / Leavers", "Action Tracker"])
    
    if st.button("Generate & Download"):
        r_df = pd.DataFrame()
        
        if report_type == "High Risk (Active)" and not active_df.empty:
            r_df = active_df[active_df["RiskBand"]=="High"]
            
        elif report_type == "Inactive / Leavers" and not inactive_df.empty:
            r_df = inactive_df
            
        elif report_type == "Action Tracker":
            r_df = actions_df
            
        if not r_df.empty:
            st.download_button(f"Download {report_type}", r_df.to_csv(index=False).encode('utf-8'), f"{report_type}.csv")
        else:
            st.error("No records found for this report.")
            
    # Fairness Check (On Active Employees Only)
    st.markdown("### ⚖️ AI Fairness Check (Active Staff)")
    if 'salary' in active_df.columns:
        fairness_check = active_df.groupby("salary")["AttritionRisk"].mean().reset_index()
        st.dataframe(fairness_check, column_config={
            "AttritionRisk": st.column_config.ProgressColumn("Avg Risk", min_value=0, max_value=100)
        }, hide_index=True)

# =================================================
# TAB 5 — ACTION EFFECTIVENESS
# =================================================
with tab5:
    st.markdown("## Action Effectiveness")
    if actions_df.empty:
        st.info("No actions recorded yet.")
    elif not employee_df_full.empty:
        # Merge current risk to see if it dropped
        merged = actions_df.merge(employee_df_full[['EmployeeID', 'AttritionRisk']], on="EmployeeID", how="left", suffixes=('_Old', '_New'))
        
        # Calculate Delta
        merged['RiskDelta'] = merged['RiskScore'] - merged['AttritionRisk']
        
        avg_delta = merged["RiskDelta"].mean()
        kpi_card("Avg Risk Reduction", f"{avg_delta:.2f}", "low" if avg_delta > 0 else "high")
        
        st.dataframe(merged[['EmployeeID', 'SelectedAction', 'RiskScore', 'AttritionRisk', 'RiskDelta']])
        
        if not merged.empty:
            fig_eff = px.bar(merged, x="SelectedAction", y="RiskDelta", title="Risk Reduction by Action")
            st.plotly_chart(fig_eff, use_container_width=True)

# =================================================
# TAB 6 — OUTCOME TRACKING
# =================================================
with tab6:
    st.markdown("## Outcome Tracking")
    
    # 1. Update Outcomes
    if not actions_df.empty:
        st.markdown("### Update Employee Status")
        track_emp = st.selectbox("Employee", actions_df["EmployeeID"].unique(), key="track_emp")
        outcome = st.selectbox("Final Outcome", ["Stayed", "Left"], key="outcome_sel")
        
        if st.button("Update Outcome"):
            idx = actions_df[actions_df["EmployeeID"] == track_emp].index[-1]
            actions_df.loc[idx, "OutcomeStatus"] = outcome
            actions_df.to_csv(ACTIONS_FILE, index=False)
            st.success("Updated!")
            
        # Charts
        outcome_counts = actions_df["OutcomeStatus"].value_counts().reset_index()
        outcome_counts.columns = ["Outcome", "Count"]
        fig_out = px.pie(outcome_counts, names="Outcome", values="Count", title="Retention Outcomes")
        st.plotly_chart(fig_out, use_container_width=True)

    # 2. Upload Historical Exits
    st.markdown("---")
    st.markdown("### Upload Exit Data")
    uploaded_file = st.file_uploader("Upload CSV (EmployeeID, ExitDate)", type=["csv"])
    if uploaded_file:
        try:
            new_exits = pd.read_csv(uploaded_file)
            if "EmployeeID" in new_exits.columns:
                if os.path.exists(EXIT_FILE):
                    old_exits = pd.read_csv(EXIT_FILE)
                    final_exits = pd.concat([old_exits, new_exits]).drop_duplicates(subset=["EmployeeID"])
                else:
                    final_exits = new_exits
                
                final_exits.to_csv(EXIT_FILE, index=False)
                st.success(f"Processed {len(new_exits)} exit records.")
            else:
                st.error("CSV must have 'EmployeeID' column.")
        except Exception as e:
            st.error(f"Error: {e}")
