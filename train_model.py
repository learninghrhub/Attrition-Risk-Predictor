import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Constants
EMPLOYEE_FILE = "employee_turnover_with_ids.csv"
MODEL_FILE = "retention_model.pkl"
def recommend_action(row):
    risk = row["AttritionRisk"]
    perf = row.get("evaluation", 0.5)
    sat = row.get("satisfaction", 0.5)
    
    if risk >= 70:
        if perf >= 0.8:
            return "💎 Retain Star: Salary Correction + Career Talk"
        elif sat < 0.4:
            return "⚠️ Burnout: 1:1 Coaching + Workload Review"
        else:
            return "🔄 Succession Plan: Prepare Backup"
    elif risk >= 40:
        return "👀 Monitor: Quarterly Check-in"
    else:
        return "✅ Safe: Maintain Engagement"

# Apply this to your View DataFrame before displaying Tab 1
if "AttritionRisk" in view_df.columns:
    view_df["AI Recommendation"] = view_df.apply(recommend_action, axis=1)

# Now include "AI Recommendation" in your st.dataframe() call in Tab 1
def train_and_save():
    print("⏳ Loading data...")
    # Load and clean data
    if not os.path.exists(EMPLOYEE_FILE):
        print(f"❌ Error: {EMPLOYEE_FILE} not found.")
        return

    df = pd.read_csv(EMPLOYEE_FILE)
    
    # --- IMPORTANT: Fix the typo here too ---
    if 'average_montly_hours' in df.columns:
        df = df.rename(columns={'average_montly_hours': 'average_monthly_hours'})
    
    df = df.drop_duplicates(subset=["EmployeeID"], keep="last")

    # 1. Encoders
    le_dept = LabelEncoder()
    le_salary = LabelEncoder()
    
    if 'department' in df.columns:
        df['dept_encoded'] = le_dept.fit_transform(df['department'].astype(str))
    if 'salary' in df.columns:
        df['salary_encoded'] = le_salary.fit_transform(df['salary'].astype(str))

    # 2. Features
    feature_cols = [
        'satisfaction', 'evaluation', 'number_of_projects', 
        'average_monthly_hours', 'time_spend_company', 
        'work_accident', 'promotion', 'dept_encoded', 'salary_encoded'
    ]
    features_present = [c for c in feature_cols if c in df.columns]
    
    X = df[features_present]
    y = df['churn']

    # 3. Train
    print("🧠 Training model...")
    model = LogisticRegression(max_iter=3000, solver='lbfgs')
    model.fit(X, y)

    # 4. Save Everything (Model + Encoders + Feature List)
    # We bundle everything into a dictionary so we don't lose track of encoders
    artifacts = {
        "model": model,
        "features": features_present,
        "le_dept": le_dept,
        "le_salary": le_salary
    }
    
    joblib.dump(artifacts, MODEL_FILE)
    print(f"✅ Success! Model saved to '{MODEL_FILE}'")

if __name__ == "__main__":
    import os
    train_and_save()