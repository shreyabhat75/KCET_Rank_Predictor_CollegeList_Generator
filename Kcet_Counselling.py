import streamlit as st
import numpy as np
import pickle
import pandas as pd

# -----------------------------
# Configuration and Setup
# -----------------------------
st.set_page_config(
    page_title="KCET Rank Predictor & College Finder",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Load trained model and scaler
# -----------------------------
@st.cache_resource
def load_model():
    try:
        with open("best_tree_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'best_tree_model.pkl' and 'scaler.pkl' are available.")
        return None, None

# -----------------------------
# Load college database
# -----------------------------
@st.cache_data
def load_college_data():
    try:
        df = pd.read_excel("colleges_list.xlsx")

        # Column mapping if needed, no website column needed here

        required_columns = ["College Code", "college name", "Course Name", "Location", "Type of College", "GM"]
        for col in required_columns:
            if col not in df.columns:
                if col == "Type of College":
                    df[col] = "Unknown"
                elif col == "GM":
                    df[col] = 99999
                else:
                    df[col] = "Not Available"

        if "GM" in df.columns:
            df["GM"] = pd.to_numeric(df["GM"], errors="coerce").fillna(99999).astype(int)

        return df

    except Exception as e:
        st.error(f"Error loading college data: {e}")
        return pd.DataFrame()

# -----------------------------
# Constants
# -----------------------------
TOTAL_STUDENTS_PER_YEAR = {
    2020: 120000,
    2021: 125000,
    2022: 130000,
    2023: 244000,
    2024: 310000,
    2025: 312000,
    2026: 318000,
}

KARNATAKA_DISTRICTS = [
    "Any Location",
    "Banglore",
    "Ramnagar",
    "Tumkur",
    "Hassan",
    "Mysore",
    "Mandya",
    "Bagalkot", "Ballari", "Belagavi",
    "Bidar", "Chamarajanagar", "Chikballapur", "Chikkamagaluru", "Chitradurga",
    "Dakshina Kannada", "Davanagere", "Dharwad", "Gadag", "Haveri",
    "Kalaburagi", "Kodagu", "Kolar", "Koppal", "Udupi", "Uttara Kannada",
    "Vijayapura", "Yadgir", "Raichur", "Shivamogga",
]

ENGINEERING_BRANCHES = [
    "Any Branch",
    "Computer Science & Engineering",
    "Information Science & Engineering",
    "Artificial Intelligence & Machine Learning",
    "Electronics & Communication Engineering",
    "Electrical & Electronics Engineering",
    "Mechanical Engineering",
    "Civil Engineering",
    "Aerospace Engineering",
    "Biotechnology",
    "Industrial Engineering",
    "Chemical Engineering",
    "Automobile Engineering",
    "Data Science",
    "Cyber Security",
    "Information Technology",
    "VLSI Design",
    "Robotics & Automation",
    "Mechatronics",
    "Environmental Engineering",
    "Agricultural Engineering",
    "Mining Engineering",
    "Petroleum Engineering",
]

UNIVERSITY_TYPES = [
    "Any Type",
    "VTU",
    "Auto",
    "University",
]

# -----------------------------
# Utility functions (case-insensitive matching)
# -----------------------------
def matches_branch(college_course, selected_branch):
    if selected_branch.lower() == "any branch" or not selected_branch:
        return True
    if pd.isna(college_course):
        return False
    college_course = college_course.lower()
    selected_branch = selected_branch.lower()
    branch_keywords = selected_branch.replace("&", " ").replace("/", " ").split()
    branch_keywords = [word for word in branch_keywords if len(word) > 2]
    return any(keyword in college_course for keyword in branch_keywords)

def matches_location(college_location, selected_location):
    if selected_location.lower() == "any location" or not selected_location:
        return True
    if pd.isna(college_location):
        return False
    college_location = str(college_location).lower()
    selected_location = selected_location.lower()
    return selected_location in college_location

def matches_type(college_type, selected_type):
    if selected_type.lower() == "any type" or not selected_type:
        return True
    if pd.isna(college_type):
        return False
    college_type = str(college_type).strip().lower()
    selected_type = selected_type.lower()
    return selected_type == college_type

# -----------------------------
# Load data
# -----------------------------
model, scaler = load_model()
college_df = load_college_data()

# -----------------------------
# UI Header
# -----------------------------
st.markdown(
    """
    <div style="background-color:#0f111a;padding:20px;border-radius:10px;margin-bottom:20px">
    <h1 style="color:#00ffea;text-align:center;font-family:'Segoe UI', sans-serif;">
        🚀 KCET Rank Predictor & College Finder
    </h1>
    <p style="color:#00ffea;text-align:center;font-family:'Segoe UI', sans-serif;">
        Predict your rank and discover the best colleges for you
    </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Initialize session state
# -----------------------------
if "predicted_rank" not in st.session_state:
    st.session_state.predicted_rank = None
if "show_college_search" not in st.session_state:
    st.session_state.show_college_search = False

# -----------------------------
# Main UI layout
# -----------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📊 Step 1: Predict Your Rank")

    if model and scaler:
        with st.form("rank_prediction_form"):
            kcet_marks = st.number_input(
                "KCET Marks (out of 180)", min_value=0, max_value=180, value=100, step=1
            )
            board_marks = st.number_input(
                "Board Marks (out of 300)", min_value=0, max_value=300, value=200, step=1
            )
            year = st.selectbox(
                "Exam Year", list(TOTAL_STUDENTS_PER_YEAR.keys()), index=len(TOTAL_STUDENTS_PER_YEAR) - 3
            )
            predict_btn = st.form_submit_button("🎯 Predict My Rank", use_container_width=True)

        if predict_btn:
            kcet_percent = (kcet_marks / 180) * 100
            board_percent = (board_marks / 300) * 100
            score = (kcet_percent + board_percent) / 2.0
            total_students = TOTAL_STUDENTS_PER_YEAR[year]
            X_input = np.array([[score, year, total_students]], dtype=float)
            X_input_scaled = scaler.transform(X_input)
            predicted_rank = model.predict(X_input_scaled)[0]
            st.session_state.predicted_rank = int(predicted_rank)
            st.session_state.show_college_search = True
            st.success(f"🎯 Your Predicted Rank: *{int(predicted_rank)}*")
            st.info(
                f"📈 Weighted Score: {score:.2f}% (KCET: {kcet_percent:.1f}%, Board: {board_percent:.1f}%)"
            )
    else:
        st.error("Model not available. Please check model files.")

with col2:
    st.markdown("### 🏫 Step 2: Find Your Colleges")

    if st.session_state.predicted_rank:
        st.success(f"Using Predicted Rank: *{st.session_state.predicted_rank}*")
        with st.form("college_search_form"):
            selected_branch = st.selectbox("🎓 Preferred Branch:", ENGINEERING_BRANCHES)
            selected_location = st.selectbox("📍 Preferred District:", KARNATAKA_DISTRICTS)
            selected_type = st.selectbox("🏛 University Type:", UNIVERSITY_TYPES)
            search_btn = st.form_submit_button("🔍 Find My Colleges", use_container_width=True)

        if search_btn:
            if college_df.empty:
                st.error("College database not available. Please check your data file.")
            else:
                st.markdown("#### 🔍 Searching with your preferences:")
                st.info(
                    f"*Branch:* {selected_branch}  \n*District:* {selected_location}  \n*Type:* {selected_type}  \n*Rank:* {st.session_state.predicted_rank}"
                )

                if "GM" in college_df.columns:
                    eligible_colleges = college_df[college_df["GM"] >= st.session_state.predicted_rank].copy()
                else:
                    st.warning("GM column not found. Showing all colleges.")
                    eligible_colleges = college_df.copy()

                if selected_branch.lower() != "any branch" and "Course Name" in eligible_colleges.columns:
                    mask = eligible_colleges["Course Name"].apply(lambda x: matches_branch(x, selected_branch))
                    eligible_colleges = eligible_colleges[mask]

                if selected_location.lower() != "any location" and "Location" in eligible_colleges.columns:
                    mask = eligible_colleges["Location"].apply(lambda x: matches_location(x, selected_location))
                    eligible_colleges = eligible_colleges[mask]

                if selected_type.lower() != "any type" and "Type of College" in eligible_colleges.columns:
                    mask = eligible_colleges["Type of College"].apply(lambda x: matches_type(x, selected_type))
                    eligible_colleges = eligible_colleges[mask]

                if "GM" in eligible_colleges.columns:
                    eligible_colleges = eligible_colleges.sort_values("GM")

                st.markdown("---")

                if eligible_colleges.empty:
                    st.warning("❌ No colleges found matching your criteria.")
                    st.markdown("*Suggestions:*")
                    st.markdown("- Try selecting 'Any Branch', 'Any Location', or 'Any Type'")
                    st.markdown("- Check if your predicted rank is realistic")
                else:
                    st.success(f"🎉 Found *{len(eligible_colleges)}* colleges matching your preferences!")
                    desired_columns = [
                        "College Code",
                        "college name",
                        "Course Name",
                        "Location",
                        "Type of College",
                        "GM",
                    ]
                    available_columns = [col for col in desired_columns if col in eligible_colleges.columns]
                    display_df = eligible_colleges[available_columns].copy() if available_columns else eligible_colleges.copy()

                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "GM": st.column_config.NumberColumn("GM", format="%d"),
                        },
                    )
    else:
        st.info("👆 Please predict your rank first to search for colleges")
        st.markdown("*Once you predict your rank, you'll be able to:*")
        st.markdown("- 🎓 Select your preferred engineering branch")
        st.markdown("- 📍 Choose from all Karnataka districts")
        st.markdown("- 🏛 Filter by VTU, Auto, or University")

# -----------------------------
# Additional Information
# -----------------------------
st.markdown("---")
with st.expander("ℹ How This Works"):
    st.markdown(
        """
    *Rank Prediction:*
    - Uses machine learning model trained on historical KCET data
    - Considers your KCET marks, board marks, and exam year
    - Accounts for varying competition levels across years

    *College Matching:*
    - Shows colleges where your predicted rank meets the cutoff
    - Filters by your preferences for branch, location, and institution type

    *Tips for Best Results:*
    - Consider colleges with cutoffs near your predicted rank as competitive options
    - Don't forget to check individual college websites for latest information
    """
    )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#666;'>Made with ❤ for KCET aspirants | Data updated for 2025</p>",
    unsafe_allow_html=True,
)
