import streamlit as st
import numpy as np
import pickle
import pandas as pd

# -----------------------------
# Load trained model and scaler
# -----------------------------
model_path = "best_tree_model.pkl"
scaler_path = "scaler.pkl"

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

# -----------------------------
# Load college database Excel
# -----------------------------
college_df = pd.read_excel("colleges_list.xlsx")

# Ensure GM column is numeric (invalid entries -> 0)
college_df["GM"] = pd.to_numeric(college_df["GM"], errors="coerce").fillna(0).astype(int)

# -----------------------------
# Total students per year (hardcoded)
# -----------------------------
total_students_per_year = {
    2020: 120000,
    2021: 125000,
    2022: 130000,
    2023: 244000,
    2024: 310000,
    2025: 312000,
    2026: 318000
}

# -----------------------------
# Streamlit Page Config and Header
# -----------------------------
st.set_page_config(
    page_title="KCET Rank Predictor",
    page_icon="🛸",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <div style="background-color:#0f111a;padding:15px;border-radius:10px">
    <h1 style="color:#00ffea;text-align:center;font-family:Orbitron, sans-serif;">
        🚀 KCET Rank Predictor & College Recommender 3000
    </h1>
    <p style="color:#00ffea;text-align:center;font-family:Orbitron, sans-serif;">
        Enter your scores below to see your predicted rank and potential colleges
    </p>
    </div>
    """, unsafe_allow_html=True
)

st.markdown("---")

# -----------------------------
# List of general branches for selection
# -----------------------------
general_courses = [
    "Computer Science & Engineering (CSE)",
    "Information Science & Engineering (ISE)",
    "Artificial Intelligence & Machine Learning (AIML)",
    "Electronics & Communication Engineering (ECE)",
    "Electrical & Electronics Engineering (EEE)",
    "Mechanical Engineering (ME)",
    "Civil Engineering",
    "Aerospace Engineering",
    "Biotechnology / Bio-Technology",
    "Industrial Engineering / Industrial Management",
    "Chemical Engineering",
    "Automobile / Automotive Engineering",
    "Computer Science & Information Technology (CS & IT)",
    "Data Science / Computer Science – Data Science",
    "Cyber Security / Information Security",
    "IoT / Internet of Things",
    "VLSI (Very-Large-Scale Integration) / VLSI Design",
    "Robotics & Automation",
    "Mechatronics",
    "Artificial Intelligence & Data Science",
    "ECE + specialization (e.g. Communications)",
    "Electronics / Electronics Engineering",
    "Instrumentation & Control",
    "Environmental Engineering",
    "Agricultural Engineering",
    "Mining Engineering",
    "Petroleum Engineering"
]

# -----------------------------
# Utility to normalize course name keywords
# -----------------------------
def normalize_keywords(course_name):
    words = course_name.lower().replace("&", " ").replace("/", " ").split()
    keywords = [w for w in words if w not in ("engineering", "–")]
    return set(keywords)

# -----------------------------
# Initialize session state variables
# -----------------------------
if 'predicted_rank' not in st.session_state:
    st.session_state.predicted_rank = None

if 'eligible_colleges' not in st.session_state:
    st.session_state.eligible_colleges = None

# -----------------------------
# User Input Form for marks and year
# -----------------------------
with st.form("score_form"):
    st.markdown("### Input Your Marks")

    kcet_marks = st.number_input("KCET Marks (out of 180)", min_value=0, max_value=180, step=1)
    board_marks = st.number_input("Board Marks (out of 300)", min_value=0, max_value=300, step=1)
    year = st.selectbox("Exam Year", options=list(total_students_per_year.keys()))

    submitted = st.form_submit_button("Predict Rank 🚀")

# -----------------------------
# Process rank prediction on form submit
# -----------------------------
if submitted:
    # Calculate percentages
    kcet_percent = (kcet_marks / 180) * 100
    board_percent = (board_marks / 300) * 100
    score = (kcet_percent + board_percent) / 2.0

    total_students = total_students_per_year[year]

    # Prepare features and predict
    X_input = np.array([[score, year, total_students]], dtype=float)
    X_input_scaled = scaler.transform(X_input)
    predicted_rank = model.predict(X_input_scaled)[0]
    int_predicted_rank = int(predicted_rank)

    # Store in session state
    st.session_state.predicted_rank = int_predicted_rank
    st.session_state.eligible_colleges = college_df[college_df['GM'] >= int_predicted_rank].sort_values('GM').reset_index(drop=True)

    # Display Calculated Scores & Predicted Rank
    st.markdown(
        f"""
        <div style="background-color:#0f111a;padding:20px;border-radius:10px;margin-top:15px">
        <h2 style="color:#ff3c00;font-family:Orbitron, sans-serif;text-align:center;">
            🎯 Calculated Scores
        </h2>
        <p style="color:#00ffea;text-align:center;font-family:Orbitron, sans-serif;">
            KCET %: {kcet_percent:.2f}%<br>
            Board %: {board_percent:.2f}%<br>
            Weighted Score: {score:.2f}%
        </p>
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style="background-color:#0f111a;padding:20px;border-radius:10px;margin-top:15px">
        <h2 style="color:#ff3c00;font-family:Orbitron, sans-serif;text-align:center;">
            🛸 Predicted Rank: {int_predicted_rank}
        </h2>
        <p style="color:#00ffea;text-align:center;font-family:Orbitron, sans-serif;">
            Based on exam year: {year} and total students: {total_students}
        </p>
        </div>
        """, unsafe_allow_html=True
    )

# -----------------------------
# If rank predicted, show branch selection and search button
# -----------------------------
if st.session_state.predicted_rank is not None:
    selected_branches = st.multiselect("Select One or More Desired Branches", general_courses)

    if st.button("Search Colleges"):
        if selected_branches:
            combined_keywords = set()
            for branch in selected_branches:
                combined_keywords.update(normalize_keywords(branch))

            def matches_any_branch(college_course):
                college_keywords = normalize_keywords(college_course)
                return bool(combined_keywords.intersection(college_keywords))

            filtered_colleges_multi = st.session_state.eligible_colleges[
                st.session_state.eligible_colleges['Course Name'].apply(matches_any_branch)
            ]

            st.markdown(
                f"""
                <div style="background-color:#0f111a;padding:20px;border-radius:10px;margin-top:20px">
                <h2 style="color:#ff3c00;font-family:Orbitron, sans-serif;text-align:center;">
                    🎓 Colleges You Can Get Into (GM ≥ {st.session_state.predicted_rank}) - Filtered by Selected Branches
                </h2>
                </div>
                """, unsafe_allow_html=True
            )

            if filtered_colleges_multi.empty:
                st.warning("No colleges found matching your rank and selected branches.")
            else:
                styled_filtered_multi = filtered_colleges_multi.style.set_table_styles(
                    [
                        {'selector':'th', 'props':[('background-color','#ff3c00'), ('color','white'), ('font-family','Orbitron, sans-serif')]},
                        {'selector':'td', 'props':[('color','#00ffea'), ('font-family','Orbitron, sans-serif')]}
                    ]
                )
                st.dataframe(styled_filtered_multi, height=400)
        else:
            st.info("Please select at least one branch before searching.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    "<p style='color:#00ffea;text-align:center;font-family:Orbitron, sans-serif;'>Made with 💫 by Your Team</p>",
    unsafe_allow_html=True
)
