import streamlit as st
import numpy as np
import pickle
import pandas as pd
import re 

# Define the columns the code expects to work with
EXPECTED_COLS = {
    'college name': 'College Name',
    'course name': 'Course Name',
    'gm': 'GM'
}
REQUIRED_COLS_LIST = list(EXPECTED_COLS.values())

# -----------------------------
# Load trained model and scaler
# -----------------------------
model_path = "best_tree_model.pkl"
scaler_path = "scaler.pkl"

try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("🚨 **Error:** Model or Scaler files not found. Please ensure 'best_tree_model.pkl' and 'scaler.pkl' are in the directory.")
    st.stop()

# -----------------------------
# Load college database Excel (FIX: Aggressively Standardize Column Names)
# -----------------------------
try:
    college_df = pd.read_excel("colleges_list.xlsx")
    
    # --- FIX 1: Aggressive Column Name Standardization to prevent KeyError ---
    column_mapping = {}
    current_cols_clean = {col.strip().lower(): col for col in college_df.columns}
    
    for expected_key, standard_name in EXPECTED_COLS.items():
        if expected_key in current_cols_clean:
            column_mapping[current_cols_clean[expected_key]] = standard_name
            
    if column_mapping:
        college_df.rename(columns=column_mapping, inplace=True)

    # Final check for missing columns after standardization
    missing_after_rename = [col for col in REQUIRED_COLS_LIST if col not in college_df.columns]
    
    if missing_after_rename:
        st.error(f"FATAL ERROR: The college data is missing the required columns: **{', '.join(missing_after_rename)}**. Please check your Excel headers to ensure they contain 'College Name', 'Course Name', and 'GM'.")
        st.stop()
    # ---------------------------------------------------------------------

except FileNotFoundError:
    st.error("🚨 **Error:** 'colleges_list.xlsx' file not found. Please ensure the file is in the directory.")
    st.stop()


# Ensure GM column is numeric (invalid entries -> 0)
college_df["GM"] = pd.to_numeric(college_df["GM"], errors="coerce").fillna(0).astype(int)

# -----------------------------
# Total students per year (hardcoded)
# -----------------------------
total_students_per_year = {
    2020: 120000, 2021: 125000, 2022: 130000, 2023: 244000, 2024: 310000, 2025: 312000, 2026: 318000
}

# -----------------------------
# Streamlit Page Config and Custom CSS (Aesthetics++)
# -----------------------------
st.set_page_config(
    page_title="KCET Rank Predictor",
    page_icon="🛸",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Main app background - Deep space theme */
    .stApp {
        background-color: #0c0e15; 
    }
    /* Custom container/card styling - Hover effect added */
    .stCard {
        background-color: #1a1e27; 
        border: 2px solid #3d4554;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 255, 234, 0.1); 
        padding: 25px;
        margin-bottom: 25px;
        transition: box-shadow 0.3s ease-in-out;
    }
    .stCard:hover {
        box-shadow: 0 4px 15px rgba(0, 255, 234, 0.3);
    }
    /* Streamlit input widgets - Matching theme */
    .stNumberInput, .stSelectbox, .stMultiSelect {
        color: #e6f0ff; 
        background-color: #262b33; 
        border-radius: 8px;
        border: 1px solid #4a5463;
    }
    /* Form submit button styling - Neon Primary */
    .stButton>button {
        background-color: #00ffea; 
        color: #0c0e15; 
        border-radius: 10px;
        font-weight: bold;
        transition: all 0.3s;
        border: none;
        padding: 10px 20px;
        font-size: 1.1em;
    }
    .stButton>button:hover {
        background-color: #33ffff; 
        box-shadow: 0 0 15px #00ffea;
    }
    /* Results Highlight Card Styling */
    .result-card {
        background-color: #1a1e27; 
        padding: 15px;
        border-radius: 10px;
        margin-top: 15px;
        text-align: center;
        box-shadow: 0 0 8px rgba(255, 60, 0, 0.5);
    }
    /* Headers and Text */
    h3 {
        color: #ff3c00; 
        font-family: 'Orbitron', sans-serif;
    }
    /* Dataframe Styling */
    .dataframe th {
        background-color: #ff3c00 !important;
        color: white !important;
        font-family: 'Orbitron', sans-serif !important;
        padding: 10px 5px !important;
    }
    .dataframe td {
        color: #00ffea !important;
        font-family: sans-serif !important;
        border-bottom: 1px solid #3d4554 !important;
        padding: 8px 5px !important;
    }
    </style>
    """, unsafe_allow_html=True
)


st.markdown(
    """
    <div style="background-color:#1a1e27;padding:25px;border-radius:15px;border: 3px solid #00ffea; box-shadow: 0 0 20px #00ffea, 0 0 8px #ff3c00;">
    <h1 style="color:#00ffea;text-align:center;font-family:Orbitron, sans-serif; margin-top:0; margin-bottom:5px;">
        🚀 KCET Rank Predictor & College Recommender 3000
    </h1>
    <p style="color:#e6f0ff;text-align:center;font-family:sans-serif;font-size:1.1em;">
        Enter your scores below to chart your course to college!
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
# Utility to normalize course name keywords (FIX 2: Improved Logic for Abbreviation Matching)
# -----------------------------
def normalize_keywords(course_name):
    if pd.isna(course_name): return set()
    cleaned = str(course_name).lower().replace("&", " ").replace("/", " ").replace("-", " ")
    
    # Original logic (simple word split)
    words = cleaned.split()
    keywords = [w for w in words if w not in ("engineering", "–", "technology", "of", "and", "the", "a")]
    
    # ADDED robust abbreviation matching to enhance original logic
    acronyms = set()
    if 'computer' in cleaned or 'cs' in cleaned or 'comp' in cleaned:
        acronyms.add('cs'); acronyms.add('cse')
    if 'information' in cleaned or 'is' in cleaned:
        acronyms.add('is'); acronyms.add('ise')
    if 'artificial' in cleaned or 'ai' in cleaned or 'ml' in cleaned:
        acronyms.add('ai'); acronyms.add('ml'); acronyms.add('aiml')
    if 'electronics' in cleaned:
        acronyms.add('ec'); acronyms.add('ece')
    if 'electrical' in cleaned:
        acronyms.add('ee'); acronyms.add('eee')
    if 'mechanical' in cleaned:
        acronyms.add('me'); acronyms.add('mech')
    if 'data' in cleaned:
        acronyms.add('ds')
    
    return set(keywords) | acronyms


# -----------------------------
# Initialize session state variables
# -----------------------------
if 'predicted_rank' not in st.session_state:
    st.session_state.predicted_rank = None

if 'eligible_colleges' not in st.session_state:
    st.session_state.eligible_colleges = None

# -----------------------------
# User Input Form for marks and year (Aesthetic Grouping)
# -----------------------------
with st.container():
    st.markdown('<div class="stCard">', unsafe_allow_html=True) 
    with st.form("score_form"):
        st.markdown("### 🔢 **Input Your Marks**")

        col1, col2 = st.columns(2)
        with col1:
            kcet_marks = st.number_input("KCET Marks (out of 180)", min_value=0, max_value=180, step=1, key="kcet_input")
        with col2:
            board_marks = st.number_input("Board Marks (out of 300)", min_value=0, max_value=300, step=1, key="board_input")

        year = st.selectbox("Exam Year", options=list(total_students_per_year.keys()), key="year_select")

        st.markdown("---") 
        submitted = st.form_submit_button("Predict Rank 🚀")
    st.markdown('</div>', unsafe_allow_html=True) 

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
    # Filtering uses the 'GM' column, which is now guaranteed to exist
    st.session_state.eligible_colleges = college_df[college_df['GM'] >= int_predicted_rank].sort_values('GM').reset_index(drop=True)

    # Display Calculated Scores & Predicted Rank (Aesthetic Grouping)
    st.markdown("---") 
    
    col_score, col_rank = st.columns(2)

    with col_score:
        st.markdown(
            f"""
            <div class="result-card" style="border-left: 5px solid #ff3c00;">
            <h4 style="color:#ff3c00;font-family:Orbitron, sans-serif;text-align:center;margin-top:0;">
                🎯 Calculated Scores
            </h4>
            <p style="color:#e6f0ff;text-align:left;font-family:sans-serif;margin-bottom:5px; padding-left:10px;">
                <strong style='color:#00ffea;'>KCET %:</strong> {kcet_percent:.2f}%<br>
                <strong style='color:#00ffea;'>Board %:</strong> {board_percent:.2f}%<br>
                <strong style='color:#00ffea;'>Weighted Score:</strong> {score:.2f}%
            </p>
            </div>
            """, unsafe_allow_html=True
        )

    with col_rank:
        st.markdown(
            f"""
            <div class="result-card" style="border-left: 5px solid #00ffea;">
            <h4 style="color:#00ffea;font-family:Orbitron, sans-serif;text-align:center;margin-top:0;">
                🛸 Predicted Rank
            </h4>
            <p style="color:#e6f0ff;text-align:center;font-family:sans-serif;margin-bottom:0;">
                <span style="font-size: 2.5em; font-weight: bold; color: #ff3c00;">{int_predicted_rank:,}</span>
            </p>
            <p style="color:#e6f0ff;text-align:center;font-family:sans-serif;font-size:0.8em; margin-top:5px;">
                (Based on {year} data)
            </p>
            </div>
            """, unsafe_allow_html=True
        )

# -----------------------------
# If rank predicted, show branch selection and search button
# -----------------------------
if st.session_state.predicted_rank is not None:
    st.markdown("---")
    st.markdown('<div class="stCard">', unsafe_allow_html=True) 
    st.markdown("### 🔍 **Filter Colleges by Branch**")
    
    selected_branches = st.multiselect(
        "Select One or More Desired Branches (Improved keyword matching applied)", 
        general_courses,
        key="branch_select"
    )

    if st.button("Search Colleges", key="search_button_main"):
        if selected_branches:
            combined_keywords = set()
            for branch in selected_branches:
                combined_keywords.update(normalize_keywords(branch))

            def matches_any_branch(college_course):
                # Ensure it handles potential NaNs and uses the improved normalizer
                if pd.isna(college_course): return False 
                college_keywords = normalize_keywords(college_course)
                return bool(combined_keywords.intersection(college_keywords))

            filtered_colleges_multi = st.session_state.eligible_colleges[
                st.session_state.eligible_colleges['Course Name'].apply(matches_any_branch)
            ]
            
            # --- Display College Results Header (FIXED & IMPROVED) ---
            formatted_rank = f"{st.session_state.predicted_rank:,}"
            num_colleges = len(filtered_colleges_multi)
            
            st.markdown(
                f"""
                <div style="background-color:#1a1e27;padding:15px;border-radius:10px;margin-top:20px; border: 1px solid #ff3c00;">
                <h3 style="color:#00ffea;text-align:center;font-family:Orbitron, sans-serif; margin: 0;">
                    🎓 Recommended Colleges (GM $\\ge$ {formatted_rank})
                </h3>
                </div>
                """, unsafe_allow_html=True
            )
            # ---------------------------------------------------------


            if filtered_colleges_multi.empty:
                st.warning("No colleges found matching your rank and selected branches. Try selecting fewer branches or checking the course names in your Excel file.")
            else:
                st.success(f"🥳 Found **{num_colleges}** College/Course combinations matching your rank and branch preferences!")
                
                # Use the standardized column names which are guaranteed to exist now
                display_df = filtered_colleges_multi[REQUIRED_COLS_LIST].rename(columns={'GM': 'GM Cutoff (Max Rank)'})
                
                styled_filtered_multi = display_df.style.set_table_styles(
                    [
                        {'selector':'th', 'props':[('background-color','#ff3c00'), ('color','white'), ('font-family','Orbitron, sans-serif')]},
                        {'selector':'td', 'props':[('color','#00ffea'), ('font-family','sans-serif'), ('border-bottom', '1px solid #3d4554')]}
                    ]
                ).hide(axis="index")
                
                st.dataframe(styled_filtered_multi, use_container_width=True, height=400)
        else:
            st.info("Please select at least one branch before searching.")
            
    st.markdown('</div>', unsafe_allow_html=True) 

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    "<p style='color:#9eabbb;text-align:center;font-family:sans-serif;'>Made with 💫 by Your Team</p>",
    unsafe_allow_html=True
)