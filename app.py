import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "pipe.pkl"

TEAMS = [
    "Chennai Super Kings",
    "Delhi Capitals",
    "Gujarat Titans",
    "Kolkata Knight Riders",
    "Lucknow Super Giants",
    "Mumbai Indians",
    "Punjab Kings",
    "Rajasthan Royals",
    "Royal Challengers Bengaluru",
    "Sunrisers Hyderabad",
]

CITIES = [
    "Abu Dhabi", "Ahmedabad", "Bangalore", "Bloemfontein", "Cape Town",
    "Centurion", "Chandigarh", "Chennai", "Cuttack", "Delhi", "Dharamsala",
    "Dubai", "Durban", "East London", "Guwahati", "Hyderabad", "Indore",
    "Jaipur", "Johannesburg", "Kimberley", "Kolkata", "Lucknow", "Mumbai",
    "Nagpur", "Navi Mumbai", "Port Elizabeth", "Pune", "Raipur", "Ranchi",
    "Sharjah", "Visakhapatnam",
]

FEATURE_COLUMNS = [
    "batting_team",
    "bowling_team",
    "city",
    "runs_left",
    "balls_left",
    "wickets_left",
    "target",
    "crr",
    "rrr",
    "pressure",
    "wicket_pressure",
    "rrr_crr_diff",
    "balls_per_wicket",
    "required_run_pressure",
    "momentum",
]

st.set_page_config(
    page_title="IPL Win Predictor",
    page_icon="🏏",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            linear-gradient(rgba(0, 0, 0, 0.76), rgba(0, 0, 0, 0.80)),
            url("https://images.unsplash.com/photo-1531415074968-036ba1b575da?auto=format&fit=crop&w=2000&q=80");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    h1, h2, h3, p, label, [data-testid="stMarkdownContainer"] {
        color: white;
    }
    .stButton > button {
        background-color: #ff4b4b;
        color: white;
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-weight: bold;
        border: none;
    }
    .stButton > button:hover {
        background-color: #e33d3d;
        color: white;
        border: none;
    }
    .result-box {
        background-color: rgba(255, 255, 255, 0.12);
        padding: 22px;
        border-radius: 15px;
        border: 1px solid #ff4b4b;
        text-align: center;
        min-height: 145px;
    }
    .result-team {
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .result-percent {
        font-size: 42px;
        font-weight: 800;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "pipe.pkl was not found. Run `python ipl.py` first to train and save the model."
        )
    with MODEL_FILE.open("rb") as model_file:
        return pickle.load(model_file)


try:
    pipe = load_model()
except Exception as error:
    st.error(f"Could not load the prediction model: {error}")
    st.stop()


def create_prediction_features(
    batting_team,
    bowling_team,
    city,
    target,
    score,
    overs_completed,
    balls_in_current_over,
    wickets_fallen,
):
    balls_bowled = overs_completed * 6 + balls_in_current_over
    balls_left = 120 - balls_bowled
    runs_left = target - score
    wickets_left = 10 - wickets_fallen

    crr = score * 6 / balls_bowled if balls_bowled > 0 else 0.0
    rrr = runs_left * 6 / balls_left

    pressure = runs_left / (balls_left + 1)
    wicket_pressure = wickets_left / 10
    rrr_crr_diff = rrr - crr
    balls_per_wicket = balls_left / (wickets_left + 1)
    required_run_pressure = runs_left / (balls_left * wickets_left + 1)
    momentum = crr - rrr

    values = {
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "city": city,
        "runs_left": runs_left,
        "balls_left": balls_left,
        "wickets_left": wickets_left,
        "target": target,
        "crr": crr,
        "rrr": rrr,
        "pressure": pressure,
        "wicket_pressure": wicket_pressure,
        "rrr_crr_diff": rrr_crr_diff,
        "balls_per_wicket": balls_per_wicket,
        "required_run_pressure": required_run_pressure,
        "momentum": momentum,
    }
    return pd.DataFrame([values], columns=FEATURE_COLUMNS)


st.title("🏏 IPL Victory Probability Predictor")
st.markdown("### Enter the current second-innings match situation")

col1, col2, col3 = st.columns(3)

with col1:
    batting_team = st.selectbox("Select Batting Team", TEAMS)

with col2:
    bowling_options = [team for team in TEAMS if team != batting_team]
    bowling_team = st.selectbox("Select Bowling Team", bowling_options)

with col3:
    selected_city = st.selectbox("Select Host City", CITIES)

st.markdown("### Match Score")

col4, col5, col6 = st.columns(3)

with col4:
    target = st.number_input(
        "Target Score",
        min_value=1,
        max_value=500,
        value=180,
        step=1,
    )

with col5:
    score = st.number_input(
        "Current Score",
        min_value=0,
        max_value=500,
        value=75,
        step=1,
    )

with col6:
    wickets = st.number_input(
        "Wickets Fallen",
        min_value=0,
        max_value=10,
        value=2,
        step=1,
    )

col7, col8 = st.columns(2)

with col7:
    overs_completed = st.number_input(
        "Completed Overs",
        min_value=0,
        max_value=19,
        value=10,
        step=1,
        help="Enter only completed overs. For 10.2 overs, enter 10 here and 2 below.",
    )

with col8:
    balls_in_current_over = st.number_input(
        "Balls Completed in Current Over",
        min_value=0,
        max_value=5,
        value=2,
        step=1,
        help="For 10.2 overs, enter 2. A completed over has 0 balls in the current over.",
    )

st.caption(
    "Overs use cricket notation: 10 overs and 2 balls should be entered as "
    "Completed Overs = 10 and Balls Completed = 2."
)

if st.button("Predict Win Probability"):
    balls_bowled = int(overs_completed) * 6 + int(balls_in_current_over)
    balls_left = 120 - balls_bowled

    if batting_team == bowling_team:
        st.warning("Batting and bowling teams cannot be the same.")
    elif score >= target:
        st.success(f"{batting_team} has already reached the target!")
    elif wickets >= 10:
        st.warning("The batting team is all out. No live prediction is available.")
    elif balls_left <= 0:
        st.warning("The 20-over innings is complete.")
    else:
        input_df = create_prediction_features(
            batting_team=batting_team,
            bowling_team=bowling_team,
            city=selected_city,
            target=int(target),
            score=int(score),
            overs_completed=int(overs_completed),
            balls_in_current_over=int(balls_in_current_over),
            wickets_fallen=int(wickets),
        )

        try:
            class_one_index = list(pipe.classes_).index(1)
            probabilities = pipe.predict_proba(input_df)[0]
            batting_win_probability = float(probabilities[class_one_index])
            bowling_win_probability = 1.0 - batting_win_probability

            runs_left = int(target) - int(score)
            wickets_left = 10 - int(wickets)
            current_overs_text = (
                f"{int(overs_completed)}.{int(balls_in_current_over)}"
            )

            st.markdown("---")
            st.markdown("### Match Situation")

            info1, info2, info3, info4 = st.columns(4)
            info1.metric("Current Score", f"{int(score)}/{int(wickets)}")
            info2.metric("Target", int(target))
            info3.metric("Runs Required", runs_left)
            info4.metric("Balls Remaining", balls_left)

            st.caption(
                f"Overs: {current_overs_text} | "
                f"Wickets remaining: {wickets_left}"
            )

            st.markdown("### Predicted Win Probability")
            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.markdown(
                    f"""
                    <div class="result-box">
                        <div class="result-team">{batting_team}</div>
                        <div class="result-percent">{batting_win_probability * 100:.2f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with result_col2:
                st.markdown(
                    f"""
                    <div class="result-box">
                        <div class="result-team">{bowling_team}</div>
                        <div class="result-percent">{bowling_win_probability * 100:.2f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.progress(batting_win_probability)
            st.caption(
                "These percentages are model-estimated probabilities, "
                "not guarantees of the match result."
            )
            st.balloons()

        except Exception as error:
            st.error(f"Prediction failed: {error}")
