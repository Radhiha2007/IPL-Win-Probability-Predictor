import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from category_encoders import TargetEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
MATCH_FILE = BASE_DIR / "matches.csv"
DELIVERY_FILE = BASE_DIR / "deliveries.csv"
MODEL_FILE = BASE_DIR / "pipe.pkl"

TEAMS = [
    "Sunrisers Hyderabad",
    "Mumbai Indians",
    "Royal Challengers Bengaluru",
    "Kolkata Knight Riders",
    "Punjab Kings",
    "Chennai Super Kings",
    "Rajasthan Royals",
    "Delhi Capitals",
    "Gujarat Titans",
    "Lucknow Super Giants",
]

NAME_CHANGES = {
    "Delhi Daredevils": "Delhi Capitals",
    "Deccan Chargers": "Sunrisers Hyderabad",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
}

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


def calculate_features(
    batting_team,
    bowling_team,
    city,
    target,
    score,
    overs_completed,
    balls_in_current_over,
    wickets_fallen,
):
    """Create the same 15 model features for training and live prediction."""

    balls_bowled = overs_completed * 6 + balls_in_current_over
    balls_left = 120 - balls_bowled
    runs_left = target - score
    wickets_left = 10 - wickets_fallen

    crr = (score * 6 / balls_bowled) if balls_bowled > 0 else 0.0
    rrr = (runs_left * 6 / balls_left) if balls_left > 0 else 0.0

    pressure = runs_left / (balls_left + 1)
    wicket_pressure = wickets_left / 10
    rrr_crr_diff = rrr - crr
    balls_per_wicket = balls_left / (wickets_left + 1)
    required_run_pressure = runs_left / (balls_left * wickets_left + 1)
    momentum = crr - rrr

    return {
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


def load_and_prepare_data():
    if not MATCH_FILE.exists() or not DELIVERY_FILE.exists():
        raise FileNotFoundError(
            "matches.csv and deliveries.csv must be in the same folder as ipl.py."
        )

    match = pd.read_csv(MATCH_FILE)
    delivery = pd.read_csv(DELIVERY_FILE)

    required_match_columns = {"id", "team1", "team2", "winner"}
    required_delivery_columns = {
        "match_id", "inning", "over", "ball", "total_runs",
        "batting_team", "bowling_team", "is_wicket",
    }

    missing_match = required_match_columns - set(match.columns)
    missing_delivery = required_delivery_columns - set(delivery.columns)

    if missing_match:
        raise ValueError(f"matches.csv is missing columns: {sorted(missing_match)}")
    if missing_delivery:
        raise ValueError(
            f"deliveries.csv is missing columns: {sorted(missing_delivery)}"
        )
    if "city" not in match.columns:
        raise ValueError("matches.csv must contain a 'city' column.")

    match.replace(NAME_CHANGES, inplace=True)
    delivery.replace(NAME_CHANGES, inplace=True)

    # Keep completed matches involving the current IPL team set.
    match = match[
        match["team1"].isin(TEAMS)
        & match["team2"].isin(TEAMS)
        & match["winner"].isin(TEAMS)
    ].copy()

    # DLS matches are excluded because this project derives the target
    # from the first-innings total.
    if "dl_applied" in match.columns:
        match = match[match["dl_applied"] == 0].copy()

    first_innings_scores = (
        delivery[delivery["inning"] == 1]
        .groupby("match_id")["total_runs"]
        .sum()
        .reset_index(name="first_innings_total")
    )
    first_innings_scores["target"] = first_innings_scores["first_innings_total"] + 1

    match_df = match.merge(
        first_innings_scores[["match_id", "target"]],
        left_on="id",
        right_on="match_id",
        how="inner",
    )

    delivery_df = match_df.merge(delivery, on="match_id", how="inner")
    delivery_df = delivery_df[delivery_df["inning"] == 2].copy()
    delivery_df.sort_values(["match_id", "over", "ball"], inplace=True)

    # Current score and match-state features after each delivery.
    delivery_df["current_score"] = (
        delivery_df.groupby("match_id")["total_runs"].cumsum()
    )
    delivery_df["runs_left"] = (
        delivery_df["target"] - delivery_df["current_score"]
    )

    delivery_df["balls_left"] = (
        120 - (delivery_df["over"] * 6 + delivery_df["ball"])
    ).clip(lower=1)

    delivery_df["wickets_left"] = (
        10 - delivery_df.groupby("match_id")["is_wicket"].cumsum()
    )

    total_balls_bowled = 120 - delivery_df["balls_left"]

    delivery_df["crr"] = np.where(
        total_balls_bowled > 0,
        delivery_df["current_score"] * 6 / total_balls_bowled,
        0,
    )
    delivery_df["rrr"] = (
        delivery_df["runs_left"] * 6 / delivery_df["balls_left"]
    )

    # Advanced features.
    delivery_df["pressure"] = (
        delivery_df["runs_left"] / (delivery_df["balls_left"] + 1)
    )
    delivery_df["wicket_pressure"] = delivery_df["wickets_left"] / 10
    delivery_df["rrr_crr_diff"] = delivery_df["rrr"] - delivery_df["crr"]
    delivery_df["balls_per_wicket"] = (
        delivery_df["balls_left"] / (delivery_df["wickets_left"] + 1)
    )
    delivery_df["required_run_pressure"] = (
        delivery_df["runs_left"]
        / (delivery_df["balls_left"] * delivery_df["wickets_left"] + 1)
    )
    delivery_df["momentum"] = delivery_df["crr"] - delivery_df["rrr"]

    delivery_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Label 1 means the team batting in the second innings eventually won.
    delivery_df["result"] = (
        delivery_df["batting_team"] == delivery_df["winner"]
    ).astype(int)

    final_df = delivery_df[
        FEATURE_COLUMNS + ["result", "match_id"]
    ].copy()

    final_df.dropna(inplace=True)

    # Exclude rows after the target has already been reached or the
    # batting side has been dismissed.
    final_df = final_df[
        (final_df["balls_left"] > 0)
        & (final_df["runs_left"] > 0)
        & (final_df["wickets_left"] > 0)
    ].copy()

    if final_df.empty:
        raise ValueError("No valid second-innings training rows were created.")
    if final_df["result"].nunique() != 2:
        raise ValueError("Training data must contain both win and loss examples.")

    return final_df


def create_pipeline():
    categorical_columns = ["batting_team", "bowling_team", "city"]

    transformer = ColumnTransformer(
        transformers=[
            ("cat", TargetEncoder(), categorical_columns),
        ],
        remainder="passthrough",
    )

    classifier = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=6,
        reg_alpha=1.0,
        reg_lambda=3.0,
        gamma=0.5,
        tree_method="hist",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline([
        ("prep", transformer),
        ("model", classifier),
    ])


def train_and_save_model():
    print("Loading IPL datasets...")
    final_df = load_and_prepare_data()
    print(f"Prepared {len(final_df):,} valid second-innings delivery records.")

    X = final_df[FEATURE_COLUMNS]
    y = final_df["result"]
    groups = final_df["match_id"]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42,
    )
    train_indices, test_indices = next(splitter.split(X, y, groups=groups))

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    print("\nTraining XGBoost model...")
    evaluation_pipeline = create_pipeline()
    evaluation_pipeline.fit(X_train, y_train)

    predictions = evaluation_pipeline.predict(X_test)
    probabilities = evaluation_pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    loss = log_loss(y_test, probabilities, labels=[0, 1])
    auc = roc_auc_score(y_test, probabilities)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    print("\n" + "=" * 55)
    print("             XGBOOST MODEL RESULTS")
    print("=" * 55)
    print(f"Accuracy   : {accuracy * 100:.2f}%")
    print(f"Log Loss   : {loss:.4f}")
    print(f"ROC-AUC    : {auc:.4f}")
    print(f"Precision  : {precision:.4f}")
    print(f"Recall     : {recall:.4f}")
    print(f"F1 Score   : {f1:.4f}")
    print("=" * 55)

    # Refit a fresh pipeline on all available records for deployment.
    # The metrics above remain based on the held-out match groups.
    print("\nTraining final deployment model on all valid records...")
    final_pipeline = create_pipeline()
    final_pipeline.fit(X, y)

    with MODEL_FILE.open("wb") as model_file:
        pickle.dump(final_pipeline, model_file)

    print(f"Final model saved to: {MODEL_FILE}")
    return final_pipeline


def normalize_team(team_name):
    aliases = {
        "csk": "Chennai Super Kings",
        "mi": "Mumbai Indians",
        "rcb": "Royal Challengers Bengaluru",
        "kkr": "Kolkata Knight Riders",
        "pbks": "Punjab Kings",
        "kxip": "Punjab Kings",
        "rr": "Rajasthan Royals",
        "dc": "Delhi Capitals",
        "srh": "Sunrisers Hyderabad",
        "gt": "Gujarat Titans",
        "lsg": "Lucknow Super Giants",
        "kings xi punjab": "Punjab Kings",
        "royal challengers bangalore": "Royal Challengers Bengaluru",
        "delhi daredevils": "Delhi Capitals",
        "deccan chargers": "Sunrisers Hyderabad",
    }
    cleaned = team_name.strip()
    return aliases.get(cleaned.lower(), cleaned)


def run_terminal_predictor(pipe):
    print("\n" + "=" * 55)
    print("             IPL WIN PREDICTOR")
    print("=" * 55)
    print("Available teams:")
    for team in TEAMS:
        print(f"- {team}")
    print("\nType 'exit' at any prompt to stop.")

    while True:
        try:
            batting_team_input = input("\nBatting team: ").strip()
            if batting_team_input.lower() == "exit":
                break

            bowling_team_input = input("Bowling team: ").strip()
            if bowling_team_input.lower() == "exit":
                break

            city = input("Match city: ").strip()
            if city.lower() == "exit":
                break

            target_text = input("Target score: ").strip()
            if target_text.lower() == "exit":
                break

            score_text = input("Current score: ").strip()
            if score_text.lower() == "exit":
                break

            overs_text = input("Overs completed (example: 10.2): ").strip()
            if overs_text.lower() == "exit":
                break

            wickets_text = input("Wickets fallen: ").strip()
            if wickets_text.lower() == "exit":
                break

            batting_team = normalize_team(batting_team_input)
            bowling_team = normalize_team(bowling_team_input)

            team_lookup = {team.lower(): team for team in TEAMS}
            batting_team = team_lookup.get(batting_team.lower(), batting_team)
            bowling_team = team_lookup.get(bowling_team.lower(), bowling_team)

            if batting_team not in TEAMS:
                raise ValueError("Invalid batting team.")
            if bowling_team not in TEAMS:
                raise ValueError("Invalid bowling team.")
            if batting_team == bowling_team:
                raise ValueError("Batting and bowling teams must be different.")

            target = int(target_text)
            score = int(score_text)
            wickets = int(wickets_text)

            if target <= 0 or score < 0:
                raise ValueError("Target must be positive and score cannot be negative.")
            if score >= target:
                raise ValueError("The target has already been reached.")
            if not 0 <= wickets <= 9:
                raise ValueError("Wickets fallen must be between 0 and 9.")
            if not city:
                raise ValueError("City cannot be empty.")

            # Cricket notation: 10.2 means 10 overs and 2 balls,
            # not 10.2 decimal overs.
            if "." in overs_text:
                parts = overs_text.split(".")
                if len(parts) != 2 or not all(part.isdigit() for part in parts):
                    raise ValueError("Enter overs in cricket notation, e.g. 10.2.")
                overs_completed = int(parts[0])
                balls_in_current_over = int(parts[1])
            else:
                if not overs_text.isdigit():
                    raise ValueError("Enter overs like 10 or 10.2.")
                overs_completed = int(overs_text)
                balls_in_current_over = 0

            if not 0 <= overs_completed <= 19:
                raise ValueError("Completed overs must be between 0 and 19.")
            if not 0 <= balls_in_current_over <= 5:
                raise ValueError("Balls in the current over must be between 0 and 5.")

            balls_bowled = overs_completed * 6 + balls_in_current_over
            if balls_bowled >= 120:
                raise ValueError("The 20-over innings is complete.")

            feature_values = calculate_features(
                batting_team=batting_team,
                bowling_team=bowling_team,
                city=city,
                target=target,
                score=score,
                overs_completed=overs_completed,
                balls_in_current_over=balls_in_current_over,
                wickets_fallen=wickets,
            )
            input_df = pd.DataFrame([feature_values], columns=FEATURE_COLUMNS)

            class_one_index = list(pipe.classes_).index(1)
            win_probability = pipe.predict_proba(input_df)[0][class_one_index]
            loss_probability = 1 - win_probability

            print("\n" + "-" * 55)
            print(f"Score        : {score}/{wickets}")
            print(f"Target       : {target}")
            print(f"Overs        : {overs_text}")
            print(f"Runs left    : {feature_values['runs_left']}")
            print(f"Balls left   : {feature_values['balls_left']}")
            print("-" * 55)
            print(f"{batting_team}: {win_probability * 100:.2f}%")
            print(f"{bowling_team}: {loss_probability * 100:.2f}%")
            print("-" * 55)

        except ValueError as error:
            print(f"\nInvalid input: {error}")
            print("Please enter the match details again.")
        except Exception as error:
            print(f"\nPrediction error: {error}")


if __name__ == "__main__":
    trained_pipe = train_and_save_model()
    run_terminal_predictor(trained_pipe)
