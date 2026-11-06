# IPL Win Probability Predictor

An IPL second-innings win probability prediction system using Machine Learning and an interactive Streamlit dashboard.

## Features

- Predicts winning probabilities for both teams.
- Uses an XGBoost classification model.
- Considers runs remaining, balls remaining, wickets remaining, run rate and match pressure.
- Includes an interactive Streamlit dashboard.
- Supports IPL teams and match situations.

## Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- Category Encoders
- Streamlit

## Project Structure

    IPL-Win-Probability-Predictor/
    +-- app.py
    +-- ipl.py
    +-- pipe.pkl
    +-- requirements.txt
    +-- README.md
    +-- .gitignore

## Installation

Install dependencies:

    pip install -r requirements.txt

## Run the Application

    python -m streamlit run app.py

## Train the Model

Place matches.csv and deliveries.csv in the project folder, then run:

    python ipl.py

The training script generates pipe.pkl.

## Disclaimer

Predictions are machine learning estimates and do not guarantee actual match outcomes.
