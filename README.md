# 🏏 IPL Win Probability Predictor

An interactive Machine Learning application that predicts the winning probability of the batting and bowling teams during the second innings of an Indian Premier League (IPL) cricket match.

The application uses an **XGBoost classification model** to analyze the current match situation and estimate each team's chances of winning. Predictions are displayed through an interactive **Streamlit dashboard**.

🔗 **Live Demo:** https://ipl-win-probability-predictor-010906.streamlit.app/

---

## 📌 Project Overview

Cricket matches can change rapidly with every over. This project uses historical IPL match and ball-by-ball delivery data to estimate the probability of a team's victory based on the current match situation.

Users can enter match details such as teams, venue, current score, wickets lost, and target. The application processes these inputs and displays the predicted win probability for both teams.

**Example Prediction:**

| Match Information | Value |
|---|---|
| Batting Team | Chennai Super Kings |
| Bowling Team | Delhi Capitals |
| Current Score | 75/2 |
| Target | 180 |
| Runs Required | 105 |
| Balls Remaining | 58 |
| Overs Completed | 10.2 |

**Predicted Win Probability**

- Chennai Super Kings: 64.99%
- Delhi Capitals: 35.01%

*These probabilities are model estimates, not guarantees of the actual match result.*

---

## ✨ Features

- **Live Match Situation Input:** Enter the current score, target, wickets lost, and overs completed.
- **Win Probability Prediction:** Estimate the winning chances of both competing teams.
- **Feature Engineering:** Calculate important match indicators such as required run rate, current run rate, and match pressure.
- **Interactive Dashboard:** View predictions through a user-friendly Streamlit interface.
- **Team Selection:** Supports IPL teams included in the trained model.
- **Machine Learning Pipeline:** Uses preprocessing and classification in a single pipeline.

---

## 🧠 Machine Learning Approach

The project uses an **XGBoost Classifier** to predict the match outcome.

### Model Pipeline

1. Load and prepare historical IPL match and delivery datasets.
2. Extract valid second-innings delivery records.
3. Engineer match situation features.
4. Encode categorical variables using Target Encoding.
5. Train an XGBoost classification model.
6. Evaluate performance on held-out match groups.
7. Train the final deployment pipeline and save it as a pickle file.

### Input Features

The model uses 15 features:

| Feature | Description |
|---|---|
| batting_team | Team currently batting |
| bowling_team | Team currently bowling |
| city | Match venue city |
| runs_left | Runs required to win |
| balls_left | Balls remaining |
| wickets_left | Wickets remaining |
| target | Target score |
| crr | Current run rate |
| rrr | Required run rate |
| pressure | Match pressure indicator |
| wicket_pressure | Wicket-related pressure |
| rrr_crr_diff | Difference between required and current run rates |
| balls_per_wicket | Remaining balls per available wicket |
| required_run_pressure | Required scoring pressure |
| momentum | Match momentum indicator |

### Model Evaluation

The training script evaluates the model using:

- Accuracy
- Log Loss
- ROC-AUC
- Precision
- Recall
- F1 Score

The evaluation uses a group-based train-test split to keep deliveries from the same match together and reduce data leakage between training and testing.

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Categorical Encoding | Category Encoders |
| Web Application | Streamlit |
| Model Storage | Pickle |
| Deployment | Streamlit Community Cloud |
| Version Control | Git, GitHub |

---

## 📂 Project Structure

```text
IPL-Win-Probability-Predictor/
│
├── app.py
│   └── Streamlit dashboard
│
├── ipl.py
│   └── Data preprocessing, feature engineering,
│       model training and evaluation
│
├── pipe.pkl
│   └── Trained machine learning pipeline
│
├── requirements.txt
│   └── Python dependencies
│
├── README.md
│   └── Project documentation
│
└── .gitignore
    └── Files excluded from version control
```

---

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Radhiha2007/IPL-Win-Probability-Predictor.git
```

### 2. Navigate to the Project Folder

```bash
cd IPL-Win-Probability-Predictor
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser, usually at:

```text
http://localhost:8501
```

**Note:** The pre-trained `pipe.pkl` file is required to run the dashboard. The original training datasets (`matches.csv` and `deliveries.csv`) are required if you want to retrain the model using `ipl.py`.

---

## 🚀 Deployment

The application is deployed using **Streamlit Community Cloud** and connected to the GitHub repository.

The deployment automatically installs the dependencies listed in `requirements.txt` and runs `app.py`.

🔗 **Live Application:** https://ipl-win-probability-predictor-010906.streamlit.app/

---

## 🔮 Future Improvements

- Incorporate real-time IPL match data through a live cricket API.
- Experiment with additional machine learning algorithms.
- Improve probability calibration and evaluate performance across different seasons.
- Add historical match analysis and prediction visualizations.
- Extend predictions to pre-match win probability.

---

## 👩‍💻 Author

**Radhiha Balasankaran Ramamoorthi**

B.Tech – Computer Science and Engineering (Cyber Security)

GitHub: [Radhiha2007](https://github.com/Radhiha2007)

---

*This project was developed as a practical application of machine learning, data preprocessing, feature engineering, and interactive web application development.*
