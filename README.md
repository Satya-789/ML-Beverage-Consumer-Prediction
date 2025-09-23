==============================================================
               🍹 BEVERAGE CONSUMER PREDICTION 🍹
==============================================================

📌 Project Overview
--------------------------------------------------------------
Predict consumer behavior and beverage preferences using 
Machine Learning. This project helps beverage companies 
understand demographics, consumption habits, and health 
concerns to provide actionable insights.

Project includes:
✔ Data preprocessing & exploration
✔ ML model training (XGBoost)
✔ Interactive Streamlit web application

--------------------------------------------------------------
🎯 Objective
--------------------------------------------------------------
- Understand consumer demographics and preferences
- Predict beverage consumption patterns
- Provide an interactive tool for real-time insights

--------------------------------------------------------------
🛠 Workflow
--------------------------------------------------------------
1️⃣ Data Collection & Preprocessing
   - Collect data (age, income, consumption frequency, health concerns, location)
   - Clean, normalize, encode categorical features

2️⃣ Feature Engineering
   - One-hot encoding for categorical variables
   - Create derived features (health scores, preferences)

3️⃣ Model Training
   - Train **XGBoost** model
   - Save trained model as `MY_project.pkl` using `pickle`

4️⃣ Evaluation & Tuning
   - Evaluate with accuracy, F1-score, confusion matrix
   - Tune hyperparameters for best performance

5️⃣ Streamlit App
   - Interactive UI for real-time predictions
   - Input validation and error handling
   - Dynamic charts and tables

--------------------------------------------------------------
🚀 Streamlit App Features
--------------------------------------------------------------
- Input consumer demographics:
  • Age Group
  • Income Level
  • Consumption Frequency
  • Health Concerns
  • Location (Rural / Semi-Urban / Urban)
- Predicts consumer behavior instantly
- Visualizations for better insights

--------------------------------------------------------------
📦 Requirements
--------------------------------------------------------------
Python 3.10+ and libraries:

streamlit, pandas, numpy, scikit-learn, xgboost, pickle-mixin, joblib

Install all dependencies via:

