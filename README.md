# 📉 Telco Customer Churn Prediction Engine (v1.0)

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-Machine_Learning-orange?style=for-the-badge&logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient_Boosting-red?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-v0.100+-009688?style=for-the-badge&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-v1.25+-red?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

> **End-to-end Machine Learning solution designed to predict customer churn in telecommunications, identify key risk drivers, and empower retention strategies using advanced ensemble modeling.**

---

## 📌 Problem Statement & Business Impact

Customer acquisition costs in telecom are **5x to 25x higher** than customer retention. Identifying churn-prone customers *before* they leave allows telecom providers to launch proactive retention campaigns and protect revenue.

**Telco Churn Prediction Engine** delivers:
* **Exploratory Data Analysis (EDA):** Deep analysis of tenure, contract types, internet services, and billing structures.
* **Feature Engineering & Preprocessing:** Automated handling of categorical encodings, scaling, and class imbalance mitigation (SMOTE).
* **Supervised Machine Learning Pipeline:** Trained on multiple ensemble models (XGBoost, Random Forest, LightGBM, Logistic Regression).
* **Interactive Risk Predictor:** Live user interface for inputting customer profiles and generating real-time churn probabilities.

---

## 📸 Live Streamlit App Showcase

Below are live production screenshots of the interactive Streamlit Churn Analysis & Risk Scoring Dashboard:

| 1. Customer Profile Input | 2. Real-Time Churn Scoring | 3. Risk Factor Breakdown | 4. Retention Insights |
| :---: | :---: | :---: | :---: |
| ![Profile Input](Screenshot%202026-07-22%20151721.png) | ![Churn Scoring](Screenshot%202026-07-22%20151738.png) | ![Risk Factor Analysis](Screenshot%202026-07-22%20151751.png) | ![Retention Recommendations](Screenshot%202026-07-22%20151807.png) |

> 💡 **Note:** The Streamlit interface computes dynamic churn probabilities based on input features, helping retention teams target at-risk users proactively.

---

## 🏗️ System Architecture & Workflow

```text
[ Raw Telecom Customer Data ]
              │
              ▼
┌────────────────────────────────────────┐
│     Data Cleaning & Preprocessing      │
│  (Imputation, Encoding, Normalization) │
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│      Exploratory Data Analysis &       │
│         Feature Engineering            │
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│   Model Training & Class Balancing     │
│   (XGBoost / LightGBM / SMOTE)         │
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│   Evaluation & Feature Importance      │
│     (ROC-AUC, Confusion Matrix)        │
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│     Deployment & Scoring Interface     │
│       (FastAPI / Streamlit Dashboard)  │
└────────────────────────────────────────┘

---
## 🛠️ Tech Stack & Libraries
Core Language: Python 3.10+

Data Processing & Analytics: Pandas, NumPy, Scikit-Learn

Machine Learning Algorithms: XGBoost, LightGBM, Random Forest, Logistic Regression

Visualization: Seaborn, Matplotlib, Plotly

Deployment & UI: Streamlit / FastAPI

---
##🚀 Quick Start Guide
1. Clone & Setup
git clone [https://github.com/saqer-alabdallat/Telco-Customer-Churn.git](https://github.com/saqer-alabdallat/Telco-Customer-Churn.git)
cd Telco-Customer-Churn

2. Install Dependencies
pip install -r requirements.txt

3. Run Application
# Launch Streamlit Predictor App
streamlit run app.py
