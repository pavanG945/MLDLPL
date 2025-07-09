# 🏏 IPL 2025 Winner Prediction – MLDLPL Hackathon Project

## 🧠 Overview

This project aims to predict the top-performing teams in IPL 2025 using historical player and team performance data. Developed as part of the **MLDLPL Hackathon**, the project uses machine learning models and streamlines insights through a web interface and a Power BI dashboard.

---

## 👥 Team

- **Team Name:** Bhishma  
- **Team Lead:** M. Bala Rajendra Reddy  
- **Team Member:** Gembali Pavan Kumar

---

## 📁 Project Structure

```
├── trail.py                # Model training, feature engineering, evaluation
├── app.py                  # Predicts 2025 results using trained model
├── web.py                  # Streamlit app to display final results
├── DATATHON.xlsx           # Cleaned and normalized IPL dataset
├── mldlpl.pbix             # Power BI dashboard for data visualization
├── best_model.pkl          # Saved best model from training
├── predictions_standings_2025.csv # Final predicted results
```

---

## ⚙️ Key Features

- **Feature Engineering:** Win Momentum, Boundary Ratio, Bowling Efficiency, Batting Impact, Powerplay Dominance
- **ML Models Used:** Random Forest, XGBoost, Logistic Regression, Linear Regression (baseline)
- **Data Handling:** Cleaned team names, one-hot encoded, normalized
- **Model Evaluation:** Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Imbalance Handling:** SMOTE
- **Web Interface:** Built using Streamlit for selecting and viewing team standings
- **Visualization:** Power BI report with interactive charts

---

## 📊 Output

- Predictions saved in: `predictions_standings_2025.csv`
- Top teams sorted by predicted standing and win momentum
- Visual results hosted via a Streamlit interface and Power BI

---

## 🚀 How to Run

1. **Install Dependencies:**

```bash
pip install pandas streamlit plotly scikit-learn numpy xgboost imbalanced-learn joblib
```

2. **Train Model:**

```bash
python trail.py
```

3. **Predict 2025 Results:**

```bash
python app.py
```

4. **Launch Web Interface:**

```bash
streamlit run web.py
```

---

## 🏆 Achievements

- **Runner-Up** – MLDLPL Hackathon, Thiagarajar School of Management, April 2025

---

## 📌 Conclusion

A complete pipeline for predicting IPL 2025 standings using past performance data, built with explainable ML models, and presented with interactive UI and dashboards for enhanced usability.

---

## 📫 Contact

For queries, contact:  
**Gembali Pavan Kumar** – pavangembali945@gmail.com
