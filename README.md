# Customer Churn Prediction

A machine learning project to predict customer churn in telecom companies using Python and Streamlit.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)

## 🎥 Demo

[![Watch Demo](https://img.shields.io/badge/Watch-Demo%20Video-red?style=for-the-badge&logo=youtube)](https://youtu.be/5VF1I2Vohh4?si=zOI5ezg4slqVqUi_)


## 📖 About

Predicts whether a telecom customer will churn using machine learning. The project includes data analysis, model training, and an interactive web dashboard for real-time predictions.

- **Dataset**: 7,043 customer records
- **Model**: Random Forest Classifier
- **Accuracy**: 78%
- **Dashboard**: Streamlit web app

## ✨ Features

- Exploratory Data Analysis (EDA) with visualizations
- Handles class imbalance using SMOTE
- Compares multiple models (Decision Tree, Random Forest, XGBoost)
- Interactive dashboard for real-time predictions
- Feature engineering and preprocessing pipeline

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost, LightGBM
- SMOTE (imbalanced-learn)
- Streamlit
- Matplotlib, Seaborn

## 🚀 Quick Start

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
```

2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

### Run the Dashboard

```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

## 📊 Model Performance

| Model | CV Accuracy | Test Accuracy |
|-------|-------------|---------------|
| Decision Tree | 79% | - |
| **Random Forest** | **85%** | **78%** |
| XGBoost | 84% | - |

**Metrics:**
- Precision (Non-Churn): 85%
- Recall (Non-Churn): 85%
- Precision (Churn): 58%
- Recall (Churn): 58%

## 📁 Project Structure

```
customer-churn-prediction/
├── app.py                          # Streamlit dashboard
├── Customer_Churn_Prediction.ipynb # Jupyter notebook (EDA + Training)
├── customer_churn.csv              # Dataset
├── Customer_churn_model.pkl        # Trained model
├── scaler.pkl                      # Feature scaler
├── encoders.pkl                    # Label encoders
├── requirements.txt                # Dependencies
└── README.md
```

## 🔍 Key Findings

- **Tenure** and **Contract Type** are strong predictors of churn
- Month-to-month contracts have higher churn rates
- Longer tenure = lower churn probability
- SMOTE oversampling improved minority class predictions

## 📝 License

MIT License
