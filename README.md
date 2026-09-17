# 🚗 Used Car Price Prediction

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-006400)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-02569B)](https://lightgbm.readthedocs.io/)

## Overview

This project predicts the resale (selling) price of used cars based on attributes like their current market price, age, mileage, fuel type, transmission, and ownership history. It covers the full pipeline end-to-end: exploratory data analysis, feature engineering, training and comparing five regression models (with hyperparameter variants), and deploying the best-performing model as an interactive Streamlit web app that lets a user enter a car's details and get an instant price estimate.

## Dataset

- **Source:** [Kaggle – Vehicle Dataset from CarDekho](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho) (`car data.csv`)
- **Features used:** `Present_Price`, `Kms_Driven`, `Car_Age` (engineered from `Year`), `Fuel_Type`, `Seller_Type`, `Transmission`, `Owner`
- **Target:** `Selling_Price`
- **Total samples:** 301 rows, 9 original columns (`Car_Name` and `Year` are dropped after feature engineering — see below)

## Key EDA Findings

- `Present_Price` (the car's current market price) shows the strongest positive correlation with `Selling_Price` — it's the single best predictor of resale value.
- `Selling_Price` is right-skewed: most cars sell in the lower price range, with a smaller number of high-priced cars pulling the tail to the right.
- `Car_Age` (engineered from `Year`) shows a clear negative relationship with `Selling_Price` — older cars sell for less, confirming depreciation over time.
- Diesel cars and automatic-transmission cars tend to have a higher median `Selling_Price` than petrol/CNG and manual cars respectively.
- `Car_Name` (high cardinality) and `Owner` (very low variance — most cars have had 0 previous owners) carry little independent signal once the other features are included.

## Model Comparison

Sorted by Test R² descending:

| Model                | Train R² | Test R² | Test RMSE |
|----------------------|----------|---------|-----------|
| LightGBM (tuned)     | 0.9810   | 0.9494  | 0.8162    |
| Random Forest n=200  | 0.9932   | 0.9465  | 0.8391    |
| Random Forest n=100  | 0.9929   | 0.9448  | 0.8523    |
| XGBoost (tuned)      | 0.9988   | 0.9443  | 0.8559    |
| LightGBM (default)   | 0.9773   | 0.9422  | 0.8724    |
| XGBoost (default)    | 1.0000   | 0.9380  | 0.9033    |
| Ridge α=1            | 0.8190   | 0.8674  | 1.3213    |
| Ridge α=0.1          | 0.8197   | 0.8670  | 1.3232    |
| Linear Regression    | 0.8197   | 0.8669  | 1.3237    |
| Ridge α=10           | 0.8140   | 0.8621  | 1.3471    |

## Final Model

**Model:** LightGBM (tuned)
**Test R²:** 0.9494
**Test RMSE:** 0.8162

**Why this model:**
- It achieved the highest Test R² and lowest Test RMSE among all 10 model configurations tried, edging out the default LightGBM and both Random Forest sizes — the tuning (more trees, smaller leaves, lower learning rate) gave it a small but consistent advantage.
- **Overfitting check:** Train R² (0.9810) vs. Test R² (0.9494) leaves a gap of about 0.032. That's small enough to call reasonable generalization, though it's worth noting all the tree-based models fit the training data much more tightly than Linear Regression/Ridge (whose train and test R² are nearly identical) — a sign tree ensembles have more capacity to memorize, even if it isn't severe here.
- **Connection to EDA:** every tree-based model (Random Forest, XGBoost, LightGBM) clearly outperformed Linear Regression and Ridge, whose Test R² topped out around 0.867. This suggests the relationship between the features and `Selling_Price` isn't purely linear — there are likely interaction effects (e.g., how `Kms_Driven` and `Car_Age` combine, or how `Fuel_Type` shifts price differently across price brackets) that only a tree ensemble can capture. The EDA's correlation findings (`Present_Price` and `Car_Age` as the strongest linear correlates) were directionally right but incomplete on their own.

## Web Application

Deployed using Streamlit.

**Live URL:** [https://rotoncsedu-car-price-prediction.streamlit.app/](https://rotoncsedu-car-price-prediction.streamlit.app/)

### Screenshot

![Streamlit App](screenshots/app.png)

## Installation

```bash
git clone https://github.com/rotoncsedu/Car-Price-Prediction
cd Car-Price-Prediction
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

## Project Structure

```
Car-Price-Prediction/
├── data/
│   └── car data.csv
├── models/
│   └── best_model.pkl
├── screenshots/
│   └── app.png
├── notebooks/
│   ├── 1-eda.ipynb
│   └── 2_training.ipynb
├── app.py
├── requirements.txt
└── README.md
```

## Technologies Used

- Python
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn, XGBoost, LightGBM
- Streamlit

## 👤 Author

**Md. Al Imran**
Programmer
Begum Rokeya University, Rangpur, Bangladesh