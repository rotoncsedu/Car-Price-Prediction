# Used Car Price Prediction

## Overview

This project predicts the resale (selling) price of used cars based on attributes like their current market price, age, mileage, fuel type, transmission, and ownership history. It covers the full pipeline end-to-end: exploratory data analysis, feature engineering, training and comparing five regression models, and deploying the best-performing model as an interactive Streamlit web app that lets a user enter a car's details and get an instant price estimate.

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

> _Note: the bullets above summarize the patterns identified during EDA (see `1-eda.ipynb`). Fill in the exact correlation coefficients from your notebook output if you'd like more precise figures here._

## Model Comparison

| Model              | Train R² | Test R² | Test RMSE |
|--------------------|----------|---------|-----------|
| Linear Regression  | [value]  | [value] | [value]   |
| Ridge Regression   | [value]  | [value] | [value]   |
| Random Forest      | [value]  | [value] | [value]   |
| XGBoost            | [value]  | [value] | [value]   |
| LightGBM           | [value]  | [value] | [value]   |

> _Copy these numbers directly from the "Sorted Model Comparison Table" cell output in `2_training.ipynb` — do not leave this table blank._

## Final Model

**Model:** [Name of the winning model from your sorted comparison table]
**Test R²:** [value]
**Why this model:** [Copy/adapt the explanation from the "Model Selection" section of `2_training.ipynb` — cover (1) why it outperformed the others, (2) whether the Train vs. Test R² gap indicates overfitting, and (3) how this connects back to the EDA findings above.]

## Web Application

Deployed using Streamlit.

**Live URL:** [your Streamlit Cloud URL, e.g. https://your-app-name.streamlit.app]

### Screenshot

![Streamlit App](screenshots/streamlit_app.png)

> _Take a screenshot of your running app (with a filled-in prediction shown) and save it to `screenshots/streamlit_app.png` in your repo so this image renders correctly._

## Installation

```bash
git clone [your-repo-url]
cd used-car-price-prediction
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

## Project Structure

```
used-car-price-prediction/
├── data/
│   └── car data.csv
├── models/
│   └── best_model.pkl
├── screenshots/
│   └── streamlit_app.png
├── 1-eda.ipynb
├── 2_training.ipynb
├── app.py
├── requirements.txt
└── README.md
```

## Technologies Used

- Python
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn, XGBoost, LightGBM
- Streamlit
