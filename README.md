# 🚖 Time Series Forecasting — Taxi Demand Prediction

Predicting hourly taxi demand at airports using time series analysis and machine learning models.

---

## 📌 Overview

Sweet Lift Taxi collected historical data on taxi orders at airports. The goal is to **predict the number of taxi requests for the next hour**, enabling the company to allocate drivers efficiently during peak times.

**Business constraint:** RMSE on the test set must not exceed **48**.

---

## 📊 Dataset

| Feature | Description |
|---|---|
| `datetime` | Timestamp of taxi orders (10-minute intervals) |
| `num_orders` | Number of taxi orders |

- **Source:** `taxi.csv`
- **Period:** March 2018 – August 2018
- **Raw records:** 26,496 rows (resampled to 4,416 hourly records)

---

## 🔧 Methodology

### 1. Data Preparation
- Parsed datetime column and set as index
- Resampled data from 10-minute intervals to **hourly frequency** using `.resample('1H').sum()`
- Verified no missing values or true duplicate records

### 2. Exploratory Data Analysis
- Decomposed time series into **trend**, **seasonality**, and **residuals**
- Identified consistent **demand growth** over the months
- Detected two daily demand peaks: **4–6 PM** (end of business day) and **8–11 PM**

### 3. Feature Engineering
- **Lag features:** 48 lag variables (1h to 48h)
- **Rolling mean features:** 24h and 48h rolling averages (shift=1 to prevent leakage)
- **Calendar features:** hour of day, day of week

### 4. Model Training
- Chronological 90/10 train-test split (no shuffle — respects time series order)
- Models trained: **Linear Regression**, **Random Forest**, **Gradient Boosting**

---

## 📈 Results

| Model | RMSE (Test Set) | Within Limit? |
|---|---|---|
| Linear Regression | **42.86** | ✅ |
| Random Forest | 47.09 | ✅ |
| Gradient Boosting | 63.17 | ❌ |

🏆 **Best model: Linear Regression** (RMSE ≈ 42.86)

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-lightgrey)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Statsmodels](https://img.shields.io/badge/Statsmodels-Time%20Series-green)

- **Python** — Pandas, NumPy
- **Machine Learning** — Scikit-learn (LinearRegression, RandomForestRegressor, GradientBoostingRegressor)
- **Time Series Analysis** — Statsmodels (seasonal_decompose)
- **Visualization** — Matplotlib

---

## 📁 Project Structure

```
TimeSeriesForecasting/
│
├── notebook.ipynb       # Full analysis and modeling
├── README.md
└── .gitignore
```

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/laaurasaporski/TimeSeriesForecasting.git

# Install dependencies
pip install pandas numpy scikit-learn statsmodels matplotlib

# Open the notebook
jupyter notebook notebook.ipynb
```

---

## 💡 Key Insights

- Demand follows a clear **upward trend** from March to August 2018
- Two daily peaks suggest driver allocation should prioritize **late afternoon and evening**
- Linear Regression outperformed ensemble methods, indicating that **lag features capture the linear temporal pattern** effectively
