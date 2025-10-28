# 📊 Tesla Stock Price Prediction (Machine Learning Project)

This project focuses on building a **machine learning model** to analyze and predict Tesla’s stock price trends using historical market data.  
It demonstrates **data preprocessing, visualization, and predictive modeling** with various ML algorithms.

---

## 🧾 Dataset Overview

- **Company:** Tesla Inc. (TSLA)
- **Time Range:** June 29, 2010 → March 17, 2017  
- **Records:** 1,693 rows  
- **Columns:**
  | Column | Description |
  |:--------|:-------------|
  | Date | Trading day |
  | Open | Opening price (USD) |
  | High | Highest price of the day |
  | Low | Lowest price of the day |
  | Close | Closing price (USD) |
  | Adj Close | Adjusted closing price |
  | Volume | Number of shares traded |

---

## ⚙️ Libraries Used

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
````

---

## 🚀 Project Workflow

1. **Import Dependencies** – Load required libraries for data manipulation, visualization, and modeling.
2. **Load Dataset** – Read Tesla historical price data (`tesla_stock.csv`).
3. **Preprocessing** – Handle missing data, convert dates, normalize features.
4. **EDA (Exploratory Data Analysis)** –

   * Visualize trends in closing price
   * Analyze correlations using Seaborn heatmaps and line plots
5. **Feature Engineering** – Create additional indicators like daily returns or moving averages.
6. **Model Training** –

   * Split data into training/testing sets
   * Train multiple models (Logistic Regression, SVM, XGBoost)
7. **Evaluation** –

   * Compare performance using metrics like Accuracy, Precision, Recall, and F1-score
   * Visualize confusion matrices and classification reports

---

## 🧠 Machine Learning Models

| Model                           | Description                                          |
| :------------------------------ | :--------------------------------------------------- |
| Logistic Regression             | Baseline linear model                                |
| SVC (Support Vector Classifier) | Non-linear kernel-based model                        |
| XGBClassifier                   | Boosted decision trees, efficient for large datasets |

---

## 📊 Visualizations

* Tesla closing price over time
* Moving average trends
* Correlation heatmaps
* Predicted vs actual price classifications

---

## 📈 Results Summary

* The **XGBoost Classifier** showed the best generalization and prediction accuracy.
* Effective at capturing non-linear patterns and stock momentum.
* Demonstrated improvement after feature scaling and tuning.

---

## 🧩 Folder Structure

```
📦 tesla-stock-price-prediction
├── tesla_stock.csv
├── tesla_model.ipynb
├── README.md
```

---

## ⚡ How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/tesla-stock-price-prediction.git
   cd tesla-stock-price-prediction
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Open and run the notebook:

   ```bash
   jupyter notebook notebooks/tesla_model.ipynb
   ```

---

## 🧮 Requirements

Create a `requirements.txt` with the following:

```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
jupyter
```

---

## 📚 Future Enhancements

* Use regression models (LSTM, GRU) for continuous price prediction
* Incorporate real-time API data from Yahoo Finance
* Add interactive dashboards using Plotly or Streamlit
* Perform hyperparameter optimization using GridSearchCV

---

## 👤 Author

**Gireesh Kasa**
B.Tech, NIT Warangal
📧 [[kasagireesh@gmail.com](mailto:kasagireesh@gmail.com]
🔗 [LinkedIn](https://linkedin.com/in/gireesh-kasa-33a546250/) | [GitHub](https://github.com/gireesh27)

---

> “Stock prediction is not about certainty — it’s about probability, pattern, and precision.”
