# 💳 Credit Card Fraud Detection

A complete machine learning pipeline to detect fraudulent credit card transactions.

---
### Dataset
The dataset (`creditcard.csv`) can be downloaded from:
https://www.kaggle.com/mlg-ulb/creditcardfraud
Place it inside the `data/` folder before running the notebook or app.

## 🧠 Models Used
- Logistic Regression  
- Random Forest  
- XGBoost  
- SMOTE for handling imbalanced data  

---

## 📊 Key Metrics
|       Model        | Accuracy | F1 Score | ROC-AUC |
|--------------------|----------|----------|---------|
| Logistic Regression| 0.992    | 0.87     | 0.987   |
| Random Forest      | 0.999    | 0.93     | 0.997   |
| XGBoost            | 0.999    | 0.94     | 0.998   |