# Real-Time Fraud Detection System

Name: Shreya Dey  
Domain: AI & Data Analytics  
Dataset: IEEE-CIS Fraud Detection

This project detects credit card fraud using machine learning and explains the predictions using SHAP. I also built a Streamlit dashboard to check transaction risk scores and view important fraud patterns.

## Project Workflow

1. Load `train_transaction.csv` and `train_identity.csv`.
2. Merge both files using `TransactionID`.
3. Check class imbalance, missing values, amount distribution, and correlations.
4. Drop columns with very high missing values.
5. Impute remaining missing values.
6. Encode categorical columns.
7. Create amount, time, and device-based features.
8. Split the data using stratified train-test split.
9. Apply SMOTE only on the training data.
10. Train and compare LightGBM, XGBoost, and Isolation Forest.
11. Tune the decision threshold using F1-score.
12. Explain model predictions using SHAP.
13. Create risk tiers and dashboard outputs.

## Main Decisions

- I used median imputation for numerical columns because transaction data can contain outliers.
- I used mode imputation for categorical columns because it keeps the most common known category.
- I applied SMOTE only after train-test split to avoid data leakage.
- I focused more on PR-AUC, recall, and F1-score than accuracy because fraud cases are rare.
- I used SHAP so that the model output can be explained to a non-technical user.

## Dataset Setup

Download the dataset from Kaggle:

https://www.kaggle.com/c/ieee-fraud-detection/data

The notebook uses these two files:

- `train_transaction.csv`
- `train_identity.csv`

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Open and run:

```bash
analysis.ipynb
```

Then start the dashboard:

```bash
streamlit run dashboard/app.py
```

## Main Files

- `analysis.ipynb`: complete notebook with all tasks
- `data/train_transaction.csv`: transaction dataset
- `data/train_identity.csv`: identity dataset
- `dashboard/app.py`: Streamlit dashboard
- `dashboard/model.pkl`: trained model artifact
- `model_comparison.png`: model comparison/threshold chart
- `shap_summary.png`: SHAP summary plot
- `charts/`: saved plots
- `summary.docx`: short project summary
- `requirements.txt`: required libraries
- `README.md`: project instructions


