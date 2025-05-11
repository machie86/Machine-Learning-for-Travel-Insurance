# Machine Learning for Travel Insurance Claim Prediction

## Overview
This project focuses on predicting travel insurance claims using machine learning techniques to address the extreme class imbalance in the dataset (98% "No Claim" vs. 2% "Claim"). The goal is to develop a robust model with high recall for the minority class (claims) to help insurance companies manage risk, optimize premium pricing, and improve operational efficiency. The project tackles challenges such as missing values, data anomalies, and low precision due to false positives.

The notebook, `Travel_Insurance_V10_Final.ipynb`, provides a comprehensive analysis, including data preprocessing, feature engineering, model training, evaluation, and actionable business recommendations.

## Dataset
The dataset contains historical travel insurance policyholder data with 44,328 entries and 11 features, including:

- **Categorical Features**: Agency, Agency Type, Distribution Channel, Product Name, Gender, Destination, Claim (target).
- **Numerical Features**: Duration, Net Sales, Commission, Age.

**Key Challenges**:
- Extreme class imbalance (98% No Claim vs. 2% Claim).
- Missing values in Gender (71.3% missing).
- Data anomalies (e.g., negative Duration, unrealistic Age values like 0 or 118).
- Duplicated rows (4,667 duplicates).

**Data Source**: [Google Drive Link](https://drive.google.com/drive/folders/1iVx5k6tWglqfHb05o0DElg8JHg7VVG_J)

## Project Structure
The notebook is organized into the following sections:

1. **Business Problem Understanding**:
   - Background on travel insurance and the importance of claim prediction.
   - Problem statement highlighting the impact of class imbalance and data quality issues.
   - Goals to build a high-recall model, improve data quality, and provide actionable insights.

2. **Data Understanding**:
   - Description of features, data types, and unique values.
   - Statistical summary of numerical features, identifying anomalies (e.g., negative Net Sales, extreme Duration).
   - Observations on missing values and class imbalance.

3. **Data Preprocessing**:
   - Handling missing values (e.g., imputing Gender using IterativeImputer).
   - Removing duplicates and clipping outliers using the IQR method.
   - Feature engineering (e.g., creating `Price_Category` based on Net Sales).

4. **Modeling**:
   - Algorithms tested: Logistic Regression, K-Nearest Neighbors, Decision Tree, Random Forest, and more.
   - Techniques to address imbalance: RandomOverSampler, class weight adjustment.
   - Pipeline with preprocessing (scaling, encoding) and feature selection (SelectKBest, SelectFromModel).
   - Hyperparameter tuning using GridSearchCV.

5. **Evaluation**:
   - Primary metric: Recall for the minority class (Claim) and F2-score to prioritize recall.
   - Best model: Logistic Regression with recall of 0.88 and F2-score of 0.2457 for the Claim class.
   - Confusion matrix analysis showing high recall (58 True Positives, 8 False Negatives) but low precision due to 847 False Positives.

6. **Model Limitations**:
   - Low precision (<0.07) due to high false positives.
   - Limited feature engineering and unresolved anomalies.
   - Potential overfitting or underfitting for new scenarios.

7. **Conclusion & Recommendations**:
   - Model effectively identifies high-risk policyholders but requires precision improvement.
   - Business recommendations include optimizing premium pricing, reducing false positives, enhancing data quality, and improving operational efficiency.

## Key Findings
- **Model Performance**: Logistic Regression achieves a recall of 0.88 for claims, detecting 88% of actual claims, but low precision (<0.07) due to 847 false positives.
- **Important Features**: `Price_Category`, `Net Sales`, and `Duration` are the most influential predictors of claims.
- **Business Impact**: High recall supports risk management, but false positives increase operational costs. Improved data quality and feature engineering can enhance model accuracy.

## Installation
To run the notebook, ensure you have the following dependencies installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn category_encoders jcopml
