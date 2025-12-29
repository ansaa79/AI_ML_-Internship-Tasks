#    Task 02: Customer Churn Prediction - ML Pipeline

## Objective
Build a production-ready machine learning pipeline to predict customer churn using the Telco Customer Churn dataset. The goal is to create a reusable end-to-end workflow for preprocessing, model training, evaluation, and deployment.

---

## Dataset
- **Source:** Telco Customer Churn Dataset (Kaggle)
- **Samples:** ~7,000 customers  
- **Features:** 19 features including demographics, services, and account information  
- **Target Variable:** `Churn` (Yes/No)

---

## Technologies Used
- **Python 3.x** 
- **Pandas:** Data manipulation
- **Numpy:** Numerical operations  
- **scikit-learn :** Machine learning pipeline and models
- **Matplotlib & Seaborn :** Data visualization 
- **Joblib :** Model serialization

---

## Methodology

### 1. Data Preprocessing
- Removed `customerID` (non-predictive)  
- Converted `TotalCharges` to numeric  
- Handled missing values with median imputation  
- Encoded target (`Churn`: Yes → 1, No → 0)

### 2. Feature Engineering
- **Numerical:** tenure, MonthlyCharges, TotalCharges  
- **Categorical:** gender, Partner, Dependents, PhoneService, etc.  
- Applied StandardScaler to numerical features
- Applied OneHotEncoder to categorical features

### 3. Pipeline Construction
- **Preprocessing Pipeline:** Imputation, scaling, encoding  
- **Model Pipeline:** Combines preprocessing and classifier  
- **Full Pipeline:** End-to-end workflow for predictions  

### 4. Models Trained
- **Logistic Regression**: Baseline, fast training & prediction  
- **Random Forest Classifier**: More complex, hyperparameter tuning with `GridSearchCV`  
- Tuned parameters: `n_estimators`, `max_depth`, `min_samples_split`

### 5. Model Evaluation
- Metrics used: 
- **Accuracy**: Overall correctness
- **Precision**: How many predicted churns were correct
- **Recall**: How many actual churns were caught
- **F1-Score**: Harmonic mean of precision and recall

---

## Key Results

| Model                 | Accuracy | Precision | Recall | F1-Score |
|-----------------------|--------- |---------- |--------|----------|
| Logistic Regression   | 0.80     | 0.66      | 0.55   | 0.60     |
| Random Forest (Tuned) | 0.79     | 0.65      | 0.51   | 0.57     |

**Best Model:** Random Forest with optimized hyperparameters: 
- `n_estimators`: 100  
- `max_depth`: 20  
- `min_samples_split`: 5  

---

## Key Insights
1. **Feature Importance:** `tenure` and `MonthlyCharges` are strong predictors of churn  
2. **Class Imbalance:** ~73% non-churn, ~27% churn  
3. **Model Trade-offs:** Random Forest has better precision but lower recall than Logistic Regression  
4. **Production Ready:** Pipeline can be deployed using `joblib`

---

## Using the Saved Pipeline

```python
import joblib
import pandas as pd

# Load the pipeline
pipeline = joblib.load('churn_prediction_pipeline.pkl')

# Make predictions on new data
new_data = pd.DataFrame([...])  # Replace with your new customer data
predictions = pipeline.predict(new_data)
```

---

## Skills Demonstrated
- Data preprocessing and cleaning  
- Feature engineering (numerical + categorical)  
- Pipeline construction with scikit-learn  
- Hyperparameter tuning using GridSearchCV  
- Model evaluation with multiple metrics  
- Model serialization for production deployment  

---

## Learning Outcomes
- Built reproducible ML pipelines  
- Handled mixed data types (numerical + categorical)  
- Gained experience in hyperparameter tuning  
- Practiced model evaluation and comparison  
- Developed production-ready ML workflow 

---

## Contact
**Intern Name:** Ansa Bint E Zia  
**Role:** AI/ML Engineering Intern at DevelopersHub Corporation

**GitHub:** https://github.com/ansaa79
**Email:**  ansabintezia72@gmail.com

**Date:** December 2025
