# Diabetes Prediction Project - EB3204 Pembelajaran Mesin dalam Teknik Biomedis

A machine learning project to predict diabetes risk using the Pima Indians Diabetes Dataset.

## Overview
This project develops a classification model to predict diabetes onset based on medical measurements. The best performing model achieved **82.3% ROC AUC** on the test set.

## Contributor
| Name | NIM |
|------|-----|
| Erdianti Wiga Putri Andini | 13522053 |

## Dataset
- **Source**: Pima Indians Diabetes Dataset
- **Size**: 768 samples, 8 features
- **Target**: Binary classification (Diabetes/No Diabetes)
- **Class Distribution**: 34.9% diabetes prevalence

## Features
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

## Model Performance
**Best Model**: Random Forest (Tuned)
- **Test Accuracy**: 72.7%
- **Test ROC AUC**: 82.3%
- **Test F1-Score**: 58.0%
- **Precision**: 63.0%
- **Recall**: 53.7%

## Key Features (by importance)
1. Glucose (32.8%)
2. BMI (16.5%)
3. Age (11.8%)
4. Diabetes Pedigree Function (11.4%)
5. Insulin (8.1%)

## Usage
```python
# Load saved model components
model = joblib.load('best_diabetes_model.pkl')
imputer = joblib.load('imputer.pkl')
scaler = joblib.load('scaler.pkl')

# Example prediction
patient_data = {
    'Pregnancies': 2,
    'Glucose': 120,
    'BloodPressure': 80,
    'SkinThickness': 25,
    'Insulin': 100,
    'BMI': 28.5,
    'DiabetesPedigreeFunction': 0.5,
    'Age': 45
}

result = predict_diabetes_risk(patient_data)
# Returns: prediction, probability, risk category, recommendation
```

## Risk Categories
- **Low Risk** (< 30%): Maintain healthy lifestyle
- **Medium Risk** (30-70%): Consider lifestyle modifications
- **High Risk** (> 70%): Immediate medical consultation

## Dependencies
```
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
```

## Files
- `diabetes.csv` - Dataset
- `best_diabetes_model.pkl` - Trained Random Forest model
- `imputer.pkl` - Data imputation pipeline
- `scaler.pkl` - Feature scaling pipeline

## Limitations
- Trained on specific population (Pima Indians)
- Should be used as decision support tool only
- Requires validation on diverse populations