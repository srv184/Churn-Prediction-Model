# train_model.py

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# Load your dataset (replace with your actual dataset path)
df = pd.read_csv(r"C:\Users\soura\Downloads\customer_churn_dataset-training-master.csv")

# Clean & enrich just like in the Streamlit app
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df['tenure'] = pd.to_numeric(df.get('tenure', pd.Series([0]*len(df))), errors='coerce')
df['monthly_charges'] = pd.to_numeric(df.get('monthly_charges', pd.Series([0]*len(df))), errors='coerce')
if 'total_charges' not in df.columns:
    df['total_charges'] = df['monthly_charges'] * df['tenure']
df['CLV'] = df['monthly_charges'] * df['tenure']
df['long_term_customer'] = df['tenure'] > 12
df['high_value_customer'] = df['total_charges'] > 2000
df['recent_low_value_customer'] = (df['tenure'] < 12) & (df['total_charges'] < 1000)

# Define target label (temporary logic, update with your actual churn column if available)
df['churn'] = ((df['tenure'] < 12) & (df['total_charges'] < 1000)).astype(int)

# Prepare features
X = df.drop(columns=['churn'])
X = pd.get_dummies(X, drop_first=True)
y = df['churn']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model with GridSearchCV for optimization
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
}

clf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy', n_jobs=-1)
clf.fit(X_train, y_train)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, clf.predict(X_test)))
print(classification_report(y_test, clf.predict(X_test)))

# Save best model
joblib.dump(clf.best_estimator_, "optimized_churn_model.pkl")
print("✅ Model saved as optimized_churn_model.pkl")
