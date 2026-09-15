"""
Model Training and Serialization Script for Lung Cancer Prediction
Faithfully reproduces the preprocessing and model training from Lung_Cancer_Prediction.ipynb.
Exports trained model, scaler, imputers, and metadata for the Flask web application.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def train_and_export():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'dataset.csv')
    if not os.path.exists(data_path):
        data_path = os.path.join(base_dir, 'lung cancer dataset (updated).csv')
    
    print(f"Loading dataset from: {data_path}")
    dataset = pd.read_csv(data_path)
    
    # 1. Drop redundant / duplicate columns as done in notebook Cell 13
    dataset = dataset.drop(['Patient Id', 'Smoking.1', 'Swallowing Difficulty.1'], axis=1)
    
    # 2. Handle missing values with mean imputation as done in notebook Cell 15
    impute_cols = ['Alcohol use', 'OccuPational Hazards', 'Obesity', 'Smoking', 'Weight Loss']
    imputers = {}
    for col in impute_cols:
        if col in dataset.columns:
            imp = SimpleImputer(missing_values=np.nan, strategy='mean')
            imp.fit(dataset[[col]])
            dataset[col] = imp.transform(dataset[[col]]).ravel()
            imputers[col] = imp
    
    # 3. Categorical encoding as done in notebook Cells 18 & 20
    level_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    gender_mapping = {'Male': 0, 'Female': 1}
    dataset['Level'] = dataset['Level'].map(level_mapping)
    dataset['Gender'] = dataset['Gender'].map(gender_mapping)
    
    # 4. Drop columns based on correlation analysis as done in notebook Cell 24
    dropcolumn = ["Genetic Risk", "OccuPational Hazards"]
    newdata = dataset.drop(dropcolumn, axis=1)
    
    # 5. Split features and target as done in notebook Cell 26
    features = newdata.drop(['Result'], axis=1)
    target = newdata['Result']
    
    feature_names = list(features.columns)
    print(f"Number of model features: {len(feature_names)}")
    print(f"Features: {feature_names}")
    
    # 6. Train-test split (75/25 split, random_state=1)
    X_train, X_test, Y_train, Y_test = train_test_split(features, target, random_state=1, test_size=0.25)
    
    # 7. Feature scaling using MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    scaled_X_train = scaler.transform(X_train)
    scaled_X_test = scaler.transform(X_test)
    
    # 8. Train primary model: Random Forest (n_estimators=50, random_state=1) from notebook Cell 37
    primary_model = RandomForestClassifier(n_estimators=50, random_state=1)
    primary_model.fit(scaled_X_train, Y_train)
    
    # Evaluate primary model
    train_preds = primary_model.predict(scaled_X_train)
    test_preds = primary_model.predict(scaled_X_test)
    
    cm = confusion_matrix(Y_test, test_preds).tolist()
    primary_metrics = {
        'model_name': 'Random Forest Classifier',
        'n_estimators': 50,
        'train_accuracy': float(accuracy_score(Y_train, train_preds)),
        'test_accuracy': float(accuracy_score(Y_test, test_preds)),
        'precision': float(precision_score(Y_test, test_preds, zero_division=0)),
        'recall': float(recall_score(Y_test, test_preds, zero_division=0)),
        'f1_score': float(f1_score(Y_test, test_preds, zero_division=0)),
        'confusion_matrix': cm
    }
    print("Primary Model Evaluation:")
    print(json.dumps(primary_metrics, indent=2))
    
    # 9. Evaluate benchmark models from notebook for comparison on the About page
    benchmarks = {}
    
    # Decision Tree
    dt = DecisionTreeClassifier(random_state=1)
    dt.fit(scaled_X_train, Y_train)
    benchmarks['Decision Tree'] = {
        'train_accuracy': float(accuracy_score(Y_train, dt.predict(scaled_X_train))),
        'test_accuracy': float(accuracy_score(Y_test, dt.predict(scaled_X_test)))
    }
    
    # Naive Bayes
    gnb = GaussianNB()
    gnb.fit(scaled_X_train, Y_train)
    benchmarks['Gaussian Naive Bayes'] = {
        'train_accuracy': float(accuracy_score(Y_train, gnb.predict(scaled_X_train))),
        'test_accuracy': float(accuracy_score(Y_test, gnb.predict(scaled_X_test)))
    }
    
    # Support Vector Classifier
    svc = SVC(random_state=1)
    svc.fit(scaled_X_train, Y_train)
    benchmarks['Support Vector Machine'] = {
        'train_accuracy': float(accuracy_score(Y_train, svc.predict(scaled_X_train))),
        'test_accuracy': float(accuracy_score(Y_test, svc.predict(scaled_X_test)))
    }
    
    # 10. Compute feature summary statistics and schema
    feature_schema = {}
    for col in feature_names:
        min_val = float(features[col].min())
        max_val = float(features[col].max())
        mean_val = float(features[col].mean())
        
        feature_schema[col] = {
            'min': min_val,
            'max': max_val,
            'default': round(mean_val, 1)
        }
    
    # Prepare metadata dictionary
    metadata = {
        'feature_names': feature_names,
        'feature_count': len(feature_names),
        'mappings': {
            'Level': level_mapping,
            'Gender': gender_mapping,
            'Result': {0: 'Low Risk (Negative)', 1: 'High Risk (Positive)'}
        },
        'imputed_columns': list(imputers.keys()),
        'feature_schema': feature_schema,
        'primary_metrics': primary_metrics,
        'benchmarks': benchmarks
    }
    
    # 11. Save model and preprocessing artifacts
    model_dir = os.path.join(base_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(primary_model, os.path.join(model_dir, 'lung_cancer_model.joblib'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.joblib'))
    joblib.dump(imputers, os.path.join(model_dir, 'imputers.joblib'))
    
    with open(os.path.join(model_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
        
    print("\nSuccessfully exported:")
    print(f"- {os.path.join(model_dir, 'lung_cancer_model.joblib')}")
    print(f"- {os.path.join(model_dir, 'scaler.joblib')}")
    print(f"- {os.path.join(model_dir, 'imputers.joblib')}")
    print(f"- {os.path.join(model_dir, 'metadata.json')}")

if __name__ == '__main__':
    train_and_export()