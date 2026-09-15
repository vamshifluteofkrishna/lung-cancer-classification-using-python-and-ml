"""
Flask Web Application for Lung Cancer Prediction
Backend server integrating the existing ML model, preprocessing pipeline, and clinical dashboard.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'lung-cancer-prediction-secret-key-2026')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

# Model and Preprocessor artifacts
model = None
scaler = None
imputers = None
metadata = None

def load_artifacts():
    global model, scaler, imputers, metadata
    try:
        model_path = os.path.join(MODEL_DIR, 'lung_cancer_model.joblib')
        scaler_path = os.path.join(MODEL_DIR, 'scaler.joblib')
        imputers_path = os.path.join(MODEL_DIR, 'imputers.joblib')
        metadata_path = os.path.join(MODEL_DIR, 'metadata.json')

        if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(metadata_path)):
            # If artifacts are missing, trigger training
            from model.train_model import train_and_export
            train_and_export()

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        imputers = joblib.load(imputers_path)
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        print("ML artifacts successfully loaded.")
    except Exception as e:
        print(f"Error loading ML artifacts: {e}")
        model = None
        scaler = None
        imputers = None
        metadata = None

# Initialize artifacts
load_artifacts()

# Feature mapping and validation helpers
FEATURE_ORDER = [
    'Age', 'Gender', 'Air Pollution', 'Alcohol use', 'Dust Allergy',
    'chronic Lung Disease', 'Balanced Diet', 'Obesity', 'Smoking',
    'Passive Smoker', 'Chest Pain', 'Coughing of Blood', 'Fatigue',
    'Weight Loss', 'Shortness of Breath', 'Wheezing', 'Swallowing Difficulty',
    'Clubbing of Finger Nails', 'Frequent Cold', 'Dry Cough', 'Snoring', 'Level'
]

FEATURE_RANGES = {
    'Age': (1, 120),
    'Gender': ['Male', 'Female', 0, 1],
    'Air Pollution': (1, 8),
    'Alcohol use': (1, 8),
    'Dust Allergy': (1, 8),
    'chronic Lung Disease': (1, 7),
    'Balanced Diet': (1, 7),
    'Obesity': (1, 7),
    'Smoking': (1, 8),
    'Passive Smoker': (1, 8),
    'Chest Pain': (1, 9),
    'Coughing of Blood': (1, 9),
    'Fatigue': (1, 9),
    'Weight Loss': (1, 8),
    'Shortness of Breath': (1, 9),
    'Wheezing': (1, 8),
    'Swallowing Difficulty': (1, 8),
    'Clubbing of Finger Nails': (1, 9),
    'Frequent Cold': (1, 7),
    'Dry Cough': (1, 7),
    'Snoring': (1, 7),
    'Level': ['Low', 'Medium', 'High', 0, 1, 2]
}

FEATURE_LABELS = {
    'Age': 'Patient Age (years)',
    'Gender': 'Biological Sex',
    'Level': 'Clinical Assessment Level',
    'Air Pollution': 'Air Pollution Exposure',
    'Alcohol use': 'Alcohol Consumption Frequency',
    'Dust Allergy': 'Dust Allergy Severity',
    'chronic Lung Disease': 'Chronic Lung Disease History',
    'Balanced Diet': 'Dietary Balance & Nutrition',
    'Obesity': 'Obesity / BMI Indicator',
    'Smoking': 'Active Smoking Intensity',
    'Passive Smoker': 'Passive Smoking / Secondhand Smoke',
    'Chest Pain': 'Chest Pain Intensity',
    'Coughing of Blood': 'Coughing of Blood (Hemoptysis)',
    'Fatigue': 'Fatigue / Chronic Exhaustion',
    'Weight Loss': 'Unexplained Weight Loss',
    'Shortness of Breath': 'Shortness of Breath (Dyspnea)',
    'Wheezing': 'Wheezing / Respiratory Whistling',
    'Swallowing Difficulty': 'Swallowing Difficulty (Dysphagia)',
    'Clubbing of Finger Nails': 'Clubbing of Fingernails',
    'Frequent Cold': 'Frequency of Respiratory Colds',
    'Dry Cough': 'Persistent Dry Cough',
    'Snoring': 'Severe Snoring / Sleep Apnea Indicator'
}

def validate_and_preprocess(form_data):
    """
    Validates form data and applies exact mapping and preprocessing used in notebook.
    Returns: (cleaned_df, raw_display_data, errors)
    """
    errors = []
    cleaned = {}
    display_data = {}

    for feature in FEATURE_ORDER:
        raw_val = form_data.get(feature)
        if raw_val is None or str(raw_val).strip() == '':
            errors.append(f"Missing required field: '{FEATURE_LABELS.get(feature, feature)}'")
            continue

        raw_str = str(raw_val).strip()

        # Handle Gender mapping
        if feature == 'Gender':
            display_data[feature] = raw_str.capitalize()
            if raw_str.lower() in ['male', '0']:
                cleaned[feature] = 0
            elif raw_str.lower() in ['female', '1']:
                cleaned[feature] = 1
            else:
                errors.append("Invalid value for Gender. Must be 'Male' or 'Female'.")
            continue

        # Handle Level mapping
        if feature == 'Level':
            display_data[feature] = raw_str.capitalize()
            mapping = {'low': 0, 'medium': 1, 'high': 2, '0': 0, '1': 1, '2': 2}
            if raw_str.lower() in mapping:
                cleaned[feature] = mapping[raw_str.lower()]
            else:
                errors.append("Invalid value for Clinical Level. Must be 'Low', 'Medium', or 'High'.")
            continue

        # Handle Numerical features
        try:
            val_num = float(raw_str)
            display_data[feature] = int(val_num) if val_num.is_integer() else val_num
            min_v, max_v = FEATURE_RANGES[feature]
            if val_num < min_v or val_num > max_v:
                errors.append(f"'{FEATURE_LABELS.get(feature, feature)}' must be between {min_v} and {max_v} (received {val_num}).")
            cleaned[feature] = val_num
        except ValueError:
            errors.append(f"Invalid numeric input for '{FEATURE_LABELS.get(feature, feature)}': '{raw_str}'")

    if errors:
        return None, None, errors

    # Create single-row DataFrame in exact FEATURE_ORDER
    df = pd.DataFrame([cleaned], columns=FEATURE_ORDER)

    # Impute if missing values are present
    if imputers:
        for col, imp in imputers.items():
            if col in df.columns and df[col].isnull().any():
                df[col] = imp.transform(df[[col]]).ravel()

    return df, display_data, []

def analyze_contributing_factors(cleaned_row):
    """
    Identifies elevated risk factors (scores above clinical threshold) for explanation.
    """
    elevated = []
    for feat, val in cleaned_row.items():
        if feat in ['Age', 'Gender', 'Level']:
            continue
        max_val = FEATURE_RANGES[feat][1]
        threshold = max_val * 0.65  # Consider top 35% severity as elevated
        if val >= threshold:
            severity = "High" if val >= max_val * 0.85 else "Moderate"
            elevated.append({
                'feature': feat,
                'label': FEATURE_LABELS.get(feat, feat),
                'value': int(val),
                'max': max_val,
                'severity': severity
            })
    # Sort by highest relative severity
    elevated.sort(key=lambda x: x['value'] / x['max'], reverse=True)
    return elevated

# Routes
@app.route('/')
def home():
    """Landing Page"""
    return render_template('index.html', metadata=metadata)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction intake form and result generation"""
    if request.method == 'GET':
        return render_template(
            'predict.html',
            feature_ranges=FEATURE_RANGES,
            feature_labels=FEATURE_LABELS,
            metadata=metadata
        )

    # POST: Process form submission
    if model is None or scaler is None:
        load_artifacts()
        if model is None:
            return render_template('error.html', message="Machine learning model is currently unavailable. Please check model files."), 500

    form_dict = request.form.to_dict()
    df_features, display_data, errors = validate_and_preprocess(form_dict)

    if errors:
        return render_template(
            'predict.html',
            errors=errors,
            previous_values=form_dict,
            feature_ranges=FEATURE_RANGES,
            feature_labels=FEATURE_LABELS,
            metadata=metadata
        ), 400

    try:
        # Scale features using exact fitted MinMaxScaler from training
        scaled_features = scaler.transform(df_features)

        # Predict using Random Forest Classifier
        pred_class = int(model.predict(scaled_features)[0])
        
        # Calculate probabilities
        probabilities = model.predict_proba(scaled_features)[0]
        prob_low = float(probabilities[0])
        prob_high = float(probabilities[1])

        # Risk score calculation
        risk_percentage = round(prob_high * 100, 1)

        # Risk category interpretation
        if pred_class == 1:
            result_title = "Elevated Risk Detected"
            result_badge = "High Risk (Positive)"
            result_status = "danger"
            summary_message = "The machine learning algorithm indicates patterns consistent with higher lung cancer risk factors based on the clinical indicators provided."
        else:
            result_title = "Low Risk Profile"
            result_badge = "Low Risk (Negative)"
            result_status = "success"
            summary_message = "The machine learning algorithm indicates low probability of lung cancer risk based on the clinical indicators provided."

        contributing_factors = analyze_contributing_factors(df_features.iloc[0].to_dict())

        return render_template(
            'result.html',
            prediction=pred_class,
            result_title=result_title,
            result_badge=result_badge,
            result_status=result_status,
            risk_percentage=risk_percentage,
            prob_low=round(prob_low * 100, 1),
            prob_high=round(prob_high * 100, 1),
            summary_message=summary_message,
            submitted_data=display_data,
            feature_labels=FEATURE_LABELS,
            contributing_factors=contributing_factors,
            metadata=metadata
        )

    except Exception as e:
        app.logger.error(f"Prediction execution error: {e}", exc_info=True)
        return render_template('error.html', message=f"An error occurred during prediction: {str(e)}"), 500

@app.route('/about')
def about():
    """Information page detailing model architecture, dataset, and benchmarks"""
    return render_template('about.html', metadata=metadata)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON API endpoint for programmatic predictions"""
    if not request.is_json:
        return jsonify({'success': False, 'error': 'Request must be in JSON format'}), 400

    if model is None or scaler is None:
        load_artifacts()
        if model is None:
            return jsonify({'success': False, 'error': 'Model artifacts not loaded'}), 500

    data = request.get_json()
    df_features, _, errors = validate_and_preprocess(data)

    if errors:
        return jsonify({'success': False, 'errors': errors}), 400

    try:
        scaled = scaler.transform(df_features)
        pred_class = int(model.predict(scaled)[0])
        probabilities = model.predict_proba(scaled)[0]
        
        return jsonify({
            'success': True,
            'prediction': pred_class,
            'label': 'High Risk (Positive)' if pred_class == 1 else 'Low Risk (Negative)',
            'risk_probability': round(float(probabilities[1]), 4),
            'risk_percentage': round(float(probabilities[1]) * 100, 1),
            'class_probabilities': {
                'low_risk': round(float(probabilities[0]), 4),
                'high_risk': round(float(probabilities[1]), 4)
            },
            'model': 'Random Forest Classifier (50 estimators)'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', message="The requested medical diagnostic page could not be found.", error_code=404), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('error.html', message="An internal server error occurred while processing the medical data.", error_code=500), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
