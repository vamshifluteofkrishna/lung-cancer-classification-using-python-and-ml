"""
End-to-End Automated Test Suite for Lung Cancer Prediction Application
Tests model inference, preprocessing fidelity, Flask routes, API endpoints, and error handling.
"""

import unittest
import json
from app import app, load_artifacts, FEATURE_ORDER

class LungCancerPredictionTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        load_artifacts()
        cls.client = app.test_client()
        cls.client.testing = True

        cls.low_risk_payload = {
            'Age': '27',
            'Gender': 'Male',
            'Level': 'Low',
            'Air Pollution': '3',
            'Alcohol use': '1',
            'Dust Allergy': '4',
            'chronic Lung Disease': '3',
            'Balanced Diet': '4',
            'Obesity': '3',
            'Smoking': '1',
            'Passive Smoker': '4',
            'Chest Pain': '3',
            'Coughing of Blood': '1',
            'Fatigue': '3',
            'Weight Loss': '2',
            'Shortness of Breath': '2',
            'Wheezing': '4',
            'Swallowing Difficulty': '2',
            'Clubbing of Finger Nails': '2',
            'Frequent Cold': '3',
            'Dry Cough': '4',
            'Snoring': '3'
        }

        cls.high_risk_payload = {
            'Age': '52',
            'Gender': 'Female',
            'Level': 'High',
            'Air Pollution': '7',
            'Alcohol use': '7',
            'Dust Allergy': '7',
            'chronic Lung Disease': '6',
            'Balanced Diet': '2',
            'Obesity': '6',
            'Smoking': '8',
            'Passive Smoker': '7',
            'Chest Pain': '8',
            'Coughing of Blood': '8',
            'Fatigue': '8',
            'Weight Loss': '7',
            'Shortness of Breath': '8',
            'Wheezing': '7',
            'Swallowing Difficulty': '6',
            'Clubbing of Finger Nails': '8',
            'Frequent Cold': '6',
            'Dry Cough': '6',
            'Snoring': '5'
        }

    def test_01_landing_page(self):
        """Test GET / renders successfully"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Early Risk Detection with', response.data)
        self.assertIn(b'Start Risk Assessment', response.data)

    def test_02_predict_form_page(self):
        """Test GET /predict renders form with all sections"""
        response = self.client.get('/predict')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Patient Assessment', response.data)
        self.assertIn(b'Risk Screening', response.data)
        self.assertIn(b'Coughing of Blood (Hemoptysis)', response.data)

    def test_03_about_page(self):
        """Test GET /about renders benchmark tables and documentation"""
        response = self.client.get('/about')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Algorithm Benchmarks Comparison', response.data)
        self.assertIn(b'Random Forest Classifier', response.data)

    def test_04_404_error_handler(self):
        """Test 404 page renders friendly custom error"""
        response = self.client.get('/page-that-does-not-exist')
        self.assertEqual(response.status_code, 404)
        self.assertIn(b'Diagnostic Processing Notice', response.data)

    def test_05_predict_low_risk_submission(self):
        """Test POST /predict with low-risk input"""
        response = self.client.post('/predict', data=self.low_risk_payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Low Risk Profile', response.data)
        self.assertIn(b'Low Risk (Negative)', response.data)
        self.assertIn(b'Estimated Risk', response.data)

    def test_06_predict_high_risk_submission(self):
        """Test POST /predict with high-risk input"""
        response = self.client.post('/predict', data=self.high_risk_payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Elevated Risk Detected', response.data)
        self.assertIn(b'High Risk (Positive)', response.data)
        self.assertIn(b'Elevated Clinical', response.data)

    def test_07_validation_missing_field(self):
        """Test POST /predict with missing required fields returns 400 and error alert"""
        incomplete_payload = self.low_risk_payload.copy()
        del incomplete_payload['Age']
        del incomplete_payload['Smoking']

        response = self.client.post('/predict', data=incomplete_payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn(b'Missing required field', response.data)

    def test_08_api_predict_json(self):
        """Test POST /api/predict returns valid JSON structure"""
        response = self.client.post(
            '/api/predict',
            data=json.dumps(self.high_risk_payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertEqual(data['prediction'], 1)
        self.assertIn('risk_probability', data)
        self.assertIn('class_probabilities', data)

if __name__ == '__main__':
    unittest.main()
