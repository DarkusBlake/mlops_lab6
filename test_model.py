import pytest
import pickle
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
import sys
from unittest.mock import MagicMock

try:
    import psycopg2
except ImportError:
    psycopg2 = MagicMock()
    sys.modules['psycopg2'] = psycopg2
    sys.modules['psycopg2.extras'] = MagicMock()

from main import app

client = TestClient(app)

# Загрузка модели
try:
    with open("models/wine_model.joblib", "rb") as f:
        model = pickle.load(f)
    with open("models/scaler.joblib", "rb") as f:
        scaler = pickle.load(f)
    MODEL_LOADED = True
    print(f"Model loaded. Scaler expects {scaler.n_features_in_} features")
except Exception as e:
    MODEL_LOADED = False
    print(f"Model not loaded: {e}")

# Функция preprocess
def preprocess(df):
    df = df.copy()
    df['acid_ratio'] = df['fixed_acidity'] / (df['volatile_acidity'] + 0.01)
    df['sulfur_ratio'] = df['free_sulfur_dioxide'] / (df['total_sulfur_dioxide'] + 0.01)
    df['total_acidity'] = df['fixed_acidity'] + df['volatile_acidity']
    df['log_residual_sugar'] = np.log(df['residual_sugar'] + 1)
    df['log_chlorides'] = np.log(df['chlorides'] + 0.001)
    
    features = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
                'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
                'ph', 'sulphates', 'alcohol', 'acid_ratio', 'sulfur_ratio', 'total_acidity',
                'log_residual_sugar', 'log_chlorides']
    return df[features]

# Тестовые данные
SAMPLE_WINE = {
    "fixed_acidity": 7.0,
    "volatile_acidity": 0.27,
    "citric_acid": 0.36,
    "residual_sugar": 20.7,
    "chlorides": 0.045,
    "free_sulfur_dioxide": 45.0,
    "total_sulfur_dioxide": 170.0,
    "density": 1.001,
    "ph": 3.0,
    "sulphates": 0.45,
    "alcohol": 8.8
}

class TestModel:
    """Тесты модели машинного обучения"""
    
    def test_model_exists(self):
        assert MODEL_LOADED, "Model not loaded. Run train_model.py first"
    
    def test_model_prediction_range(self):
        if not MODEL_LOADED:
            pytest.skip("Model not loaded")
        
        input_df = pd.DataFrame([SAMPLE_WINE])
        processed = preprocess(input_df)
        # Используем .values для обхода проверки имен
        processed_scaled = scaler.transform(processed.values)
        prediction = model.predict(processed_scaled)[0]
        
        assert 3 <= prediction <= 9, f"Prediction {prediction} out of range"
    
    def test_model_consistent_predictions(self):
        if not MODEL_LOADED:
            pytest.skip("Model not loaded")
        
        input_df = pd.DataFrame([SAMPLE_WINE])
        processed = preprocess(input_df)
        processed_scaled = scaler.transform(processed.values)
        
        pred1 = model.predict(processed_scaled)[0]
        pred2 = model.predict(processed_scaled)[0]
        
        assert pred1 == pred2
    
    def test_model_handles_boundary_values(self):
        if not MODEL_LOADED:
            pytest.skip("Model not loaded")
        
        boundary_wine = {
            "fixed_acidity": 3.0,
            "volatile_acidity": 0.01,
            "citric_acid": 0.01,
            "residual_sugar": 0.1,
            "chlorides": 0.01,
            "free_sulfur_dioxide": 5.0,
            "total_sulfur_dioxide": 10.0,
            "density": 0.98,
            "ph": 2.5,
            "sulphates": 0.1,
            "alcohol": 8.0
        }
        
        input_df = pd.DataFrame([boundary_wine])
        processed = preprocess(input_df)
        processed_scaled = scaler.transform(processed.values)
        prediction = model.predict(processed_scaled)[0]
        
        assert isinstance(prediction, (int, float))
        assert 3 <= prediction <= 9
    
    def test_preprocessing_creates_correct_features(self):
        input_df = pd.DataFrame([SAMPLE_WINE])
        processed = preprocess(input_df)
        
        assert len(processed.columns) == 16
        assert 'alcohol' in processed.columns
        assert 'acid_ratio' in processed.columns
    
    def test_feature_engineering_formulas(self):
        test_data = pd.DataFrame([{
            "fixed_acidity": 7.0,
            "volatile_acidity": 0.27,
            "residual_sugar": 20.7,
            "chlorides": 0.045,
            "free_sulfur_dioxide": 45.0,
            "total_sulfur_dioxide": 170.0,
            "citric_acid": 0.36,
            "density": 1.001,
            "ph": 3.0,
            "sulphates": 0.45,
            "alcohol": 8.8
        }])
        
        processed = preprocess(test_data)
        
        expected_acid_ratio = 7.0 / (0.27 + 0.01)
        assert abs(processed['acid_ratio'].iloc[0] - expected_acid_ratio) < 0.01

class TestAPI:
    """Тесты API эндпоинтов"""
    
    def test_health_endpoint(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert "model_loaded" in response.json()
    
    def test_predict_endpoint_valid(self):
        response = client.post("/predict", json=SAMPLE_WINE)
        
        if response.status_code != 200:
            print(f"Response status: {response.status_code}")
            print(f"Response body: {response.text}")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "predicted_quality" in data
        assert isinstance(data["predicted_quality"], (int, float))
    
    def test_predict_endpoint_invalid(self):
        invalid_data = {"fixed_acidity": 100.0}
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422
    
    def test_predict_with_missing_field(self):
        incomplete = SAMPLE_WINE.copy()
        del incomplete["ph"]
        response = client.post("/predict", json=incomplete)
        assert response.status_code == 422
    
    def test_history_endpoint(self):
        response = client.get("/history")
        assert response.status_code == 200
        assert "history" in response.json()

class TestPredictionQuality:
    """Тесты качества предсказаний модели"""
    
    def test_high_quality_wine_prediction(self):
        if not MODEL_LOADED:
            pytest.skip("Model not loaded")
        
        high_quality = {
            "fixed_acidity": 6.5,
            "volatile_acidity": 0.20,
            "citric_acid": 0.50,
            "residual_sugar": 5.0,
            "chlorides": 0.030,
            "free_sulfur_dioxide": 35.0,
            "total_sulfur_dioxide": 120.0,
            "density": 0.990,
            "ph": 3.2,
            "sulphates": 0.60,
            "alcohol": 10.5
        }
        
        input_df = pd.DataFrame([high_quality])
        processed = preprocess(input_df)
        processed_scaled = scaler.transform(processed.values)
        prediction = model.predict(processed_scaled)[0]
        
        assert prediction > 6.0, f"High quality wine got low score: {prediction}"
    
    def test_low_quality_wine_prediction(self):
        if not MODEL_LOADED:
            pytest.skip("Model not loaded")
        
        low_quality = {
            "fixed_acidity": 9.0,
            "volatile_acidity": 0.80,
            "citric_acid": 0.10,
            "residual_sugar": 15.0,
            "chlorides": 0.100,
            "free_sulfur_dioxide": 10.0,
            "total_sulfur_dioxide": 50.0,
            "density": 1.005,
            "ph": 3.8,
            "sulphates": 0.30,
            "alcohol": 8.5
        }
        
        input_df = pd.DataFrame([low_quality])
        processed = preprocess(input_df)
        processed_scaled = scaler.transform(processed.values)
        prediction = model.predict(processed_scaled)[0]
        
        assert prediction < 6.0, f"Low quality wine got high score: {prediction}"
    
    def test_response_time(self):
        import time
        
        start = time.time()
        response = client.post("/predict", json=SAMPLE_WINE)
        elapsed = time.time() - start
        
        assert response.status_code == 200
        assert elapsed < 2.0, f"Response too slow: {elapsed:.2f}s"
