import pytest
import pickle
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from main import app, preprocess

client = TestClient(app)

# Загрузка модели для тестов
try:
    with open("models/wine_model.joblib", "rb") as f:
        model = pickle.load(f)
    with open("models/scaler.joblib", "rb") as f:
        scaler = pickle.load(f)
    MODEL_LOADED = True
except:
    MODEL_LOADED = False

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
    "sulphates": 0.45
}

class TestModel:
    """Тесты модели машинного обучения"""
    
    def test_model_exists(self):
        """Проверка что модель загружена"""
        assert MODEL_LOADED, "Model not loaded. Run train_model.py first"
    
    def test_model_prediction_range(self):
        """Проверка что предсказания модели в допустимом диапазоне (3-9)"""
        input_df = pd.DataFrame([SAMPLE_WINE])
        processed = preprocess(input_df)
        processed_scaled = scaler.transform(processed)
        prediction = model.predict(processed_scaled)[0]
        
        # Качество вина обычно от 3 до 9
        assert 3 <= prediction <= 9, f"Prediction {prediction} out of range"
    
    def test_model_consistent_predictions(self):
        """Проверка что модель дает одинаковые предсказания для одинаковых входов"""
        input_df = pd.DataFrame([SAMPLE_WINE])
        processed = preprocess(input_df)
        processed_scaled = scaler.transform(processed)
        
        pred1 = model.predict(processed_scaled)[0]
        pred2 = model.predict(processed_scaled)[0]
        
        assert pred1 == pred2, "Model gives inconsistent predictions"
    
    def test_model_handles_boundary_values(self):
        """Проверка работы модели с граничными значениями"""
        boundary_wine = {
            "fixed_acidity": 3.0,  # минимальное
            "volatile_acidity": 0.01,
            "citric_acid": 0.01,
            "residual_sugar": 0.1,
            "chlorides": 0.01,
            "free_sulfur_dioxide": 5.0,
            "total_sulfur_dioxide": 10.0,
            "density": 0.98,
            "ph": 2.5,
            "sulphates": 0.1
        }
        
        input_df = pd.DataFrame([boundary_wine])
        processed = preprocess(input_df)
        processed_scaled = scaler.transform(processed)
        prediction = model.predict(processed_scaled)[0]
        
        # Должно вернуть число, не упасть с ошибкой
        assert isinstance(prediction, (int, float))
    
    def test_preprocessing_creates_correct_features(self):
        """Проверка что предобработка создает правильные признаки"""
        input_df = pd.DataFrame([SAMPLE_WINE])
        processed = preprocess(input_df)
        
        expected_features = [
            'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
            'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
            'ph', 'sulphates', 'acid_ratio', 'sulfur_ratio', 'total_acidity',
            'log_residual_sugar', 'log_chlorides'
        ]
        
        for feature in expected_features:
            assert feature in processed.columns, f"Missing feature: {feature}"
    
    def test_feature_engineering_formulas(self):
        """Проверка правильности формул новых признаков"""
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
            "sulphates": 0.45
        }])
        
        processed = preprocess(test_data)
        
        # Проверка acid_ratio
        expected_acid_ratio = 7.0 / (0.27 + 0.01)
        assert abs(processed['acid_ratio'].iloc[0] - expected_acid_ratio) < 0.01
        
        # Проверка sulfur_ratio
        expected_sulfur_ratio = 45.0 / (170.0 + 0.01)
        assert abs(processed['sulfur_ratio'].iloc[0] - expected_sulfur_ratio) < 0.01
        
        # Проверка total_acidity
        expected_total_acidity = 7.0 + 0.27
        assert abs(processed['total_acidity'].iloc[0] - expected_total_acidity) < 0.01

class TestAPI:
    """Тесты API эндпоинтов"""
    
    def test_health_endpoint(self):
        """Проверка health check"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
    
    def test_predict_endpoint_valid(self):
        """Проверка предсказания с валидными данными"""
        response = client.post("/predict", json=SAMPLE_WINE)
        assert response.status_code == 200
        
        data = response.json()
        assert "predicted_quality" in data
        assert isinstance(data["predicted_quality"], (int, float))
        assert 3 <= data["predicted_quality"] <= 9
    
    def test_predict_endpoint_invalid(self):
        """Проверка предсказания с невалидными данными"""
        invalid_data = {"fixed_acidity": 100.0}  # неполные данные
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422
    
    def test_predict_with_missing_field(self):
        """Проверка с отсутствующим полем"""
        incomplete = SAMPLE_WINE.copy()
        del incomplete["ph"]
        response = client.post("/predict", json=incomplete)
        assert response.status_code == 422
    
    def test_history_endpoint(self):
        """Проверка получения истории"""
        response = client.get("/history")
        assert response.status_code == 200
        assert "history" in response.json()

class TestPredictionQuality:
    """Тесты качества предсказаний модели"""
    
    def test_high_quality_wine_prediction(self):
        """Проверка что хорошее вино получает высокую оценку"""
        # Параметры качественного вина
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
            "sulphates": 0.60
        }
        
        input_df = pd.DataFrame([high_quality])
        processed = preprocess(input_df)
        processed_scaled = scaler.transform(processed)
        prediction = model.predict(processed_scaled)[0]
        
        # Хорошее вино должно иметь оценку выше 6
        assert prediction > 6.0, f"High quality wine got low score: {prediction}"
    
    def test_low_quality_wine_prediction(self):
        """Проверка что плохое вино получает низкую оценку"""
        # Параметры некачественного вина
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
            "sulphates": 0.30
        }
        
        input_df = pd.DataFrame([low_quality])
        processed = preprocess(input_df)
        processed_scaled = scaler.transform(processed)
        prediction = model.predict(processed_scaled)[0]
        
        # Плохое вино должно иметь оценку ниже 6
        assert prediction < 6.0, f"Low quality wine got high score: {prediction}"
    
    def test_response_time(self):
        """Проверка времени ответа API"""
        import time
        
        start = time.time()
        response = client.post("/predict", json=SAMPLE_WINE)
        elapsed = time.time() - start
        
        assert response.status_code == 200
        assert elapsed < 1.0, f"Response too slow: {elapsed:.2f}s"
