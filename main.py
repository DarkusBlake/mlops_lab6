from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import pickle
import pandas as pd
import logging
import uvicorn
from sklearn.preprocessing import StandardScaler
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка модели
try:
    with open("wine_model.joblib", "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

# Загрузка скейлера (если использовали при обучении)
try:
    with open("scaler.joblib", "rb") as f:
        scaler = pickle.load(f)
    logger.info("Scaler loaded successfully")
except:
    scaler = None
    logger.warning("No scaler found, using raw features")

app = FastAPI(
    title="Wine Quality Predictor",
    description="Predicts quality of white wine based on physicochemical properties",
    version="1.0.0"
)

# Модель входных данных
class WineFeatures(BaseModel):
    fixed_acidity: float = Field(..., ge=3.0, le=15.0, description="Fixed acidity (g/dm^3)")
    volatile_acidity: float = Field(..., ge=0.0, le=1.5, description="Volatile acidity (g/dm^3)")
    citric_acid: float = Field(..., ge=0.0, le=1.5, description="Citric acid (g/dm^3)")
    residual_sugar: float = Field(..., ge=0.0, le=20.0, description="Residual sugar (g/dm^3)")
    chlorides: float = Field(..., ge=0.0, le=0.5, description="Chlorides (g/dm^3)")
    free_sulfur_dioxide: float = Field(..., ge=0.0, le=300.0, description="Free sulfur dioxide (mg/dm^3)")
    total_sulfur_dioxide: float = Field(..., ge=0.0, le=500.0, description="Total sulfur dioxide (mg/dm^3)")
    density: float = Field(..., ge=0.98, le=1.05, description="Density (g/cm^3)")
    pH: float = Field(..., ge=2.5, le=4.5, description="pH level")
    sulphates: float = Field(..., ge=0.0, le=2.0, description="Sulphates (g/dm^3)")
    
    @validator('volatile_acidity')
    def validate_volatile_acidity(cls, v):
        if v > 1.2:
            raise ValueError('Volatile acidity too high, wine might be spoiled')
        return v

    class Config:
        schema_extra = {
            "example": {
                "fixed_acidity": 7.0,
                "volatile_acidity": 0.27,
                "citric_acid": 0.36,
                "residual_sugar": 20.7,
                "chlorides": 0.045,
                "free_sulfur_dioxide": 45.0,
                "total_sulfur_dioxide": 170.0,
                "density": 1.001,
                "pH": 3.0,
                "sulphates": 0.45
            }
        }

# Модель ответа
class QualityPrediction(BaseModel):
    predicted_quality: float
    quality_class: str
    confidence_score: Optional[float] = None

def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Предобработка признаков для модели
    """
    # Создаем дополнительные признаки (feature engineering)
    df['acid_ratio'] = df['fixed_acidity'] / (df['volatile_acidity'] + 0.01)
    df['sulfur_ratio'] = df['free_sulfur_dioxide'] / (df['total_sulfur_dioxide'] + 0.01)
    df['total_acidity'] = df['fixed_acidity'] + df['volatile_acidity']
    
    # Логарифмическое преобразование для некоторых признаков
    df['log_residual_sugar'] = df['residual_sugar'].apply(lambda x: x + 1).apply(np.log)
    df['log_chlorides'] = df['chlorides'].apply(lambda x: x + 0.001).apply(np.log)
    
    # Выбираем финальные признаки
    feature_columns = [
        'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
        'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
        'pH', 'sulphates', 'acid_ratio', 'sulfur_ratio', 'total_acidity',
        'log_residual_sugar', 'log_chlorides'
    ]
    
    return df[feature_columns]

def get_quality_class(quality_score: float) -> str:
    """Преобразует числовую оценку в категорию качества"""
    if quality_score >= 7.5:
        return "Excellent"
    elif quality_score >= 6.0:
        return "Good"
    elif quality_score >= 5.0:
        return "Average"
    elif quality_score >= 4.0:
        return "Below Average"
    else:
        return "Poor"

@app.get("/")
async def root():
    return {
        "message": "Wine Quality Prediction API",
        "model_type": "White Wine Quality Predictor",
        "quality_scale": "0-10 (higher is better)"
    }

@app.post("/predict", response_model=QualityPrediction, summary="Predict wine quality")
async def predict_quality(wine: WineFeatures):
    """
    Предсказывает качество белого вина на основе физико-химических свойств
    
    - **fixed_acidity**: Фиксированная кислотность (г/дм³)
    - **volatile_acidity**: Летучая кислотность (г/дм³)
    - **citric_acid**: Лимонная кислота (г/дм³)
    - **residual_sugar**: Остаточный сахар (г/дм³)
    - **chlorides**: Хлориды (г/дм³)
    - **free_sulfur_dioxide**: Свободный диоксид серы (мг/дм³)
    - **total_sulfur_dioxide**: Общий диоксид серы (мг/дм³)
    - **density**: Плотность (г/см³)
    - **pH**: Уровень pH
    - **sulphates**: Сульфаты (г/дм³)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Преобразуем входные данные в DataFrame
        input_data = pd.DataFrame([wine.dict()])
        
        # Предобработка
        processed_data = preprocess_features(input_data)
        
        # Масштабирование если нужно
        if scaler:
            processed_data = pd.DataFrame(
                scaler.transform(processed_data),
                columns=processed_data.columns
            )
        
        # Предсказание
        quality_score = model.predict(processed_data)[0]
        quality_score = round(float(quality_score), 2)
        
        # Получаем категорию качества
        quality_class = get_quality_class(quality_score)
        
        # Простейшая оценка уверенности (на основе близости к границам классов)
        confidence = 1.0 - min(abs(quality_score - 5.5), abs(quality_score - 6.5)) / 3.0
        confidence = round(min(max(confidence, 0.5), 0.98), 2)
        
        logger.info(f"Predicted quality: {quality_score} ({quality_class})")
        
        return QualityPrediction(
            predicted_quality=quality_score,
            quality_class=quality_class,
            confidence_score=confidence
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_batch", summary="Predict multiple wines")
async def predict_batch(wines: list[WineFeatures]):
    """
    Предсказывает качество для нескольких вин одновременно
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = []
        for wine in wines:
            input_data = pd.DataFrame([wine.dict()])
            processed_data = preprocess_features(input_data)
            
            if scaler:
                processed_data = pd.DataFrame(
                    scaler.transform(processed_data),
                    columns=processed_data.columns
                )
            
            quality_score = model.predict(processed_data)[0]
            quality_score = round(float(quality_score), 2)
            
            results.append({
                "predicted_quality": quality_score,
                "quality_class": get_quality_class(quality_score)
            })
        
        return {"predictions": results}
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)
