from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import pickle
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
import uvicorn
import os
from datetime import datetime
import time
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка модели
try:
    with open("models/wine_model.joblib", "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded")
except Exception as e:
    logger.error(f"Model not loaded: {e}")
    model = None

try:
    with open("models/scaler.joblib", "rb") as f:
        scaler = pickle.load(f)
    logger.info("Scaler loaded")
except:
    scaler = None

app = FastAPI(title="Wine Quality Predictor")

# Модель входных данных
class WineFeatures(BaseModel):
    fixed_acidity: float = Field(..., ge=3.0, le=15.0)
    volatile_acidity: float = Field(..., ge=0.0, le=1.5)
    citric_acid: float = Field(..., ge=0.0, le=1.5)
    residual_sugar: float = Field(..., ge=0.0, le=20.0)
    chlorides: float = Field(..., ge=0.0, le=0.5)
    free_sulfur_dioxide: float = Field(..., ge=0.0, le=300.0)
    total_sulfur_dioxide: float = Field(..., ge=0.0, le=500.0)
    density: float = Field(..., ge=0.98, le=1.05)
    ph: float = Field(..., ge=2.5, le=4.5)
    sulphates: float = Field(..., ge=0.0, le=2.0)

# Функция предобработки
def preprocess(df):
    df['acid_ratio'] = df['fixed_acidity'] / (df['volatile_acidity'] + 0.01)
    df['sulfur_ratio'] = df['free_sulfur_dioxide'] / (df['total_sulfur_dioxide'] + 0.01)
    df['total_acidity'] = df['fixed_acidity'] + df['volatile_acidity']
    df['log_residual_sugar'] = np.log(df['residual_sugar'] + 1)
    df['log_chlorides'] = np.log(df['chlorides'] + 0.001)
    
    features = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
                'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
                'ph', 'sulphates', 'acid_ratio', 'sulfur_ratio', 'total_acidity',
                'log_residual_sugar', 'log_chlorides']
    return df[features]

# Функция для работы с БД
def get_db_connection():
    return psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=os.getenv('DB_PORT', '5432'),
        database=os.getenv('DB_NAME', 'wine_predictions'),
        user=os.getenv('DB_USER', 'wine_user'),
        password=os.getenv('DB_PASSWORD', 'wine_password')
    )

# Создание таблицы при запуске
def init_db():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                fixed_acidity FLOAT,
                volatile_acidity FLOAT,
                citric_acid FLOAT,
                residual_sugar FLOAT,
                chlorides FLOAT,
                free_sulfur_dioxide FLOAT,
                total_sulfur_dioxide FLOAT,
                density FLOAT,
                ph FLOAT,
                sulphates FLOAT,
                predicted_quality FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        cur.close()
        conn.close()
        logger.info("Database initialized")
    except Exception as e:
        logger.warning(f"DB init failed: {e}")

# Инициализация БД при старте
@app.on_event("startup")
def startup():
    init_db()

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
async def predict(request: Request, wine: WineFeatures):
    start_time = time.time()
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Предсказание
        input_df = pd.DataFrame([wine.dict()])
        processed = preprocess(input_df)
        
        if scaler:
            processed = pd.DataFrame(scaler.transform(processed), columns=processed.columns)
        
        quality = float(model.predict(processed)[0])
        quality = round(quality, 2)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Сохранение в БД
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO predictions (
                    fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                    chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
                    ph, sulphates, predicted_quality
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                wine.fixed_acidity, wine.volatile_acidity, wine.citric_acid,
                wine.residual_sugar, wine.chlorides, wine.free_sulfur_dioxide,
                wine.total_sulfur_dioxide, wine.density, wine.ph, wine.sulphates,
                quality
            ))
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to save to DB: {e}")
        
        return {
            "predicted_quality": quality,
            "processing_time_ms": round(processing_time, 2)
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/history")
async def get_history(limit: int = 10):
    """Получить последние предсказания из БД"""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT id, predicted_quality, created_at 
            FROM predictions 
            ORDER BY created_at DESC 
            LIMIT %s
        """, (limit,))
        results = cur.fetchall()
        cur.close()
        conn.close()
        return {"history": results}
    except Exception as e:
        return {"error": str(e), "history": []}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)