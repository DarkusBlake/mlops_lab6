import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path

# Загрузка данных - автоопределение разделителя
print("Loading data...")

# Пробуем разные разделители
try:
    # Сначала пробуем точку с запятой (стандарт для этого датасета)
    df = pd.read_csv('winequality-white.csv', sep=';')
    if df.shape[1] > 1:
        print("Using separator: ';'")
except:
    pass

# Если получили 1 колонку - пробуем запятую
if 'df' not in locals() or df.shape[1] == 1:
    df = pd.read_csv('winequality-white.csv', sep=',')
    print("Using separator: ','")

# Если всё ещё 1 колонка - читаем первую строку и определяем разделитель
if df.shape[1] == 1:
    with open('winequality-white.csv', 'r') as f:
        first_line = f.readline()
        if ';' in first_line:
            df = pd.read_csv('winequality-white.csv', sep=';')
        elif ',' in first_line:
            df = pd.read_csv('winequality-white.csv', sep=',')
        else:
            df = pd.read_csv('winequality-white.csv', sep='\t')
    print(f"Auto-detected separator")

print(f"Data loaded: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Проверка наличия колонки quality (она может быть последней)
if 'quality' not in df.columns:
    # Возможно колонки слились в одну
    if len(df.columns) == 1:
        # Разделяем единственную колонку по запятой
        col_name = df.columns[0]
        # Разбиваем строку на отдельные значения
        data = df[col_name].str.split(',', expand=True)
        # Переименовываем колонки
        expected_cols = ['fixed acidity', 'volatile acidity', 'citric acid', 
                        'residual sugar', 'chlorides', 'free sulfur dioxide', 
                        'total sulfur dioxide', 'density', 'pH', 'sulphates', 
                        'alcohol', 'quality']
        if data.shape[1] == len(expected_cols):
            data.columns = expected_cols
            # Конвертируем в числа
            for col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            df = data
            print("Successfully split merged column")

print(f"Final columns: {df.columns.tolist()}")
print(f"First row: {df.iloc[0].to_dict()}")

# Проверка наличия quality
if 'quality' not in df.columns:
    print("ERROR: 'quality' column still not found!")
    print(f"Available columns: {df.columns.tolist()}")
    exit(1)

# Подготовка признаков
X = df.drop('quality', axis=1)
y = df['quality']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Quality range: {y.min()} - {y.max()}")

# Переименование колонок (убираем пробелы)
X.columns = [col.replace(' ', '_') for col in X.columns]

# Добавление новых признаков
print("Creating additional features...")
X['acid_ratio'] = X['fixed_acidity'] / (X['volatile_acidity'] + 0.01)
X['sulfur_ratio'] = X['free_sulfur_dioxide'] / (X['total_sulfur_dioxide'] + 0.01)
X['total_acidity'] = X['fixed_acidity'] + X['volatile_acidity']
X['log_residual_sugar'] = np.log(X['residual_sugar'] + 1)
X['log_chlorides'] = np.log(X['chlorides'] + 0.001)

print(f"Final features: {X.shape[1]}")

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабирование
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Обучение модели
print("Training model...")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)

# Оценка
score = model.score(X_test_scaled, y_test)
print(f"Model R2 score: {score:.4f}")

# Сохранение модели
Path('models').mkdir(exist_ok=True)
with open('models/wine_model.joblib', 'wb') as f:
    pickle.dump(model, f)
with open('models/scaler.joblib', 'wb') as f:
    pickle.dump(scaler, f)

print("Model saved to models/")

# Тестовое предсказание
sample = X_test.iloc[0:1]
sample_scaled = scaler.transform(sample)
pred = model.predict(sample_scaled)[0]
print(f"Sample prediction: {pred:.2f} (actual: {y_test.iloc[0]})")
