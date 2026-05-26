import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path

# Загрузка данных
df = pd.read_csv('winequality-white new.csv', sep=';')
print(f"Data loaded: {df.shape}")

# Подготовка признаков
X = df.drop('quality', axis=1)
y = df['quality']

# Переименование колонок
X.columns = [col.replace(' ', '_') for col in X.columns]

# Добавление признаков
X['acid_ratio'] = X['fixed_acidity'] / (X['volatile_acidity'] + 0.01)
X['sulfur_ratio'] = X['free_sulfur_dioxide'] / (X['total_sulfur_dioxide'] + 0.01)
X['total_acidity'] = X['fixed_acidity'] + X['volatile_acidity']
X['log_residual_sugar'] = np.log(X['residual_sugar'] + 1)
X['log_chlorides'] = np.log(X['chlorides'] + 0.001)

# Разделение
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабирование
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Обучение
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Оценка
score = model.score(X_test_scaled, y_test)
print(f"Model R2 score: {score:.4f}")

# Сохранение
Path('models').mkdir(exist_ok=True)
with open('models/wine_model.joblib', 'wb') as f:
    pickle.dump(model, f)
with open('models/scaler.joblib', 'wb') as f:
    pickle.dump(scaler, f)

print("Model saved to models/")