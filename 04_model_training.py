import pandas as pd
import numpy as np
import os
import joblib
import warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import xgboost as xgb

warnings.filterwarnings('ignore')

print("🚀 Iniciando Entrenamiento de Modelos...")

# 🔧 Parámetros
FEATURES = [
    'log_return', 'pct_change', 'sma_20', 'stddev_20',
    'bb_upper', 'bb_lower', 'bb_width',
    'ema_12', 'ema_26', 'macd', 'macd_signal', 'macd_histogram',
    'rsi', 'adx', 'atr',
    'returns_mean_5', 'returns_std_5', 'returns_mean_20', 'returns_std_20',
    'ema_diff', 'rsi_macd_ratio',
    'bb_position', 'rsi_overbought', 'rsi_oversold', 'macd_cross',
    'volatility', 'vol_ratio'
]

TARGET = 'target'

# 📥 Cargar el dataset con features
df = pd.read_csv('data/BTC_5m_features.csv')

print(f"✅ Dataset cargado: {df.shape[0]} registros y {df.shape[1]} columnas")

# Separar features y target
X = df[FEATURES]
y = df[TARGET]

# 🟦 Dividir dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

print(f"🟢 X_train shape: {X_train.shape}")
print(f"🟢 X_test shape: {X_test.shape}")

# ==========================================
# 📦 ENTRENAMIENTO XGBoost
# ==========================================
print("🎯 Entrenando modelo XGBoost...")

model_xgb = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    verbosity=0
)

model_xgb.fit(X_train, y_train)

# 🚀 Evaluación XGBoost
y_pred_xgb = model_xgb.predict(X_test)

acc_xgb = accuracy_score(y_test, y_pred_xgb)
precision_xgb = precision_score(y_test, y_pred_xgb)
recall_xgb = recall_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb)

print(f"✅ XGBoost Accuracy: {acc_xgb:.4f}")
print(f"✅ XGBoost Precision: {precision_xgb:.4f}")
print(f"✅ XGBoost Recall: {recall_xgb:.4f}")
print(f"✅ XGBoost F1 Score: {f1_xgb:.4f}")

# ==========================================
# 📦 ENTRENAMIENTO LSTM
# ==========================================
print("🎯 Entrenando modelo LSTM...")

# LSTM requiere un input 3D: [samples, timesteps, features]
timesteps = 1
X_train_lstm = np.reshape(X_train.values, (X_train.shape[0], timesteps, X_train.shape[1]))
X_test_lstm = np.reshape(X_test.values, (X_test.shape[0], timesteps, X_test.shape[1]))

model_lstm = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(timesteps, X_train.shape[1])),
    Dropout(0.2),
    LSTM(units=50),
    Dropout(0.2),
    Dense(units=1, activation='sigmoid')
])

model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_lstm.fit(
    X_train_lstm, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# 🚀 Evaluación LSTM
y_pred_lstm_proba = model_lstm.predict(X_test_lstm)
y_pred_lstm = (y_pred_lstm_proba > 0.5).astype(int)

acc_lstm = accuracy_score(y_test, y_pred_lstm)
precision_lstm = precision_score(y_test, y_pred_lstm)
recall_lstm = recall_score(y_test, y_pred_lstm)
f1_lstm = f1_score(y_test, y_pred_lstm)

print(f"✅ LSTM Accuracy: {acc_lstm:.4f}")
print(f"✅ LSTM Precision: {precision_lstm:.4f}")
print(f"✅ LSTM Recall: {recall_lstm:.4f}")
print(f"✅ LSTM F1 Score: {f1_lstm:.4f}")

# ==========================================
# 💾 GUARDAR MODELOS
# ==========================================
output_dir = 'models'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Guardar XGBoost
joblib.dump(model_xgb, os.path.join(output_dir, 'xgb_btc_model.pkl'))
print(f"💾 XGBoost model guardado en {output_dir}/xgb_btc_model.pkl")

# Guardar LSTM
model_lstm.save(os.path.join(output_dir, 'lstm_btc_model.h5'))
print(f"💾 LSTM model guardado en {output_dir}/lstm_btc_model.h5")

# ==========================================
# ✅ FINALIZADO
# ==========================================
print("✅ Entrenamiento de modelos completado")
