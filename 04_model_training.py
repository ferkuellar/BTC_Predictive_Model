import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import MinMaxScaler

import xgboost as xgb
import joblib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


# Carga el dataset con indicadores técnicos y features derivadas
df = pd.read_csv('data/BTC_5m_features.csv')

# Revisar estructura
df.head()


# Definición de features y target
features = [
    'log_return', 'pct_change',
    'rsi', 'ema_12', 'ema_26',
    'macd', 'macd_signal', 'macd_histogram',
    'bb_upper', 'bb_lower', 'bb_width',
    'returns_mean_5', 'returns_std_5',
    'returns_mean_20', 'returns_std_20',
    'ema_diff',
    'rsi_macd_ratio',
    'bb_position',
    'rsi_overbought', 'rsi_oversold',
    'macd_cross',
    'volatility', 'adx', 'atr', 'vol_ratio'
]

# Features (X) y target (y)
X = df[features]
y = df['target']


# División 80% train / 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Revisión de tamaños
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")


# Definir el modelo XGBoost
model_xgb = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42
)

# Entrenar modelo
model_xgb.fit(X_train, y_train)

# Guardar el modelo
joblib.dump(model_xgb, 'models/xgb_btc_model.pkl')
print("✅ Modelo XGBoost guardado en 'models/xgb_btc_model.pkl'")


# Predicciones en test
y_pred_xgb = model_xgb.predict(X_test)

# Métricas
print("📊 Evaluación del modelo XGBoost:")
print(f"✅ Accuracy:  {accuracy_score(y_test, y_pred_xgb):.2f}")
print(f"✅ Precision: {precision_score(y_test, y_pred_xgb):.2f}")
print(f"✅ Recall:    {recall_score(y_test, y_pred_xgb):.2f}")
print(f"✅ F1 Score:  {f1_score(y_test, y_pred_xgb):.2f}")

# Reporte detallado
print(classification_report(y_test, y_pred_xgb))


# Vamos a usar el 'close' como ejemplo para LSTM
dataset = df[['close']].values

# Escalado de datos
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Crear secuencias para LSTM
sequence_length = 60
X_lstm, y_lstm = [], []

for i in range(sequence_length, len(scaled_data)):
    X_lstm.append(scaled_data[i-sequence_length:i, 0])
    y_lstm.append(1 if scaled_data[i, 0] > scaled_data[i-1, 0] else 0)

X_lstm = np.array(X_lstm)
y_lstm = np.array(y_lstm)

# Reshape para LSTM (samples, timesteps, features)
X_lstm = X_lstm.reshape((X_lstm.shape[0], X_lstm.shape[1], 1))

print(f"X_lstm shape: {X_lstm.shape}")
print(f"y_lstm shape: {y_lstm.shape}")


from tensorflow.keras.models import load_model
# Definición del modelo LSTM
model_lstm = Sequential()
model_lstm.add(LSTM(units=50, return_sequences=True, input_shape=(X_lstm.shape[1], 1)))
model_lstm.add(Dropout(0.2))
model_lstm.add(LSTM(units=50, return_sequences=False))
model_lstm.add(Dropout(0.2))
model_lstm.add(Dense(units=1, activation='sigmoid'))

# Compilación
model_lstm.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping para evitar overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=3)

# Entrenamiento
history = model_lstm.fit(
    X_lstm, y_lstm,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# Guardar el modelo
model_lstm.save('models/lstm_btc_model.keras')
print("✅ Modelo LSTM guardado en 'models/lstm_btc_model.keras")


# Gráficos de la evolución del entrenamiento
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss durante el entrenamiento')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy durante el entrenamiento')
plt.legend()

plt.show()






