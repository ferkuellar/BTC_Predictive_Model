import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime
import os

# 📂 Crear carpeta de resultados si no existe
os.makedirs('results', exist_ok=True)

# 🔧 Parámetros iniciales
initial_balance = 10000
balance = initial_balance
position = 0
equity_curve = []
buy_price = 0

# ✅ Cargar datasets
file_prices = 'data/BTC_5m_preprocessed.csv'
file_features = 'data/BTC_5m_features_nosmote.csv'

print("🚀 Iniciando Backtesting...")

# ✅ Cargar precios
df_prices = pd.read_csv(file_prices)
df_prices['timestamp'] = pd.to_datetime(df_prices['timestamp'])

# ✅ Cargar features sin SMOTE
df_features = pd.read_csv(file_features)

# 🔎 Revisar y asegurar columnas consistentes
expected_features = [
    'log_return', 'pct_change', 'sma_20', 'stddev_20', 'bb_upper', 'bb_lower',
    'bb_width', 'ema_12', 'ema_26', 'macd', 'macd_signal', 'macd_histogram',
    'rsi', 'adx', 'atr', 'returns_mean_5', 'returns_std_5',
    'returns_mean_20', 'returns_std_20', 'ema_diff', 'rsi_macd_ratio',
    'bb_position', 'rsi_overbought', 'rsi_oversold', 'macd_cross',
    'volatility', 'vol_ratio'
]

missing_cols = [col for col in expected_features if col not in df_features.columns]
if missing_cols:
    raise ValueError(f"❌ Faltan las siguientes columnas en el dataset de features: {missing_cols}")

# ✅ Seleccionar y ordenar las columnas
X = df_features[expected_features].copy()

# ✅ Cargar modelos
print("🟢 Cargando modelos...")
model_xgb = joblib.load('models/xgb_btc_model.pkl')
model_lstm = load_model('models/lstm_btc_model.h5')

# ✅ Validación de features
print("✅ Features usadas en backtesting:", X.columns.tolist())

# 🚀 Generar predicciones
print("🎯 Generando predicciones...")

# 📍 XGBoost predict
xgb_preds = model_xgb.predict(X)

# 📍 LSTM necesita 3D -> (samples, timesteps=1, features)
X_lstm = X.values.reshape((X.shape[0], 1, X.shape[1]))

# 🧠 LSTM predict
lstm_raw_preds = model_lstm.predict(X_lstm, verbose=0)
lstm_preds = (lstm_raw_preds > 0.5).astype(int).flatten()

# 🧠 Ensemble simple: promedio de señales (puedes cambiarlo a lógica de mayoría si quieres)
ensemble_preds = (xgb_preds + lstm_preds) / 2
ensemble_preds = (ensemble_preds > 0.5).astype(int)

# ✅ Sincronizar datos de precios
if len(df_prices) != len(ensemble_preds):
    print(f"🔧 Sincronizando precios: {len(df_prices)} registros, predicciones: {len(ensemble_preds)}")
    df_prices_sync = df_prices.iloc[-len(ensemble_preds):].copy()
else:
    df_prices_sync = df_prices.copy()

df_prices_sync['prediction'] = ensemble_preds

# ⚙️ Variables para el loop de backtest
balance_history = []
position_size = 0
buy_price = 0
in_position = False

# 🔁 Backtesting Loop
for i in range(len(df_prices_sync)):
    row = df_prices_sync.iloc[i]
    price_now = row['close']
    signal = row['prediction']

    if signal == 1 and not in_position:
        # ✅ Comprar
        position_size = balance / price_now
        buy_price = price_now
        balance = 0
        in_position = True
        print(f"[{row['timestamp']}] 🟢 BUY @ {price_now:.2f}")

    elif signal == 0 and in_position:
        # ✅ Vender
        balance = position_size * price_now
        position_size = 0
        in_position = False
        print(f"[{row['timestamp']}] 🔴 SELL @ {price_now:.2f} -> Balance: {balance:.2f}")

    # ✅ Equity = balance disponible + valor actual si hay posición
    current_equity = balance if not in_position else position_size * price_now
    equity_curve.append(current_equity)

    balance_history.append({
        'timestamp': row['timestamp'],
        'equity': current_equity,
        'price': price_now,
        'signal': signal
    })

# ✅ Guardar resultados en CSV
df_results = pd.DataFrame(balance_history)
df_results['timestamp'] = pd.to_datetime(df_results['timestamp'])
df_results.to_csv('results/equity_curve.csv', index=False)

# 📈 Métricas básicas
total_return = (equity_curve[-1] - initial_balance) / initial_balance * 100
max_equity = np.max(equity_curve)
min_equity = np.min(equity_curve)
max_drawdown = (max_equity - min_equity) / max_equity * 100

# ✅ Print resumen
print("\n✅ Backtesting finalizado")
print(f"🔹 Initial Balance: ${initial_balance:.2f}")
print(f"🔹 Final Equity: ${equity_curve[-1]:.2f}")
print(f"🔹 Total Return: {total_return:.2f}%")
print(f"🔹 Max Drawdown: {max_drawdown:.2f}%")
print("✅ Resultados guardados en 'results/equity_curve.csv'")
