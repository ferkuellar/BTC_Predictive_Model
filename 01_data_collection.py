import pandas as pd
import time
import ccxt
from datetime import datetime

# ==============================
# CONFIGURACIÓN
# ==============================
exchange = ccxt.binance({
    'enableRateLimit': True
})

symbol = 'BTC/USDT'
timeframe = '5m'  # 1m, 5m, 15m, 1h, etc.
limit = 1000      # Límite de registros por solicitud (máximo recomendado por Binance)

# Rango de fechas (personalizable)
fecha_inicio = '2024-01-01 00:00:00'
fecha_fin = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

# ==============================
# FUNCIONES
# ==============================
def fetch_ohlcv_with_retry(symbol, timeframe, since=None, limit=1000, max_retries=5):
    for attempt in range(max_retries):
        try:
            data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
            return data
        except Exception as e:
            print(f"Error en intento {attempt+1}/{max_retries}: {str(e)}")
            time.sleep(exchange.rateLimit / 1000)
    return []

def date_to_milliseconds(date_str):
    dt = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    return int(dt.timestamp() * 1000)

def download_ohlcv_range(symbol, timeframe, since, until, limit=1000):
    all_data = []
    current_since = since

    while current_since < until:
        print(f"Descargando desde: {exchange.iso8601(current_since)}")

        data = fetch_ohlcv_with_retry(symbol, timeframe, since=current_since, limit=limit)

        if not data:
            print("No hay más datos o error. Saliendo.")
            break

        all_data.extend(data)

        # Evitar loops infinitos si el timestamp no avanza
        last_timestamp = data[-1][0]
        if current_since == last_timestamp:
            print("Timestamp sin avance, saliendo para evitar bucle infinito.")
            break
        current_since = last_timestamp + 1

        time.sleep(exchange.rateLimit / 1000)

    return all_data

# ==============================
# PROCESO DE DESCARGA
# ==============================
since_timestamp = date_to_milliseconds(fecha_inicio)
until_timestamp = date_to_milliseconds(fecha_fin)

ohlcv_data = download_ohlcv_range(symbol, timeframe, since_timestamp, until_timestamp, limit)

# ==============================
# PROCESO DE CONVERSIÓN Y GUARDADO
# ==============================
columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
df = pd.DataFrame(ohlcv_data, columns=columns)

df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

file_path = f'data/BTC_{timeframe}_ohlcv.csv'
df.to_csv(file_path, index=False)

print(f"✅ Datos guardados correctamente en {file_path}")
print(df.tail())
