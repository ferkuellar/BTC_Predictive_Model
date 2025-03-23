import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargamos el dataset descargado de la colección de datos (output de 01_data_collection)
file_path = 'data/BTC_5m_ohlcv.csv'
df = pd.read_csv(file_path)

# Vista previa para verificar
df.head()

# Revisamos las columnas disponibles
print("Columnas actuales:", df.columns.tolist())

# Verificamos si existen valores nulos en las columnas básicas
print("Valores nulos por columna:")
print(df.isnull().sum())


# Eliminamos duplicados si existen
df.drop_duplicates(inplace=True)

# Eliminamos filas con valores nulos en columnas clave
df.dropna(subset=['timestamp', 'open', 'high', 'low', 'close', 'volume'], inplace=True)

# Convertimos timestamp a datetime si no se ha hecho
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Revisamos que el volumen no tenga valores negativos o cero (pueden ser errores)
df = df[df['volume'] > 0]

# Confirmamos la limpieza
print(f"Dataset limpio. Total de registros: {len(df)}")


# Retorno logarítmico
df['log_return'] = np.log(df['close'] / df['close'].shift(1))

# Cambio porcentual
df['pct_change'] = df['close'].pct_change()

# Rellenar posibles nulos iniciales
df.fillna(0, inplace=True)


# Revisamos si las bandas de Bollinger existen y calculamos bb_width
if 'bb_upper' not in df.columns or 'bb_lower' not in df.columns:
    # Si no existen, calculamos las bandas de Bollinger básicas
    window = 20
    df['sma_20'] = df['close'].rolling(window=window).mean()
    df['stddev_20'] = df['close'].rolling(window=window).std()

    df['bb_upper'] = df['sma_20'] + (df['stddev_20'] * 2)
    df['bb_lower'] = df['sma_20'] - (df['stddev_20'] * 2)

# Ahora calculamos bb_width
if 'bb_width' not in df.columns:
    df['bb_width'] = df['bb_upper'] - df['bb_lower']

# Eliminamos valores nulos generados por rolling windows
df.dropna(inplace=True)

# Confirmamos las nuevas columnas
print("Columnas tras agregar bandas de Bollinger y bb_width:")
print(df.columns.tolist())


# Analizamos el volumen para detectar valores atípicos
plt.figure(figsize=(10, 4))
plt.hist(df['volume'], bins=50)
plt.title('Distribución del Volumen')
plt.xlabel('Volumen')
plt.ylabel('Frecuencia')
plt.show()

# Opcional: podemos eliminar outliers extremos
percentile_99 = df['volume'].quantile(0.99)
df = df[df['volume'] <= percentile_99]

print(f"Datos filtrados de volumen extremo. Total de registros: {len(df)}")


# Guardamos el dataset listo para feature engineering o modelado
output_file = 'data/BTC_5m_preprocessed.csv'
df.to_csv(output_file, index=False)

print(f"✅ Dataset preprocesado guardado en {output_file}")
df.tail()




