import pandas as pd
import numpy as np


# Cargar el archivo preprocesado
df = pd.read_csv('data/BTC_5m_preprocessed.csv')

# Verificar las primeras filas
df.head()


# Crea la columna target: 1 si sube el precio en la siguiente vela, 0 si baja o se mantiene igual
df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

# Mostrar algunas filas para verificar
df[['close', 'target']].head(10)


# Indicadores básicos
df['log_return'] = np.log(df['close'] / df['close'].shift(1))
df['pct_change'] = df['close'].pct_change()

# Exponential Moving Averages (EMA)
df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()

# MACD
df['macd'] = df['ema_12'] - df['ema_26']
df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
df['macd_histogram'] = df['macd'] - df['macd_signal']


bb_window = 20
bb_std = 2
df['bb_middle'] = df['close'].rolling(window=bb_window).mean()
df['bb_std'] = df['close'].rolling(window=bb_window).std()
df['bb_upper'] = df['bb_middle'] + bb_std * df['bb_std']
df['bb_lower'] = df['bb_middle'] - bb_std * df['bb_std']
df['bb_width'] = df['bb_upper'] - df['bb_lower']


window_length = 14
delta = df['close'].diff()
gain = delta.clip(lower=0)
loss = -1 * delta.clip(upper=0)

avg_gain = gain.ewm(alpha=1/window_length, min_periods=window_length).mean()
avg_loss = loss.ewm(alpha=1/window_length, min_periods=window_length).mean()

rs = avg_gain / avg_loss
df['rsi'] = 100 - (100 / (1 + rs))


def calculate_adx(df, n=14):
    high = df['high']
    low = df['low']
    close = df['close']

    df['tr1'] = high - low
    df['tr2'] = (high - close.shift(1)).abs()
    df['tr3'] = (low - close.shift(1)).abs()
    df['TR'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

    df['+DM'] = np.where((high - high.shift(1)) > (low.shift(1) - low),
                         np.maximum(high - high.shift(1), 0), 0)
    df['-DM'] = np.where((low.shift(1) - low) > (high - high.shift(1)),
                         np.maximum(low.shift(1) - low, 0), 0)

    tr14 = df['TR'].rolling(window=n).sum()
    plus_dm14 = df['+DM'].rolling(window=n).sum()
    minus_dm14 = df['-DM'].rolling(window=n).sum()

    plus_di14 = 100 * (plus_dm14 / tr14)
    minus_di14 = 100 * (minus_dm14 / tr14)

    dx = (abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14)) * 100
    adx = dx.rolling(window=n).mean()

    df.drop(columns=['tr1', 'tr2', 'tr3', 'TR', '+DM', '-DM'], inplace=True)

    return adx

# Aplicar el cálculo
df['adx'] = calculate_adx(df)


high = df['high']
low = df['low']
close = df['close']

tr1 = high - low
tr2 = (high - close.shift(1)).abs()
tr3 = (low - close.shift(1)).abs()

df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
df['atr'] = df['TR'].rolling(window=14).mean()

# Elimina la columna TR temporal si no la necesitas
df.drop(columns=['TR'], inplace=True)


df['returns_mean_5'] = df['log_return'].rolling(window=5).mean()
df['returns_std_5'] = df['log_return'].rolling(window=5).std()

df['returns_mean_20'] = df['log_return'].rolling(window=20).mean()
df['returns_std_20'] = df['log_return'].rolling(window=20).std()

df['ema_diff'] = df['ema_12'] - df['ema_26']
df['rsi_macd_ratio'] = df['rsi'] / (df['macd'] + 1e-6)
df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

df['rsi_overbought'] = np.where(df['rsi'] > 70, 1, 0)
df['rsi_oversold'] = np.where(df['rsi'] < 30, 1, 0)
df['macd_cross'] = np.where(df['macd'] > df['macd_signal'], 1, 0)

df['volatility'] = df['returns_std_20']
df['vol_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()


df.dropna(inplace=True)

# Revisar estructura final
df.info()


df.to_csv('data/BTC_5m_features.csv', index=False)
print("✅ Dataset guardado en 'data/BTC_5m_features.csv'")




