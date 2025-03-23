import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
import warnings
warnings.filterwarnings('ignore')

# Cargar el dataset preprocesado
print("🔧 Iniciando Feature Engineering...")

df = pd.read_csv('data/BTC_5m_preprocessed.csv')
print(f"📌 Antes de limpieza - Registros: {len(df)}")

# ========== INDICADORES TÉCNICOS ==========

# EMA
df['ema_12'] = EMAIndicator(df['close'], window=12).ema_indicator()
df['ema_26'] = EMAIndicator(df['close'], window=26).ema_indicator()

# MACD
macd = MACD(df['close'])
df['macd'] = macd.macd()
df['macd_signal'] = macd.macd_signal()
df['macd_histogram'] = macd.macd_diff()

# RSI
df['rsi'] = RSIIndicator(df['close'], window=14).rsi()

# ADX
adx = ADXIndicator(df['high'], df['low'], df['close'], window=14)
df['adx'] = adx.adx()

# ATR
atr = AverageTrueRange(df['high'], df['low'], df['close'], window=14)
df['atr'] = atr.average_true_range()

# ========== CARACTERÍSTICAS ESTADÍSTICAS ==========

df['returns_mean_5'] = df['log_return'].rolling(window=5).mean()
df['returns_std_5'] = df['log_return'].rolling(window=5).std()
df['returns_mean_20'] = df['log_return'].rolling(window=20).mean()
df['returns_std_20'] = df['log_return'].rolling(window=20).std()

# Diferencias y relaciones
df['ema_diff'] = df['ema_12'] - df['ema_26']
df['rsi_macd_ratio'] = df['rsi'] / (df['macd'] + 1e-6)

# Posición en las bandas de Bollinger
df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

# Flags de señales
df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
df['macd_cross'] = (df['macd'] > df['macd_signal']).astype(int)

# Volatilidad
df['volatility'] = df['returns_std_20']
df['vol_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()

# Target (clasificación binaria: si el precio sube en la siguiente vela)
df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

# ========== LIMPIEZA POST CÁLCULOS ==========
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"✅ Después de limpieza - Registros: {len(df)}")

# ========== DEFINICIÓN DE FEATURES ==========
features = [
    'log_return', 'pct_change',
    'sma_20', 'stddev_20',
    'bb_upper', 'bb_lower', 'bb_width',
    'ema_12', 'ema_26',
    'macd', 'macd_signal', 'macd_histogram',
    'rsi', 'adx', 'atr',
    'returns_mean_5', 'returns_std_5',
    'returns_mean_20', 'returns_std_20',
    'ema_diff', 'rsi_macd_ratio',
    'bb_position',
    'rsi_overbought', 'rsi_oversold',
    'macd_cross', 'volatility', 'vol_ratio'
]

# ========= DATASETS SIN SMOTE ==========
X_no_smote = df[features]
y_no_smote = df['target']

# Guardar dataset sin SMOTE (para backtesting)
df_no_smote = X_no_smote.copy()
df_no_smote['target'] = y_no_smote
df_no_smote.to_csv('data/BTC_5m_features_nosmote.csv', index=False)
print("✅ Dataset sin SMOTE guardado en 'data/BTC_5m_features_nosmote.csv'")

# ========== SMOTE BALANCEO ==========
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_no_smote, y_no_smote)

# Dataset balanceado
df_features = pd.DataFrame(X_res, columns=features)
df_features['target'] = y_res

# Guardar dataset balanceado
df_features.to_csv('data/BTC_5m_features.csv', index=False)
print("✅ Dataset balanceado guardado en 'data/BTC_5m_features.csv'")

# Información del dataframe
print(df_features.info())
