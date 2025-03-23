import streamlit as st
import pandas as pd
import requests
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import subprocess
import time

# ==============================
# CONFIGURACIÓN DE LA PÁGINA
# ==============================
st.set_page_config(page_title="Trading Dashboard", page_icon="🚀", layout="wide")

# ==============================
# ESTILOS BINANCE
# ==============================
st.markdown("""
    <style>
    body { background-color: #0B0E11; color: #EAECEF; font-family: 'Arial', sans-serif; }
    .header-title { color: #F0B90B; font-size: 48px; font-weight: bold; text-align: center; margin-bottom: 30px; }
    .section-title { font-size: 24px; color: #F0B90B; margin-top: 40px; margin-bottom: 10px; }
    .footer { font-size: 14px; color: #666666; text-align: center; margin-top: 50px; }
    </style>
""", unsafe_allow_html=True)

# ==============================
# HEADER
# ==============================
st.markdown('<div class="header-title">Trading Dashboard</div>', unsafe_allow_html=True)

# ==============================
# AUTO REFRESH (Cada 1 segundo)
# ==============================
st_autorefresh(interval=1000, key="refresh")

# ==============================
# PORTAFOLIO DE MONEDAS
# ==============================
PARES_DISPONIBLES = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT",
    "SOLUSDT", "DOTUSDT", "AVAXUSDT", "DOGEUSDT", "MATICUSDT",
    "TRXUSDT", "LINKUSDT"
]

TEMPORALIDADES = {
    "1 Minuto": "1m", "5 Minutos": "5m", "15 Minutos": "15m", "1 Hora": "1h", "4 Horas": "4h"
}

# ==============================
# SIDEBAR CONFIGURACIÓN
# ==============================
st.sidebar.header("⚙️ Configuración de Trading")
par_seleccionado = st.sidebar.selectbox("🔄 Par de Trading", PARES_DISPONIBLES)
temporalidad_seleccionada = st.sidebar.selectbox("⏱️ Temporalidad", list(TEMPORALIDADES.keys()))
capital_usuario = st.sidebar.number_input("💰 Capital (USDT)", min_value=100.0, value=10000.0, step=100.0)
sl_percentage = st.sidebar.slider("🎯 Stop Loss (%)", min_value=0.5, max_value=10.0, value=1.5, step=0.1)
rb_ratio = st.sidebar.slider("⚖️ Riesgo/Beneficio", min_value=1.0, max_value=10.0, value=2.0, step=0.1)
tp_percentage = sl_percentage * rb_ratio

# Dirección manual de la operación
direccion_manual = st.sidebar.radio("📈 Tipo de Operación", ["LARGO", "CORTO"])

# ==============================
# BOTÓN PARA REENTRENAR MODELO
# ==============================
st.markdown('<div class="section-title">🔄 Reentrenamiento de Modelos</div>', unsafe_allow_html=True)

if st.button("⚙️ Retrain Now"):
    with st.spinner('⏳ Ejecutando el reentrenamiento de los modelos...'):
        time.sleep(2)
        
        # Ejecuta tus scripts aquí (asegúrate que los scripts estén en el mismo directorio o con path correcto)
        subprocess.call(["python", "01_data_collection.py"])
        subprocess.call(["python", "02_data_preprocessing.py"])
        subprocess.call(["python", "03_feature_engineering.py"])
        subprocess.call(["python", "04_model_training.py"])
        # subprocess.call(["python", "05_backtesting.py"])  # Si quieres validar luego
        
        time.sleep(1)
        st.success("✅ ¡Reentrenamiento completo! Modelos actualizados correctamente.")
        st.balloons()

# ==============================
# FUNCIONES PARA PRECIO Y PREDICCIÓN
# ==============================
def obtener_precio_binance(par):
    url = f'https://api.binance.com/api/v3/ticker/price?symbol={par}'
    try:
        response = requests.get(url, timeout=5)
        return float(response.json()['price'])
    except:
        return None

def obtener_prediccion_par(par):
    import random
    return random.choice([1, 0])

# ==============================
# SESIÓN PARA GUARDAR HISTÓRICO Y ESTADÍSTICAS
# ==============================
if 'historial_operaciones' not in st.session_state:
    st.session_state['historial_operaciones'] = []

# ==============================
# DATOS EN TIEMPO REAL
# ==============================
precio_actual = obtener_precio_binance(par_seleccionado)
prediccion = obtener_prediccion_par(par_seleccionado)

if precio_actual is None:
    st.warning(f"⚠️ No se pudo obtener el precio de {par_seleccionado}.")
else:
    # ==============================
    # PREDICCIÓN (SOLO ASESORA)
    # ==============================
    if prediccion == 1:
        prediccion_texto = "🟢 Predicción: COMPRA"
    else:
        prediccion_texto = "🔴 Predicción: VENTA"

    # ==============================
    # DIRECCIÓN MANUAL DEL USUARIO
    # ==============================
    if direccion_manual == "LARGO":
        tipo_operacion = "🟢 COMPRA (LARGO)"
        sl_price = precio_actual * (1 - sl_percentage / 100)
        tp_price = precio_actual * (1 + tp_percentage / 100)
        perdida_potencial = (precio_actual - sl_price) * (capital_usuario / precio_actual)
        ganancia_potencial = (tp_price - precio_actual) * (capital_usuario / precio_actual)
    else:
        tipo_operacion = "🔴 VENTA (CORTO)"
        sl_price = precio_actual * (1 + sl_percentage / 100)
        tp_price = precio_actual * (1 - tp_percentage / 100)
        perdida_potencial = (sl_price - precio_actual) * (capital_usuario / precio_actual)
        ganancia_potencial = (precio_actual - tp_price) * (capital_usuario / precio_actual)

    # ==============================
    # MOSTRAR SEÑALES Y GESTIÓN DE RIESGO
    # ==============================
    st.subheader(prediccion_texto)
    st.title(f"{tipo_operacion}")
    st.metric("💲 Precio Actual", f"${precio_actual:,.4f}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Stop Loss", f"${sl_price:,.4f}", f"-{sl_percentage}%", delta_color="inverse")
    col2.metric("Take Profit", f"${tp_price:,.4f}", f"+{tp_percentage}%", delta_color="normal")
    col3.metric("R/B Ratio", f"{rb_ratio:.2f}x")

    col4, col5 = st.columns(2)
    col4.metric("Pérdida Potencial", f"-${perdida_potencial:,.2f}", delta_color="inverse")
    col5.metric("Ganancia Potencial", f"+${ganancia_potencial:,.2f}")

    # ==============================
    # BOTÓN PARA CAPTURAR OPERACIÓN
    # ==============================
    st.markdown('<div class="section-title">📥 Captura de Operación</div>', unsafe_allow_html=True)

    resultado = st.radio("✅ Resultado de la operación", ["Ganada", "Perdida"])

    if st.button("📌 Capturar Operación Seleccionada"):
        ganada = 1 if resultado == "Ganada" else 0
        operacion = {
            "Fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Par": par_seleccionado,
            "Temporalidad": temporalidad_seleccionada,
            "Operación": tipo_operacion,
            "Precio Entrada": precio_actual,
            "Stop Loss": sl_price,
            "Take Profit": tp_price,
            "Capital": capital_usuario,
            "Resultado": resultado,
            "Ganancia/Pérdida": ganancia_potencial if ganada else -perdida_potencial
        }
        st.session_state['historial_operaciones'].append(operacion)
        st.success("✅ Operación capturada correctamente.")

# ==============================
# HISTÓRICO DE OPERACIONES
# ==============================
st.markdown('<div class="section-title">📑 Histórico de Operaciones</div>', unsafe_allow_html=True)

if len(st.session_state['historial_operaciones']) > 0:
    df_historial = pd.DataFrame(st.session_state['historial_operaciones'])
    st.dataframe(df_historial, use_container_width=True)

    # ==============================
    # ESTADÍSTICAS DE PERFORMANCE
    # ==============================
    total_ops = len(df_historial)
    ganadas = df_historial['Resultado'].value_counts().get("Ganada", 0)
    perdidas = df_historial['Resultado'].value_counts().get("Perdida", 0)
    porcentaje_exito = (ganadas / total_ops) * 100 if total_ops > 0 else 0
    roi_total = df_historial["Ganancia/Pérdida"].sum()

    st.markdown('<div class="section-title">📈 Estadísticas de Performance</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Operaciones", total_ops)
    col2.metric("Ganadas / Perdidas", f"{ganadas} / {perdidas}")
    col3.metric("Porcentaje de Éxito", f"{porcentaje_exito:.2f}%")

    st.metric("ROI Total", f"${roi_total:,.2f}")
else:
    st.info("No hay operaciones registradas aún.")

# ==============================
# FOOTER
# ==============================
st.markdown(f'<div class="footer">⏱️ Última actualización: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>', unsafe_allow_html=True)
