
---

# 🚀 BTC Predictive Trading System  
### *Automated Crypto Trading powered by Machine Learning & Deep Learning*  

---

## 🎯 Propósito del Proyecto
BTC Predictive Trading System es una **plataforma de trading automatizado de criptomonedas**, basada en modelos de **Machine Learning (ML)** y **Deep Learning (DL)**, diseñada para **predecir movimientos de precios** y ejecutar **operaciones de alta probabilidad**.  
El objetivo es facilitar decisiones informadas y **gestión de riesgos dinámica** en un mercado volátil como el de Bitcoin.

---

## 🏗️ ¿Qué Problema Resuelve?  
- La **incertidumbre** y **emocionalidad** al operar en mercados cripto.  
- La **ineficiencia** en el análisis manual de datos masivos.  
- La **falta de gestión de riesgos profesional** en traders retail.  
- La necesidad de **automatización** y **monitoreo en tiempo real** de las operaciones.

---

## 🚀 ¿Cómo Funciona?  
Este sistema se compone de 4 módulos principales:  
1. **Recolección de Datos**  
   - Obtiene datos de mercado (OHLCV) en tiempo real desde Binance.  
2. **Predicción de Precios**  
   - Modelos híbridos **XGBoost + LSTM** generan señales de compra/venta de alta precisión.  
3. **Gestión de Riesgos**  
   - Stop Loss / Take Profit **dinámicos**, ajustados por volatilidad (ATR).  
4. **Automatización & Alertas**  
   - Dashboard en **Streamlit** para el seguimiento en vivo.  
   - Notificaciones automáticas por **Telegram**.

---

## 🛠️ Tecnología Utilizada  
| Componente         | Tecnología                |
|--------------------|---------------------------|
| Lenguaje           | Python 3.9                |
| Machine Learning   | XGBoost, LightGBM         |
| Deep Learning      | TensorFlow, Keras         |
| Visualización      | Streamlit, Plotly         |
| Notificaciones     | Telegram Bot API          |
| Fuentes de Datos   | Binance API, CCXT         |
| Contenedores       | Docker, Docker Compose    |

---

## ✨ Características Clave  
| ✅ Funcionalidad              | 📌 Descripción                                                         |
|------------------------------|----------------------------------------------------------------------|
| 🎯 Predicciones de Alta Precisión  | Señales de trading con modelos híbridos ML/DL.                      |
| 🔒 Gestión de Riesgo Dinámica      | Cálculo automático de SL/TP basado en volatilidad y ATR.           |
| 📈 Backtesting                   | Motor de simulación con métricas clave (ROI, Sharpe Ratio, etc.).  |
| 📊 Dashboard en Tiempo Real       | Monitoreo de señales y portafolio vía Streamlit.                   |
| 📡 Alertas en Telegram            | Notificaciones instantáneas de Buy/Sell y alertas de riesgo.       |
| 🐳 Despliegue con Docker          | Entorno totalmente contenedorizado y escalable.                    |

---

## 🧠 Arquitectura del Sistema  
```
BTC_Predictive_Model/
├── data/                     # Datos crudos e históricos
├── models/                   # Modelos entrenados
├── 01_data_collection.py     # Recolección de datos de mercado
├── 02_data_preprocessing.py  # Limpieza y normalización de datos
├── 03_feature_engineering.py # Generación de indicadores técnicos
├── 04_model_training.py      # Entrenamiento ML/DL
├── 05_backtesting.py         # Backtesting de estrategias
├── 06_streamlit_dashboard.py # Dashboard en tiempo real
├── pipeline_training.py      # Orquestación automática
├── requirements.txt          # Dependencias del proyecto
└── Dockerfile                # Contenedor Docker
```

---

## 📊 Métricas de Evaluación  
| Métrica           | Descripción                                 |
|-------------------|---------------------------------------------|
| ROI               | Retorno de inversión sobre el período de test |
| Win/Loss Ratio    | Ratio de operaciones ganadoras vs perdedoras |
| Max Drawdown      | Pérdida máxima desde el punto más alto       |
| Sharpe Ratio      | Retorno ajustado por el riesgo               |

---

## 📡 Ejemplo de Alerta en Telegram  
```
📈 TRADE SIGNAL  
Par: BTCUSDT  
Acción: BUY (LONG)  
Entrada: $40,000  
Stop Loss: $39,400  
Take Profit: $41,200  
```

---

## 🛣️ Roadmap del Proyecto  
| Fase                        | Estado        |
|-----------------------------|---------------|
| Recolección de Datos        | ✅ Completado |
| Ingeniería de Características | ✅ Completado |
| Entrenamiento de Modelos    | ✅ Completado |
| Dashboard en Tiempo Real    | ✅ Completado |
| Alertas Telegram            | ✅ Completado |
| Trading en Tiempo Real      | 🔜 En Desarrollo |
| Soporte Multi-Activos       | 🔜 Planeado   |

---
## 📓 Google Colab Notebooks  
¿Prefieres trabajar en la nube? Abre cualquiera de nuestros notebooks directamente en Google Colab:

| Notebook                                   | Descripción                                          | Google Colab |
|--------------------------------------------|------------------------------------------------------|--------------|
| `01_data_collection.ipynb`                 | Recolección de datos históricos de Binance           | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fernandokuellar/BTC_Predictive_Model/blob/main/01_data_collection.ipynb) |
| `02_data_preprocessing.ipynb`              | Limpieza, normalización y preparación de datos       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fernandokuellar/BTC_Predictive_Model/blob/main/02_data_preprocessing.ipynb) |
| `03_feature_engineering.ipynb`             | Generación de características e indicadores técnicos | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fernandokuellar/BTC_Predictive_Model/blob/main/03_feature_engineering.ipynb) |
| `04_model_training.ipynb`                  | Entrenamiento de modelos XGBoost y LSTM              | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fernandokuellar/BTC_Predictive_Model/blob/main/04_model_training.ipynb) |
| `05_backtesting.ipynb`                     | Evaluación y backtesting de la estrategia            | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fernandokuellar/BTC_Predictive_Model/blob/main/05_backtesting.ipynb) |

## ⚠️ Disclaimer  
> Este proyecto es **experimental** y de uso **educativo**. El trading de criptomonedas implica **alto riesgo**. El autor no asume responsabilidad alguna por pérdidas financieras derivadas del uso de este software.

---

## 👨‍💻 Autor  
**Fernando Kuellar**  
🔗 [LinkedIn](https://www.linkedin.com/in/fernando-kuellar)  
📧 kuellarfer@gmail.com  

---
