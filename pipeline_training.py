import subprocess
import time
import os
import sys

# Usa el Python actual del entorno para ejecutar los scripts
python_exe = sys.executable

# Ruta base de los scripts
ruta_scripts = "/Users/ferkuellar/Desktop/bitacora/BTC_Predictive_Model/"

def run_pipeline():
    steps = [
        ("01_data_collection.py", "✅ Recolección de Datos Completa"),
        ("02_data_preprocessing.py", "✅ Preprocesamiento Completo"),
        ("03_feature_engineering.py", "✅ Feature Engineering Completo"),
        ("04_model_training.py", "✅ Entrenamiento del Modelo Completo"),
        ("05_backtesting.py", "✅ Backtesting Completo")  # Opcional
    ]

    print("\n🚀 Iniciando Pipeline de Trading Automatizado...\n")

    for script, mensaje in steps:
        try:
            full_path = os.path.join(ruta_scripts, script)
            print(f"▶️ Ejecutando {full_path} con {python_exe}...")
            subprocess.check_call([python_exe, full_path])
            print(mensaje)
            print("-" * 50)
            time.sleep(1)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error en {script}: {e}")
            break

    print("\n🏁 Pipeline Finalizado. Modelos y datos actualizados.\n")

if __name__ == "__main__":
    run_pipeline()
