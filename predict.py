import pandas as pd
import joblib
import streamlit as st

MODELO_PATH_KNN = "proyecto/modelos/modelo_fatiga_knn.pkl"
MODELO_PATH_LR  = "proyecto/modelos/modelo_fatiga_lr.pkl"
MODELO_PATH_DT  = "proyecto/modelos/modelo_fatiga_dt.pkl"

# --- Carga de modelos ---
try:
    modelo_knn = joblib.load(MODELO_PATH_KNN)
    modelo_lr  = joblib.load(MODELO_PATH_LR)
    modelo_dt  = joblib.load(MODELO_PATH_DT)
except FileNotFoundError as e:
    st.error(f"No se encontró el archivo del modelo: {e}")
    st.info("Ejecuta primero 'train.py' para entrenar y guardar los modelos.")
    st.stop()
except Exception as e:
    st.error(f"Error inesperado: {e}")
    st.stop()

# metricas
@st.cache_data
def cargar_metricas():
    return joblib.load("proyecto/modelos/metricas.pkl")

try:
    metricas = cargar_metricas()
except:
    metricas = None

def nivel_fatiga(valor):
    if valor <= 20:
        return "Muy baja", "Sin fatiga significativa", "success"
    elif valor <= 40:
        return "Baja", "Esfuerzo leve", "success"
    elif valor <= 60:
        return "Media", "Fatiga moderada", "warning"
    elif valor <= 80:
        return "Alta", "Fatiga evidente", "warning"
    else:
        return "Muy alta", "Fatiga extrema / agotamiento", "error"

def mostrar_resultado(label, fatiga):
    nivel, interpretacion, color = nivel_fatiga(fatiga)
    st.metric(label=label, value=f"{fatiga:.1f} / 100")
    st.progress(int(fatiga) / 100)
    if color == "success":
        st.success(f"**Nivel:** {nivel} — {interpretacion}")
    elif color == "warning":
        st.warning(f"**Nivel:** {nivel} — {interpretacion}")
    else:
        st.error(f"**Nivel:** {nivel} — {interpretacion}")

st.title("Predicción de Fatiga en Ciclistas")
st.subheader("Comparación de modelos: KNN, Regresión Lineal y Árbol de Decisión")

# Métricas de los modelos
if metricas:
    col1, col2, col3 = st.columns(3)
    for col, nombre, key in zip(
        [col1, col2, col3],
        ["KNN", "Regresión Lineal", "Árbol de Decisión"],
        ["knn", "lr", "dt"]
    ):
        with col:
            st.write(f"**Evaluación {nombre}:**")
            st.write("MSE:", round(metricas[key]["mse"], 2))
            st.write("MAE:", round(metricas[key]["mae"], 2))
            st.write("R²:",  round(metricas[key]["r2"],  4))
            st.info(f"R²: {metricas[key]['r2']:.2%}")

# Entradas
st.divider()
col_a, col_b = st.columns(2)
with col_a:
    bmp = st.number_input("Frecuencia cardíaca (bpm):")
    watts = st.number_input("Potencia (watts):")
    rpm = st.number_input("Cadencia (rpm):")
    tiempo = st.number_input("Tiempo (minutos acumulados):")
with col_b:
    temperatura = st.number_input("Temperatura (°C):")
    pendiente = st.number_input("Pendiente (%):")
    velocidad = st.number_input("Velocidad (km/h):")

# Predicción
if st.button("Predecir Fatiga"):
    entrada = [[bmp, watts, rpm, tiempo, temperatura, pendiente, velocidad]]

    fatiga_knn = modelo_knn.predict(entrada)[0]
    fatiga_lr  = modelo_lr.predict(entrada)[0]
    fatiga_dt  = modelo_dt.predict(entrada)[0]

    col1_r, col2_r, col3_r = st.columns(3)
    with col1_r:
        mostrar_resultado("Fatiga estimada — KNN", fatiga_knn)
    with col2_r:
        mostrar_resultado("Fatiga estimada — Regresión Lineal", fatiga_lr)
    with col3_r:
        mostrar_resultado("Fatiga estimada — Árbol de Decisión", fatiga_dt)