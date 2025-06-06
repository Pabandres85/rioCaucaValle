import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Calidad del Agua - Río Cauca en el Valle del Cauca", layout="wide")

# Estilos para ajustar imágenes y gráficos
st.markdown("""
    <style>
    img, .stImage > img {
        max-width: 720px;
        height: auto;
        border-radius: 10px;
        display: block;
        margin: auto;
    }
    .stPlotlyChart, .stAltairChart, .stPyplot {
        max-width: 720px !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌊 Calidad del Agua del Río Cauca en el Valle del Cauca")

@st.cache_data
def load_data():
    df = pd.read_csv("data/cleaned/agua_rio_cauca_limpio.csv", parse_dates=["FECHA DE MUESTREO"])
    return df

df = load_data()
estaciones = sorted(df["ESTACIONES"].dropna().unique())
min_fecha = df["FECHA DE MUESTREO"].min().strftime("%Y-%m-%d")
max_fecha = df["FECHA DE MUESTREO"].max().strftime("%Y-%m-%d")

descripcion_parametros = {
    "pH": "Medida de acidez o alcalinidad del agua.",
    "TEMPERATURA (°C)": "Temperatura del agua al momento del muestreo.",
    "TURBIEDAD (UNT)": "Cantidad de partículas suspendidas que reducen la claridad.",
    "DEMANDA BIOQUIMICA DE OXIGENO (mg O2/l)": "Oxígeno necesario para descomponer materia orgánica.",
    "DUREZA TOTAL (mg CaCO3/l)": "Concentración de sales de calcio y magnesio.",
    "SOLIDOS DISUELTOS (mg SD/l)": "Cantidad total de sólidos disueltos en el agua.",
    "COLOR (UPC)": "Presencia de compuestos que afectan el color.",
    "OXIGENO DISUELTO (mg O2/l)": "Oxígeno disponible en el agua para organismos vivos."
}

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Exploración de Datos",
    "🧪 Comparación entre Estaciones",
    "🔍 Correlaciones",
    "🤖 Predicciones con IA",
    "📘 Información del Proyecto"
])

with tab1:
    st.header("📊 Exploración de Datos")
    st.markdown(f"📆 **Rango de datos:** {min_fecha} → {max_fecha}")
    selected_var = st.selectbox("Selecciona una variable:", df.select_dtypes("float").columns)
    selected_station = st.selectbox("Selecciona una estación:", estaciones)
    df_filtered = df[df["ESTACIONES"] == selected_station].sort_values("FECHA DE MUESTREO")

    with st.container():
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(df_filtered["FECHA DE MUESTREO"], df_filtered[selected_var], marker="o")
        ax.set_title(f"Evolución de {selected_var} en el tiempo", fontsize=12)
        ax.set_xlabel("Fecha")
        ax.set_ylabel(selected_var)
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)

    with st.expander("ℹ️ Descripción del parámetro"):
        descripcion = descripcion_parametros.get(selected_var, "No se encontró descripción para este parámetro.")
        st.markdown(f"📘 **Descripción de la variable {selected_var}:** {descripcion}")

with tab2:
    st.header("🧪 Comparación entre Estaciones")
    selected_stations = st.multiselect("Selecciona hasta 3 estaciones:", estaciones, default=estaciones[:2])
    selected_param = st.selectbox("Selecciona el parámetro a comparar:", df.select_dtypes("float").columns)

    if len(selected_stations) >= 1:
        df_comp = df[df["ESTACIONES"].isin(selected_stations)]
        fig, ax = plt.subplots(figsize=(7, 3))
        sns.boxplot(data=df_comp, x="ESTACIONES", y=selected_param, ax=ax)
        ax.set_title(f"{selected_param} por estación", fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)

with tab3:
    st.header("🔍 Correlación entre Parámetros")
    df_corr = df.select_dtypes("float").dropna()
    corr_matrix = df_corr.corr()
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(corr_matrix, cmap="coolwarm", ax=ax)
    ax.set_title("Matriz de correlación", fontsize=12)
    st.pyplot(fig)

with tab4:
    st.header("🤖 Predicción de Oxígeno Disuelto con IA")
    st.markdown("Modelo entrenado con Random Forest sobre parámetros como pH, temperatura, turbidez, etc.")

    # Inputs
    pH = st.slider("pH", 5.0, 9.0, 7.0)
    temp = st.slider("Temperatura (°C)", 0.0, 40.0, 25.0)
    turb = st.slider("Turbiedad (UNT)", 0.0, 100.0, 10.0)
    dbo = st.slider("DBO (mg O2/l)", 0.0, 20.0, 5.0)
    dureza = st.slider("Dureza total (mg CaCO3/l)", 0.0, 1000.0, 200.0)
    solidos = st.slider("Sólidos disueltos (mg/l)", 0.0, 1000.0, 300.0)

    X_features = ['pH', 'TEMPERATURA (°C)', 'TURBIEDAD (UNT)',
                  'DEMANDA BIOQUIMICA DE OXIGENO (mg O2/l)',
                  'DUREZA TOTAL (mg CaCO3/l)', 'SOLIDOS DISUELTOS (mg SD/l)']
    df_model = df[X_features + ['OXIGENO DISUELTO (mg O2/l)']].dropna()
    X = df_model[X_features]
    y = df_model['OXIGENO DISUELTO (mg O2/l)']

    model = RandomForestRegressor()
    model.fit(X, y)

    pred_input = np.array([[pH, temp, turb, dbo, dureza, solidos]])
    pred_avg_input = np.array([[df_model[col].mean() for col in X_features]])

    prediction_user = model.predict(pred_input)[0]
    prediction_avg = model.predict(pred_avg_input)[0]

    st.success(f"🤖 Predicción con tus valores: **{prediction_user:.2f} mg O2/l**")
    st.info(f"📘 Predicción con valores promedio históricos: **{prediction_avg:.2f} mg O2/l**")

    st.subheader("📊 Comparación con valores históricos")
    st.markdown(f"""
- **Valor mínimo registrado:** `{y.min():.2f} mg O2/l`  
- **Valor máximo registrado:** `{y.max():.2f} mg O2/l`  
- **Promedio histórico:** `{y.mean():.2f} mg O2/l`

🔹 Esta predicción **estima el valor típico de oxígeno disuelto bajo las condiciones seleccionadas**.  
🚫 **No representa una proyección futura**, sino una inferencia contextual basada en el comportamiento pasado del río.  
🧠 Útil para **simular escenarios de intervención ambiental o monitoreo preventivo**.

🧾 **Resultado respecto al promedio:** {"🔻 Bajo" if prediction_user < y.mean() else "🔺 Alto"}
""")

    fig, ax = plt.subplots(figsize=(7, 3))
    sns.histplot(y, kde=True, color="skyblue", ax=ax)
    ax.axvline(prediction_user, color="red", linestyle="--", label="Tu predicción")
    ax.axvline(prediction_avg, color="green", linestyle=":", label="Predicción promedio")
    ax.set_title("Distribución histórica del oxígeno disuelto")
    ax.set_xlabel("OXIGENO DISUELTO (mg O2/l)")
    ax.legend()
    st.pyplot(fig)

    st.subheader("📌 Importancia de los parámetros")
    importance_df = pd.DataFrame({
        "Parámetro": X.columns,
        "Importancia": model.feature_importances_
    }).sort_values(by="Importancia", ascending=False)
    st.bar_chart(importance_df.set_index("Parámetro"))

    with st.expander("ℹ️ Interpretación del resultado"):
        if prediction_user < 4:
            st.warning("⚠️ Niveles bajos pueden indicar contaminación orgánica o escasa oxigenación.")
        elif prediction_user > 8:
            st.success("✅ Niveles altos sugieren buena salud acuática.")
        else:
            st.info("🟡 Nivel moderado, monitoreo recomendado.")

with tab5:
    st.header("📘 Información del Proyecto")
    st.markdown(f"""
- **Objetivo:** Evaluar la calidad del agua del Río Cauca mediante análisis de datos públicos.
- **Rango de datos:** desde {min_fecha} hasta {max_fecha}.
- **Fuente de datos:** [Calidad del agua del Río Cauca - datos.gov.co](https://www.datos.gov.co/Ambiente-y-Desarrollo-Sostenible/Calidad-del-agua-del-Rio-Cauca/d3ft-wu2b/about_data)
- **Autor:** [Pablo Andrés Muñoz](https://pabandres85.github.io/PortafolioPablo/)
""")
