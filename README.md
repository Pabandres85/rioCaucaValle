# 🌊 Calidad del Agua del Río Cauca en el Valle del Cauca (Colombia)

Este proyecto analiza datos públicos sobre la calidad del agua del Río Cauca, aplicando técnicas de limpieza, análisis exploratorio, visualización interactiva y modelado predictivo con inteligencia artificial. Está diseñado para funcionar tanto en entornos locales (con Docker) como en la nube mediante [Streamlit Cloud](https://streamlit.io/cloud).

---

## 📌 Objetivo

Evaluar la evolución de la calidad del agua a lo largo del tiempo y por estaciones de muestreo, identificar variables críticas y predecir los niveles de oxígeno disuelto mediante modelos de machine learning.

---

## 🔗 Fuente de datos

Los datos provienen del portal oficial de Datos Abiertos de Colombia:
🔗 [Calidad del agua del Río Cauca - datos.gov.co](https://www.datos.gov.co/Ambiente-y-Desarrollo-Sostenible/Calidad-del-agua-del-Rio-Cauca/d3ft-wu2b/about_data)

---

## 📂 Estructura del Proyecto

```
calidad-agua-cauca/
│
├── app/                    # Aplicación Streamlit
│   ├── streamlit_app.py
│   └── style.css           # Estilos visuales para la app(en desarrollo)
│
├── data/
│   ├── raw/                # Datos originales descargados
│   └── cleaned/            # Datos limpios y listos para análisis
│
├── notebooks/              # Desarrollo exploratorio y modelado
│   ├── 01_limpieza_datos.ipynb
│   ├── 02_eda_visualizaciones.ipynb
│   └── 03_modelos_predictivos.ipynb
│
├── docker/
│   └── Dockerfile          # Dockerfile para ejecución local 
│
├── requirements.txt        # Requisitos mínimos para Streamlit Cloud
├── requirements-dev.txt    # Requisitos completos para entorno local con docker
└── README.md
```

---

## ⚙️ Instalación y Ejecución

### 🟢 Opción 1: Entorno local (recomendado para desarrollo)

```bash
pip install -r requirements-dev.txt
streamlit run app/streamlit_app.py
```

### 🐳 Opción 2: Usar Docker (entorno limpio y replicable)

```bash
cd docker
docker build -t calidad-agua .
docker run -p 8501:8501 calidad-agua
```

---

## 📊 Funcionalidades Destacadas

✅ Exploración de datos por parámetro y estación
✅ Comparación entre estaciones con gráficos tipo boxplot
✅ Mapa de calor de correlaciones entre variables
✅ Predicción de oxígeno disuelto con modelo Random Forest
✅ Comparación de predicción con promedio histórico
✅ Interpretación contextual del resultado (no proyectiva)
✅ Descripciones dinámicas de parámetros técnicos

---

## 📈 Modelado Predictivo (IA)

* Se entrena un modelo Random Forest Regressor con variables clave (pH, temperatura, turbidez, DBO, etc.)
* Se permite al usuario ajustar parámetros manualmente para observar la predicción de oxígeno disuelto bajo esas condiciones
* Se compara la predicción con:

  * El valor histórico promedio
  * El rango observado en el dataset

> ⚠️ Las predicciones son inferencias contextuales, **no proyecciones a futuro**.

---

## 🚧 Contribuciones Futuras

* Agregar más años de datos y estaciones del río
* Mejorar los modelos con interpretabilidad (SHAP, LIME)
* Exportar resultados de predicción en PDF
* Incorporar mapas geoespaciales de estaciones

---

## 👨‍💻 Autor

Desarrollado por [Pablo Andrés Muñoz](https://pabandres85.github.io/PortafolioPablo/)
📧 Contacto: [ingenieropanloandres0@gmail.com](mailto:ingenieropanloandres0@gmail.com)
💡 Proyecto de ciencia de datos aplicada al análisis ambiental en Colombia

---
