# 📊 Predictify

**Radar de Reputación de Producto con Predicción Temprana**

Predictify analiza reseñas de clientes, detecta los temas de queja más frecuentes y **proyecta la tendencia de reputación a futuro** — para que un equipo de producto pueda actuar antes de que un problema se refleje en las ventas.


---

## El problema

Las empresas reciben miles de reseñas de clientes, pero nadie tiene tiempo de leerlas una por una. Los problemas de calidad, funcionamiento o servicio suelen detectarse tarde — cuando ya afectaron las ventas o la reputación de la marca.

## La solución

Un sistema que lee automáticamente las reseñas, mide su sentimiento, detecta los temas de queja reales y **proyecta hacia adelante** — convirtiendo miles de reseñas dispersas en una sola pregunta respondida con confianza: *¿mi producto está bien, o hay algo que debería estar viendo ahora mismo?*

---

## ✨ Funcionalidades principales

- **Ingesta y limpieza de datos** — carga del dataset de reseñas, filtrado por producto y eliminación de registros incompletos o corruptos.
- **Análisis exploratorio (EDA)** — patrones de volumen, estacionalidad y distribución de calificaciones.
- **Análisis de sentimiento** — puntaje de polaridad por reseña, contrastado contra el rating numérico para detectar discrepancias relevantes (ej. reseñas con rating alto pero quejas específicas escondidas en el texto).
- **Detección de temas de queja** — extracción automática de los términos más frecuentes en reseñas negativas.
- **Modelo predictivo de tendencia** — proyección del sentimiento promedio a futuro mediante regresión lineal sobre datos agregados semanalmente.
- **Dashboard interactivo** — visualización en vivo de la evolución histórica, los temas de queja actuales y la proyección, con alertas automáticas cuando la tendencia es preocupante.

---

## 🛠️ Tecnologías utilizadas

| Categoría | Herramienta |
|---|---|
| Lenguaje | Python |
| Análisis de datos | pandas |
| Base de datos | SQLite / SQL |
| Machine Learning | scikit-learn (CountVectorizer, regresión lineal) |
| NLP | NLTK, TextBlob |
| Entorno de desarrollo | Google Colab |
| Dashboard | Streamlit |
| Control de versiones y despliegue | GitHub, Streamlit Community Cloud |

---

## 📁 Estructura del repositorio

```
predictify/
├── app.py                      # código de la aplicación Streamlit
├── requirements.txt            # dependencias del proyecto
├── semanal_sentimiento.csv     # sentimiento promedio agregado por semana
├── top_temas.csv               # términos más frecuentes en reseñas negativas
├── prediccion.csv              # proyección del modelo a 4 semanas
└── README.md
```

---

## 📊 Fuente de datos

Dataset: [Consumer Reviews of Amazon Products](https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products) (Datafiniti, vía Kaggle).

Se trabajó con un subconjunto filtrado de **14,272 reseñas** correspondientes a dos productos de alto volumen: **Fire Tablet (7", Wi-Fi, 8GB)** y **Echo (White)**, priorizados por tener suficiente densidad de datos para un análisis de tendencia semanal confiable.

---

## 🔍 Cómo funciona el pipeline

1. **Carga y limpieza**: se filtra el dataset a los productos de mayor volumen y se eliminan filas sin fecha, rating o texto.
2. **Exploración (SQL + pandas)**: se agregan las reseñas por mes/producto/rating para entender el volumen y la distribución antes de modelar.
3. **NLP**: el texto se limpia (minúsculas, sin puntuación, sin palabras vacías) y se calcula un puntaje de sentimiento por reseña con TextBlob.
4. **Extracción de temas**: se identifican los términos más frecuentes en reseñas negativas con `CountVectorizer`.
5. **Modelo predictivo**: el sentimiento se agrega por semana y se entrena una regresión lineal para proyectar las próximas 4 semanas.
6. **Dashboard**: los resultados se exportan a CSV livianos y se visualizan en una app de Streamlit, sin necesidad de recalcular el análisis NLP en cada carga.

---

## 📈 Hallazgos principales

- Solo el **2.7%** de las reseñas analizadas son negativas (1-2★), por lo que el sistema prioriza detectar **cambios** en ese porcentaje, no solo el volumen absoluto de quejas.
- Las reseñas negativas son en promedio **~64% más largas** que las de 5 estrellas — los clientes insatisfechos se explayan más.
- **Fire Tablet** tiene casi el doble de tasa de quejas que **Echo** (2.97% vs. 1.63%), pese a que el conteo absoluto de negativas es mayor por su volumen total.
- Las quejas se concentran en **apps que no funcionan bien** y **lentitud del dispositivo** — no en defectos de hardware.
- El modelo predictivo no detectó una tendencia significativa de deterioro en el periodo analizado (pendiente ≈ 0), un resultado igual de válido y útil que detectar una caída real.

---

## 🚀 Cómo correrlo localmente

```bash
git clone https://github.com/tu-usuario/predictify.git
cd predictify
pip install -r requirements.txt
streamlit run app.py
```

---

## 🔮 Posibles mejoras futuras

- Selector de columnas para que la app funcione con cualquier CSV de reseñas, no solo el dataset actual.
- Auditoría cualitativa adicional generada por un LLM (Anthropic API o Qwen local) sobre los temas detectados.
- Actualización automática programada (job diario/semanal) en vez de análisis estático.

---

## 👥 Integrantes

- John Dimate
- Fernando Granadillo

*Proyecto desarrollado como entrega para el curso de Analista Junior de Datos.*
