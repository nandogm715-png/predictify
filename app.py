import streamlit as st
import pandas as pd

st.set_page_config(page_title="Radar de Reputación de Producto", layout="wide")
st.title("📊 Radar de Reputación de Producto")
st.caption("Fire Tablet & Echo (White) — Amazon Consumer Reviews")

semanal = pd.read_csv("semanal_sentimiento.csv")
temas = pd.read_csv("top_temas.csv")
prediccion = pd.read_csv("prediccion.csv")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Evolución del sentimiento")
    st.line_chart(semanal.set_index("semana_num")["sentimiento"])

with col2:
    st.subheader("Top temas de queja")
    st.table(temas)

st.subheader("Proyección próximas 4 semanas")
st.line_chart(prediccion.set_index("semana_num")["sentimiento_predicho"])

pendiente = prediccion["sentimiento_predicho"].iloc[-1] - semanal["sentimiento"].iloc[-1]
if pendiente < -0.05:
    st.error("⚠️ Alerta: tendencia a la baja en el sentimiento proyectado")
else:
    # ----------------------------------------------------------
# MODO 2: SUBIR ARCHIVO PROPIO (con segmentación)
# ----------------------------------------------------------
    st.write("Sube un archivo CSV con reseñas de productos y selecciona qué columna es cada cosa.")
    archivo = st.file_uploader("Archivo CSV", type="csv")

    if archivo:
        df_nuevo = pd.read_csv(archivo, low_memory=False)
        st.write("Vista previa:", df_nuevo.head())

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            col_fecha = st.selectbox("Columna de fecha", df_nuevo.columns)
        with col2:
            col_rating = st.selectbox("Columna de rating", df_nuevo.columns)
        with col3:
            col_texto = st.selectbox("Columna de texto de reseña", df_nuevo.columns)
        with col4:
            opciones_producto = ["(Analizar todo, sin segmentar)"] + list(df_nuevo.columns)
            col_producto = st.selectbox("Columna de producto (opcional)", opciones_producto)
            if len({col_fecha, col_rating, col_texto}) < 3:
    st.error("⚠️ Debes elegir 3 columnas distintas para fecha, rating y texto — no pueden repetirse.")
    st.stop()

        # --- Segmentación por producto ---
        productos_seleccionados = None
        if col_producto != "(Analizar todo, sin segmentar)":
            conteo = df_nuevo[col_producto].value_counts()
            top_productos = conteo.head(15).index.tolist()  # limitar a los 15 de más volumen
            productos_seleccionados = st.multiselect(
                "¿Qué producto(s) quieres analizar o comparar?",
                options=top_productos,
                default=top_productos[:2] if len(top_productos) >= 2 else top_productos,
                help="Se muestran los 15 productos con más reseñas. Elige uno para analizarlo solo, o varios para compararlos."
            )

        # --- Filtro de rango de fechas ---
        usar_filtro_fecha = st.checkbox("Filtrar por rango de fechas")
        fecha_inicio, fecha_fin = None, None
        if usar_filtro_fecha:
            fechas_preview = pd.to_datetime(df_nuevo[col_fecha], errors='coerce').dropna()
            if len(fechas_preview) > 0:
                fmin, fmax = fechas_preview.min().date(), fechas_preview.max().date()
                fecha_inicio, fecha_fin = st.slider(
                    "Rango de fechas", min_value=fmin, max_value=fmax, value=(fmin, fmax)
                )

        if st.button("Analizar"):
            with st.spinner("Procesando reseñas..."):
                try:
                    df_trabajo = df_nuevo.copy()
                    df_trabajo[col_rating] = pd.to_numeric(df_trabajo[col_rating], errors='coerce')

                    # Aplicar filtro de producto
                    if productos_seleccionados:
                        df_trabajo = df_trabajo[df_trabajo[col_producto].isin(productos_seleccionados)]

                    # Aplicar filtro de fecha
                    if usar_filtro_fecha and fecha_inicio and fecha_fin:
                        fechas_dt = pd.to_datetime(df_trabajo[col_fecha], errors='coerce')
                        df_trabajo = df_trabajo[(fechas_dt.dt.date >= fecha_inicio) & (fechas_dt.dt.date <= fecha_fin)]

                    if len(df_trabajo) == 0:
                        st.warning("No quedan reseñas después de aplicar los filtros. Ajusta la selección.")
                    else:
                        semanal, top_temas, prediccion, total = analizar(df_trabajo, col_fecha, col_rating, col_texto)

                        if len(semanal) == 0:
                            st.error("No se pudieron procesar fechas válidas. Verifica el formato de la columna de fecha.")
                        else:
                            # Comparación entre productos (si hay más de uno seleccionado)
                            if productos_seleccionados and len(productos_seleccionados) > 1:
                                st.subheader("Comparación de sentimiento entre productos seleccionados")
                                comparacion = df_trabajo.copy()
                                comparacion['fecha_dt'] = pd.to_datetime(comparacion[col_fecha], errors='coerce')
                                comparacion = comparacion.dropna(subset=['fecha_dt'])

                                stop_words = cargar_stopwords()
                                from textblob import TextBlob
                                comparacion['texto_limpio'] = comparacion[col_texto].apply(lambda t: limpiar_texto(t, stop_words))
                                comparacion['sentimiento'] = comparacion['texto_limpio'].apply(lambda t: TextBlob(t).sentiment.polarity)
                                comparacion['semana'] = comparacion['fecha_dt'].dt.to_period('W').astype(str)

                                tabla_comparacion = comparacion.groupby(['semana', col_producto])['sentimiento'].mean().unstack()
                                st.line_chart(tabla_comparacion)

                            mostrar_resultados(semanal, top_temas, prediccion, total)

                except Exception as e:
                    st.error(f"Ocurrió un error al procesar el archivo: {e}")
