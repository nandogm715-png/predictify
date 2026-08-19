import streamlit as st
import pandas as pd
import re

st.set_page_config(page_title="Predictify — Radar de Reputación", layout="wide")

# ============================================================
# FUNCIONES DEL PIPELINE
# ============================================================

@st.cache_resource
def cargar_stopwords():
    import nltk
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    return set(stopwords.words('english'))

def limpiar_texto(texto, stop_words):
    texto = str(texto).lower()
    texto = re.sub(r'[^a-z\s]', '', texto)
    palabras = [p for p in texto.split() if p not in stop_words]
    return ' '.join(palabras)

def sugerir_columna(df, palabras_clave, tipo_esperado=None):
    columnas = list(df.columns)
    for palabra in palabras_clave:
        for col in columnas:
            if palabra in col.lower():
                if tipo_esperado == 'numerico':
                    if pd.to_numeric(df[col], errors='coerce').notna().mean() > 0.7:
                        return col
                elif tipo_esperado == 'fecha':
                    if pd.to_datetime(df[col], errors='coerce', format='mixed').notna().mean() > 0.5:
                        return col
                else:
                    return col
    return columnas[0] if columnas else None

def limpiar_dataset_inteligente(df, col_fecha, col_rating, col_texto):
    reporte = []
    df = df.copy()
    filas_inicial = len(df)

    fechas = pd.to_datetime(df[col_fecha], errors='coerce', format='mixed')
    reporte.append(f"Fechas reconocidas correctamente: {fechas.notna().mean():.0%}")
    df['_fecha_limpia'] = fechas

    rating_num = pd.to_numeric(df[col_rating], errors='coerce')
    if rating_num.notna().sum() > 0:
        max_rating = rating_num.max()
        if max_rating > 5:
            rating_num = (rating_num / max_rating) * 5
            reporte.append(f"Rating detectado en escala 0-{int(max_rating)}, normalizado a escala 1-5")
    df['_rating_limpio'] = rating_num

    texto_col = df[col_texto].astype(str)
    pct_texto_unico = texto_col.nunique() / max(len(texto_col), 1)
    if pct_texto_unico < 0.3:
        reporte.append(
            f"⚠️ Advertencia: solo {pct_texto_unico:.0%} del texto en '{col_texto}' es único — "
            f"puede que NO sea el texto real de la reseña."
        )
    df['_texto_limpio'] = texto_col.str.strip().replace('nan', '')

    antes = len(df)
    df = df.dropna(subset=['_fecha_limpia', '_rating_limpio'])
    df = df[df['_texto_limpio'].str.len() > 3]
    if antes - len(df) > 0:
        reporte.append(f"Se eliminaron {antes - len(df)} filas sin fecha/rating válido o texto vacío")

    antes = len(df)
    df = df.drop_duplicates(subset=['_texto_limpio'])
    if antes - len(df) > 0:
        reporte.append(f"Se eliminaron {antes - len(df)} reseñas duplicadas (texto idéntico)")

    reporte.insert(0, f"Filas originales: {filas_inicial} → Filas limpias: {len(df)}")
    return df, reporte

@st.cache_data(show_spinner=False)
def analizar(df, col_fecha, col_rating, col_texto):
    from textblob import TextBlob
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LinearRegression
    import numpy as np

    stop_words = cargar_stopwords()
    df = df.dropna(subset=[col_fecha, col_rating, col_texto]).copy()
    df['fecha'] = pd.to_datetime(df[col_fecha], errors='coerce')
    df = df.dropna(subset=['fecha'])
    df['texto_limpio'] = df[col_texto].apply(lambda t: limpiar_texto(t, stop_words))
    df['sentimiento'] = df['texto_limpio'].apply(lambda t: TextBlob(t).sentiment.polarity)

    semanal = df.groupby(df['fecha'].dt.to_period('W'))['sentimiento'].mean().reset_index()
    semanal['semana_num'] = range(len(semanal))

    negativas = df[df[col_rating] <= 2]['texto_limpio']
    top_temas = []
    if len(negativas) >= 5:
        vec = CountVectorizer(max_features=15, ngram_range=(1, 2))
        vec.fit_transform(negativas)
        top_temas = list(vec.get_feature_names_out())

    prediccion = None
    if len(semanal) >= 5:
        X = semanal[['semana_num']]
        y = semanal['sentimiento']
        modelo = LinearRegression().fit(X, y)
        proximas = np.array([[semanal['semana_num'].max() + i] for i in range(1, 5)])
        prediccion = pd.DataFrame({
            'semana_num': proximas.flatten(),
            'sentimiento_predicho': modelo.predict(proximas)
        })

    return semanal, top_temas, prediccion, len(df)

def mostrar_resultados(semanal, top_temas, prediccion, total_filas):
    st.success(f"Análisis completado sobre {total_filas} reseñas.")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Evolución del sentimiento")
        st.line_chart(semanal.set_index("semana_num")["sentimiento"])

    with col2:
        st.subheader("Top temas de queja")
        if top_temas:
            st.table(pd.DataFrame({"término": top_temas}))
        else:
            st.info("No hay suficientes reseñas negativas (mínimo 5) para detectar temas.")

    if prediccion is not None:
        st.subheader("Proyección próximas 4 semanas")
        st.line_chart(prediccion.set_index("semana_num")["sentimiento_predicho"])
        pendiente = prediccion["sentimiento_predicho"].iloc[-1] - semanal["sentimiento"].iloc[-1]
        if pendiente < -0.05:
            st.error("⚠️ Alerta: tendencia a la baja en el sentimiento proyectado")
        else:
            st.success("✅ Sentimiento estable, sin señales de alerta")
    else:
        st.info("No hay suficientes semanas de datos (mínimo 5) para proyectar una tendencia.")

# ============================================================
# BARRA LATERAL — nombre de la app + controles PRIMERO
# ============================================================

st.sidebar.title("📊 Predictify")
st.sidebar.caption("Radar de Reputación de Producto")
st.sidebar.divider()

modo = st.sidebar.radio("¿Qué quieres ver?", ["Ejemplo (Fire Tablet / Echo)", "Subir mi propio archivo"])

archivo = None
df_nuevo = None
col_fecha = col_rating = col_texto = col_producto = None
productos_seleccionados = None
usar_filtro_fecha = False
fecha_inicio = fecha_fin = None
analizar_click = False

if modo == "Subir mi propio archivo":
    st.sidebar.divider()
    st.sidebar.subheader("1. Sube tu archivo")
    archivo = st.sidebar.file_uploader("Archivo CSV", type="csv")

    if archivo:
        df_nuevo = pd.read_csv(archivo, low_memory=False)

        st.sidebar.subheader("2. Columnas")
        sug_fecha = sugerir_columna(df_nuevo, ['date', 'fecha', 'time', 'reviewed'], 'fecha')
        sug_rating = sugerir_columna(df_nuevo, ['rating', 'star', 'score', 'calif'], 'numerico')
        sug_texto = sugerir_columna(df_nuevo, ['review', 'text', 'comment', 'body', 'reseña'])

        columnas = list(df_nuevo.columns)
        col_fecha = st.sidebar.selectbox("Fecha", columnas,
                                          index=columnas.index(sug_fecha) if sug_fecha in columnas else 0)
        col_rating = st.sidebar.selectbox("Rating", columnas,
                                           index=columnas.index(sug_rating) if sug_rating in columnas else 0)
        col_texto = st.sidebar.selectbox("Texto de reseña", columnas,
                                          index=columnas.index(sug_texto) if sug_texto in columnas else 0)
        opciones_producto = ["(Analizar todo, sin segmentar)"] + columnas
        col_producto = st.sidebar.selectbox("Producto (opcional)", opciones_producto)
        st.sidebar.caption("💡 Sugeridas automáticamente — verifica que tengan sentido.")

        st.sidebar.subheader("3. Segmentación")
        if col_producto != "(Analizar todo, sin segmentar)":
            top_productos = df_nuevo[col_producto].value_counts().head(15).index.tolist()
            productos_seleccionados = st.sidebar.multiselect(
                "¿Qué producto(s) comparar?",
                options=top_productos,
                default=top_productos[:2] if len(top_productos) >= 2 else top_productos,
            )

        usar_filtro_fecha = st.sidebar.checkbox("Filtrar por rango de fechas")
        if usar_filtro_fecha:
            fechas_preview = pd.to_datetime(df_nuevo[col_fecha], errors='coerce', format='mixed').dropna()
            if len(fechas_preview) > 0:
                fmin, fmax = fechas_preview.min().date(), fechas_preview.max().date()
                fecha_inicio, fecha_fin = st.sidebar.slider("Rango de fechas", min_value=fmin, max_value=fmax, value=(fmin, fmax))

        st.sidebar.subheader("4. Analizar")
        analizar_click = st.sidebar.button("🔍 Analizar", use_container_width=True)

# ============================================================
# ÁREA PRINCIPAL — resultados
# ============================================================

st.title("📊 Predictify")
st.caption("Radar de Reputación de Producto con Predicción Temprana")

if modo == "Ejemplo (Fire Tablet / Echo)":
    semanal = pd.read_csv("semanal_sentimiento.csv")
    temas_df = pd.read_csv("top_temas.csv")
    prediccion = pd.read_csv("prediccion.csv")
    st.info("Mostrando el análisis precargado de Fire Tablet y Echo (Amazon Consumer Reviews).")
    mostrar_resultados(semanal, list(temas_df["termino"]), prediccion, 14272)

else:
    if not archivo:
        st.info("⬅️ Sube un archivo CSV desde la barra lateral para comenzar.")
    else:
        st.write("Vista previa del archivo:", df_nuevo.head())

        if len({col_fecha, col_rating, col_texto}) < 3:
            st.error("⚠️ Elige 3 columnas distintas para fecha, rating y texto — no pueden repetirse.")
        elif analizar_click:
            with st.spinner("Limpiando y procesando datos..."):
                try:
                    df_trabajo = df_nuevo.copy()

                    if productos_seleccionados:
                        df_trabajo = df_trabajo[df_trabajo[col_producto].isin(productos_seleccionados)]
                    if usar_filtro_fecha and fecha_inicio and fecha_fin:
                        fechas_dt = pd.to_datetime(df_trabajo[col_fecha], errors='coerce', format='mixed')
                        df_trabajo = df_trabajo[(fechas_dt.dt.date >= fecha_inicio) & (fechas_dt.dt.date <= fecha_fin)]

                    if len(df_trabajo) == 0:
                        st.warning("No quedan reseñas después de aplicar los filtros.")
                    else:
                        df_limpio, reporte_limpieza = limpiar_dataset_inteligente(df_trabajo, col_fecha, col_rating, col_texto)

                        with st.expander("🧹 Reporte de limpieza"):
                            for linea in reporte_limpieza:
                                st.write("•", linea)

                        if len(df_limpio) < 5:
                            st.warning("Quedan muy pocas filas después de limpiar. Revisa columnas o filtros.")
                        else:
                            semanal, top_temas, prediccion, total = analizar(
                                df_limpio, '_fecha_limpia', '_rating_limpio', '_texto_limpio'
                            )

                            if len(semanal) == 0:
                                st.error("No se pudieron procesar fechas válidas después de la limpieza.")
                            else:
                                if productos_seleccionados and len(productos_seleccionados) > 1:
                                    st.subheader("Comparación de sentimiento entre productos seleccionados")
                                    comparacion = df_limpio.copy()
                                    comparacion['semana'] = comparacion['_fecha_limpia'].dt.to_period('W').astype(str)
                                    from textblob import TextBlob
                                    comparacion['sentimiento_cmp'] = comparacion['_texto_limpio'].apply(lambda t: TextBlob(t).sentiment.polarity)
                                    tabla_comparacion = comparacion.groupby(['semana', col_producto])['sentimiento_cmp'].mean().unstack()
                                    st.line_chart(tabla_comparacion)

                                mostrar_resultados(semanal, top_temas, prediccion, total)
                except Exception as e:
                    st.error(f"Ocurrió un error al procesar el archivo: {e}")
        else:
            st.info("Configura las columnas en la barra lateral y presiona '🔍 Analizar'.")
