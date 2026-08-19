import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px

st.set_page_config(page_title="Predictify — Radar de Reputación", layout="wide")

# ============================================================
# ESQUEMA ESTÁNDAR INTERNO: fecha, rating, texto, entidad
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

def mapear_a_esquema_estandar(df, col_fecha, col_rating, col_texto, col_entidad):
    estandar = pd.DataFrame()
    estandar['fecha'] = pd.to_datetime(df[col_fecha], errors='coerce', format='mixed')
    estandar['rating'] = pd.to_numeric(df[col_rating], errors='coerce')
    estandar['texto'] = df[col_texto].astype(str)
    estandar['entidad'] = df[col_entidad].astype(str) if col_entidad else 'General'
    return estandar

def limpiar_dataset_inteligente(df):
    reporte = []
    df = df.copy()
    filas_inicial = len(df)

    reporte.append(f"Fechas reconocidas correctamente: {df['fecha'].notna().mean():.0%}")

    if df['rating'].notna().sum() > 0:
        max_rating = df['rating'].max()
        if max_rating > 5:
            df['rating'] = (df['rating'] / max_rating) * 5
            reporte.append(f"Rating detectado en escala 0-{int(max_rating)}, normalizado a escala 1-5")

    pct_texto_unico = df['texto'].nunique() / max(len(df), 1)
    if pct_texto_unico < 0.3:
        reporte.append(
            f"⚠️ Advertencia: solo {pct_texto_unico:.0%} del texto es único — puede que la columna elegida no sea la reseña real."
        )
    df['texto'] = df['texto'].str.strip().replace('nan', '')

    antes = len(df)
    df = df.dropna(subset=['fecha', 'rating'])
    df = df[df['texto'].str.len() > 3]
    if antes - len(df) > 0:
        reporte.append(f"Se eliminaron {antes - len(df)} filas sin fecha/rating válido o texto vacío")

    antes = len(df)
    df = df.drop_duplicates(subset=['texto'])
    if antes - len(df) > 0:
        reporte.append(f"Se eliminaron {antes - len(df)} reseñas duplicadas (texto idéntico)")

    reporte.insert(0, f"Filas originales: {filas_inicial} → Filas limpias: {len(df)}")
    return df, reporte

@st.cache_data(show_spinner=False)
def calcular_sentimiento(df):
    from textblob import TextBlob
    stop_words = cargar_stopwords()
    df = df.copy()
    df['texto_limpio'] = df['texto'].apply(lambda t: limpiar_texto(t, stop_words))
    df['sentimiento'] = df['texto_limpio'].apply(lambda t: TextBlob(t).sentiment.polarity)
    return df

@st.cache_data(show_spinner=False)
def calcular_temas(df, min_negativas=5, max_temas=15):
    from sklearn.feature_extraction.text import CountVectorizer
    negativas = df[df['rating'] <= 2]
    if len(negativas) < min_negativas:
        return pd.DataFrame(columns=['termino', 'frecuencia', 'sentimiento_promedio'])

    vec = CountVectorizer(max_features=max_temas, ngram_range=(1, 2))
    matriz = vec.fit_transform(negativas['texto_limpio'])
    terminos = vec.get_feature_names_out()
    frecuencias = matriz.sum(axis=0).A1

    filas = []
    for termino, freq in zip(terminos, frecuencias):
        mask = negativas['texto_limpio'].str.contains(termino, regex=False)
        filas.append({
            'termino': termino,
            'frecuencia': int(freq),
            'sentimiento_promedio': negativas.loc[mask, 'sentimiento'].mean()
        })
    return pd.DataFrame(filas).sort_values('frecuencia', ascending=False)

@st.cache_data(show_spinner=False)
def calcular_tendencia_semanal(df):
    semanal = df.groupby(df['fecha'].dt.to_period('W'))['sentimiento'].agg(['mean', 'count']).reset_index()
    semanal.columns = ['periodo', 'sentimiento', 'n_reseñas']
    semanal['fecha'] = semanal['periodo'].dt.start_time
    semanal['semana_num'] = range(len(semanal))
    return semanal

@st.cache_data(show_spinner=False)
def calcular_prediccion(semanal, semanas_futuras=4):
    from sklearn.linear_model import LinearRegression
    if len(semanal) < 5:
        return None
    X = semanal[['semana_num']]
    y = semanal['sentimiento']
    modelo = LinearRegression().fit(X, y)
    ultimo_num = semanal['semana_num'].max()
    ultima_fecha = semanal['fecha'].max()
    proximas_num = np.array([[ultimo_num + i] for i in range(1, semanas_futuras + 1)])
    proximas_fechas = [ultima_fecha + pd.Timedelta(weeks=i) for i in range(1, semanas_futuras + 1)]
    pred = modelo.predict(proximas_num)
    pred_df = pd.DataFrame({'semana_num': proximas_num.flatten(), 'fecha': proximas_fechas, 'sentimiento_predicho': pred})
    return pred_df, modelo.coef_[0]

# ============================================================
# KPIs
# ============================================================

def mostrar_kpis(df, semanal, prediccion_info):
    total = len(df)
    rating_prom = df['rating'].mean()
    pct_negativas = (df['rating'] <= 2).mean() * 100
    sentimiento_actual = semanal['sentimiento'].iloc[-1] if len(semanal) else np.nan

    if prediccion_info:
        _, pendiente = prediccion_info
        if pendiente > 0.001:
            tendencia = "📈 Mejorando"
        elif pendiente < -0.001:
            tendencia = "📉 Deteriorando"
        else:
            tendencia = "➖ Estable"
    else:
        tendencia = "— Sin datos suficientes"

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total reseñas", f"{total:,}")
    c2.metric("Rating promedio", f"{rating_prom:.2f} ★")
    c3.metric("% Reseñas negativas", f"{pct_negativas:.1f}%")
    c4.metric("Sentimiento actual", f"{sentimiento_actual:.2f}")
    c5.metric("Tendencia proyectada", tendencia)

# ============================================================
# PESTAÑAS
# ============================================================

def tab_tendencias(df, semanal, prediccion_info):
    st.subheader("Evolución del sentimiento en el tiempo")
    entidades = sorted(df['entidad'].unique())
    seleccion = st.multiselect("Filtrar por entidad/producto", entidades,
                                default=entidades[:3] if len(entidades) > 3 else entidades, key="tend_filtro")

    df_f = df[df['entidad'].isin(seleccion)] if seleccion else df
    semanal_ent = df_f.groupby([df_f['fecha'].dt.to_period('W'), 'entidad'])['sentimiento'].mean().reset_index()
    semanal_ent.columns = ['periodo', 'entidad', 'sentimiento']
    semanal_ent['fecha'] = semanal_ent['periodo'].dt.start_time

    titulo = ', '.join(seleccion) if len(seleccion) <= 3 else f"{len(seleccion)} entidades"
    fig = px.line(semanal_ent, x='fecha', y='sentimiento', color='entidad', markers=True,
                  title=f"Sentimiento promedio semanal — {titulo}",
                  labels={'fecha': 'Semana', 'sentimiento': 'Sentimiento promedio', 'entidad': 'Producto/Empresa'})
    fig.add_hline(y=0, line_dash="dot", line_color="gray")

    if prediccion_info:
        pred_df, _ = prediccion_info
        fig.add_scatter(x=pred_df['fecha'], y=pred_df['sentimiento_predicho'], mode='lines+markers',
                        name='Proyección (general)', line=dict(dash='dash', color='red'))

    st.plotly_chart(fig, use_container_width=True)
    st.caption("💡 La línea punteada roja es la proyección del modelo sobre el conjunto general, no solo lo filtrado.")

    st.subheader("Volumen de reseñas por semana")
    fig_vol = px.bar(semanal, x='fecha', y='n_reseñas', title="Cantidad de reseñas por semana (conjunto completo)")
    st.plotly_chart(fig_vol, use_container_width=True)
    st.caption("💡 Semanas con muy pocas reseñas pueden generar señales de sentimiento poco confiables.")


def tab_distribuciones(df):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribución de ratings")
        fig = px.histogram(df, x='rating', nbins=5, title="Cantidad de reseñas por rating")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Distribución de sentimiento")
        fig = px.histogram(df, x='sentimiento', nbins=30, title="Distribución del puntaje de sentimiento")
        fig.add_vline(x=0, line_dash="dot", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Longitud del texto según rating")
    df_plot = df.copy()
    df_plot['longitud_texto'] = df_plot['texto'].str.len()
    fig = px.box(df_plot, x='rating', y='longitud_texto', title="Longitud del texto de reseña por rating")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("💡 Si las reseñas negativas son más largas, sugiere que los clientes insatisfechos dan más contexto.")


def tab_correlaciones(df):
    st.subheader("Rating vs. sentimiento del texto")
    df_plot = df.copy()
    df_plot['rating_str'] = df_plot['rating'].round().astype(int).astype(str)
    fig = px.box(df_plot, x='rating_str', y='sentimiento', color='rating_str',
                title="Distribución del sentimiento del texto según el rating dado",
                labels={'rating_str': 'Rating', 'sentimiento': 'Sentimiento del texto'})
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("💡 Cajas que se superponen entre ratings revelan discrepancias entre lo calificado y lo escrito.")

    corr = df['rating'].corr(df['sentimiento'])
    st.metric("Correlación rating ↔ sentimiento", f"{corr:.2f}",
              help="1 = coinciden perfectamente. 0 = no hay relación.")

    st.subheader("Volumen de reseñas vs. sentimiento promedio (por semana)")
    semanal = calcular_tendencia_semanal(df)
    fig2 = px.scatter(semanal, x='n_reseñas', y='sentimiento', size='n_reseñas',
                      title="¿El volumen de reseñas se relaciona con el sentimiento de esa semana?")
    st.plotly_chart(fig2, use_container_width=True)


def tab_temas(temas_df):
    st.subheader("Temas de queja más frecuentes (en reseñas negativas)")
    if temas_df.empty:
        st.info("No hay suficientes reseñas negativas para detectar temas (mínimo 5).")
        return

    fig = px.bar(temas_df.sort_values('frecuencia'), x='frecuencia', y='termino', orientation='h',
                color='sentimiento_promedio', color_continuous_scale='RdYlGn',
                title="Frecuencia de términos en quejas, coloreado por su sentimiento promedio")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("💡 Rojo = el término aparece en reseñas con sentimiento más negativo dentro del grupo de quejas.")
    st.dataframe(temas_df, use_container_width=True, hide_index=True)


def tab_comparativa(df):
    top_entidades = df['entidad'].value_counts().head(15).index.tolist()
    seleccion = st.multiselect("Compara entidades/productos", top_entidades,
                                default=top_entidades[:3] if len(top_entidades) >= 3 else top_entidades,
                                key="comp_filtro")
    if not seleccion:
        st.info("Selecciona al menos una entidad para comparar.")
        return

    resumen = df[df['entidad'].isin(seleccion)].groupby('entidad').agg(
        reseñas=('rating', 'count'),
        rating_promedio=('rating', 'mean'),
        sentimiento_promedio=('sentimiento', 'mean'),
        pct_negativas=('rating', lambda x: (x <= 2).mean() * 100)
    ).reset_index()

    st.dataframe(resumen.style.format({
        'rating_promedio': '{:.2f}', 'sentimiento_promedio': '{:.2f}', 'pct_negativas': '{:.1f}%'
    }), use_container_width=True, hide_index=True)

    fig = px.bar(resumen, x='entidad', y='pct_negativas', color='sentimiento_promedio',
                color_continuous_scale='RdYlGn', title="% de reseñas negativas por entidad")
    st.plotly_chart(fig, use_container_width=True)

    df_comp = df[df['entidad'].isin(seleccion)]
    semanal_comp = df_comp.groupby([df_comp['fecha'].dt.to_period('W'), 'entidad'])['sentimiento'].mean().reset_index()
    semanal_comp.columns = ['periodo', 'entidad', 'sentimiento']
    semanal_comp['fecha'] = semanal_comp['periodo'].dt.start_time
    fig2 = px.line(semanal_comp, x='fecha', y='sentimiento', color='entidad', markers=True,
                  title="Evolución de sentimiento — comparación entre entidades")
    st.plotly_chart(fig2, use_container_width=True)


def render_dashboard(df):
    semanal = calcular_tendencia_semanal(df)
    prediccion_info = calcular_prediccion(semanal)
    temas_df = calcular_temas(df)

    mostrar_kpis(df, semanal, prediccion_info)
    st.divider()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Tendencias", "📊 Distribuciones", "🔗 Correlaciones", "💬 Temas de queja", "⚖️ Comparativa"
    ])
    with tab1: tab_tendencias(df, semanal, prediccion_info)
    with tab2: tab_distribuciones(df)
    with tab3: tab_correlaciones(df)
    with tab4: tab_temas(temas_df)
    with tab5: tab_comparativa(df)

# ============================================================
# BARRA LATERAL
# ============================================================

st.sidebar.title("📊 Predictify")
st.sidebar.caption("Radar de Reputación de Producto")
st.sidebar.divider()

modo = st.sidebar.radio("¿Qué quieres ver?", ["Ejemplo (Fire Tablet / Echo)", "Subir mi propio archivo"])

df_final = None

if modo == "Ejemplo (Fire Tablet / Echo)":
    df_final = pd.read_csv("datos_ejemplo.csv")
    df_final['fecha'] = pd.to_datetime(df_final['fecha'])

else:
    st.sidebar.subheader("1. Sube tu archivo")
    archivo = st.sidebar.file_uploader("Archivo CSV", type="csv")

    if archivo:
        df_nuevo = pd.read_csv(archivo, low_memory=False)

        st.sidebar.subheader("2. Mapeo de columnas")
        sug_fecha = sugerir_columna(df_nuevo, ['date', 'fecha', 'time', 'reviewed'], 'fecha')
        sug_rating = sugerir_columna(df_nuevo, ['rating', 'star', 'score', 'calif'], 'numerico')
        sug_texto = sugerir_columna(df_nuevo, ['review', 'text', 'comment', 'body', 'reseña'])

        columnas = list(df_nuevo.columns)
        col_fecha = st.sidebar.selectbox("Fecha", columnas, index=columnas.index(sug_fecha) if sug_fecha in columnas else 0)
        col_rating = st.sidebar.selectbox("Rating", columnas, index=columnas.index(sug_rating) if sug_rating in columnas else 0)
        col_texto = st.sidebar.selectbox("Texto de reseña", columnas, index=columnas.index(sug_texto) if sug_texto in columnas else 0)
        opciones_entidad = ["(Sin segmentar — 'General')"] + columnas
        col_entidad_sel = st.sidebar.selectbox("Producto / Empresa (opcional)", opciones_entidad)
        col_entidad = col_entidad_sel if col_entidad_sel != "(Sin segmentar — 'General')" else None
        st.sidebar.caption("💡 Sugeridas automáticamente — verifica que tengan sentido.")

        if len({col_fecha, col_rating, col_texto}) < 3:
            st.sidebar.error("⚠️ Elige 3 columnas distintas para fecha, rating y texto.")
        else:
            st.sidebar.subheader("3. Analizar")
            if st.sidebar.button("🔍 Procesar y analizar", use_container_width=True):
                with st.spinner("Mapeando, limpiando y analizando..."):
                    df_estandar = mapear_a_esquema_estandar(df_nuevo, col_fecha, col_rating, col_texto, col_entidad)
                    df_limpio, reporte = limpiar_dataset_inteligente(df_estandar)
                    st.session_state['reporte_limpieza'] = reporte

                    if len(df_limpio) < 5:
                        st.warning("Quedan muy pocas filas después de limpiar. Revisa las columnas seleccionadas.")
                        st.session_state['df_procesado'] = None
                    else:
                        st.session_state['df_procesado'] = calcular_sentimiento(df_limpio)

    if st.session_state.get('df_procesado') is not None:
        df_final = st.session_state['df_procesado']

# ============================================================
# ÁREA PRINCIPAL
# ============================================================

st.title("📊 Predictify")
st.caption("Radar de Reputación de Producto con Predicción Temprana")

if modo == "Ejemplo (Fire Tablet / Echo)":
    st.info("Mostrando el dataset de ejemplo — Fire Tablet y Echo (Amazon Consumer Reviews).")
    render_dashboard(df_final)
else:
    if 'reporte_limpieza' in st.session_state:
        with st.expander("🧹 Reporte de limpieza"):
            for linea in st.session_state['reporte_limpieza']:
                st.write("•", linea)
    if df_final is None:
        st.info("⬅️ Sube un archivo CSV, mapea las columnas y presiona 'Procesar y analizar'.")
    else:
        render_dashboard(df_final)
