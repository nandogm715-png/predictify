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
    st.success("✅ Sentimiento estable, sin señales de alerta")
