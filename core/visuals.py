import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

def render_visuals(df: pd.DataFrame):
    num = df.select_dtypes(include=[np.number])
    cat = df.select_dtypes(exclude=[np.number])

    if num.shape[1] > 0:
        st.markdown("#### 🔥 Distribuições (Top numéricas)")
        for c in list(num.columns[:6]):
            fig = px.histogram(df, x=c, marginal="box", nbins=40, title=f"Distribuição: {c}")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 🧊 Correlação")
        if num.shape[1] >= 2:
            corr = num.corr(numeric_only=True)
            fig = px.imshow(corr, text_auto=True, title="Matriz de Correlação (numéricas)")
            st.plotly_chart(fig, use_container_width=True)

    if cat.shape[1] > 0:
        st.markdown("#### 🧩 Categóricas (Top colunas)")
        for c in list(cat.columns[:6]):
            vc = df[c].astype(str).value_counts().head(20).reset_index()
            vc.columns = [c, "count"]
            fig = px.bar(vc, x=c, y="count", title=f"Top categorias: {c}")
            st.plotly_chart(fig, use_container_width=True)

def build_report_figures(df: pd.DataFrame) -> list[bytes]:
    figs = []
    num = df.select_dtypes(include=[np.number])

    if num.shape[1] >= 2:
        corr = num.corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=True, title="Matriz de Correlação (numéricas)")
        figs.append(fig.to_image(format="png", width=1200, height=900, scale=2))

    for c in list(num.columns[:2]):
        fig = px.histogram(df, x=c, marginal="box", nbins=40, title=f"Distribuição: {c}")
        figs.append(fig.to_image(format="png", width=1200, height=900, scale=2))

    return figs
