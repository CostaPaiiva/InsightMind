import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

# Ajustes de performance/segurança
_MAX_PLOT_ROWS = 20000           # amostra p/ gráficos
_MAX_NUM_COLS_DIST = 6           # histos no máximo
_MAX_CAT_COLS = 6                # barras no máximo
_MAX_CORR_COLS = 25              # correlação no máximo
_MAX_CAT_CARDINALITY = 200       # se tiver mais que isso, evita plot (muito pesado)


def _sample_df(df: pd.DataFrame, max_rows: int = _MAX_PLOT_ROWS) -> pd.DataFrame:
    if df.shape[0] > max_rows:
        return df.sample(max_rows, random_state=42)
    return df


def render_visuals(df: pd.DataFrame):
    if df is None or df.empty:
        st.warning("Dataset vazio. Envie um CSV válido.")
        return

    dff = _sample_df(df)

    num = dff.select_dtypes(include=[np.number])
    cat = dff.select_dtypes(exclude=[np.number])

    # ----------------------------
    # Numéricas
    # ----------------------------
    if num.shape[1] > 0:
        st.markdown("#### 🔥 Distribuições (Top numéricas)")
        for c in list(num.columns[:_MAX_NUM_COLS_DIST]):
            try:
                fig = px.histogram(
                    dff,
                    x=c,
                    marginal="box",
                    nbins=40,
                    title=f"Distribuição: {c}",
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Não consegui plotar **{c}**: {e}")

        st.markdown("#### 🧊 Correlação")
        if num.shape[1] >= 2:
            try:
                # limita quantidade de colunas para não travar
                corr_cols = list(num.columns[:_MAX_CORR_COLS])
                corr = dff[corr_cols].corr(numeric_only=True)

                fig = px.imshow(
                    corr,
                    text_auto=True,
                    title=f"Matriz de Correlação (numéricas) — até {_MAX_CORR_COLS} colunas",
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Não consegui gerar correlação: {e}")
    else:
        st.info("Não há colunas numéricas para gerar distribuições/correlação.")

    # ----------------------------
    # Categóricas
    # ----------------------------
    if cat.shape[1] > 0:
        st.markdown("#### 🧩 Categóricas (Top colunas)")
        for c in list(cat.columns[:_MAX_CAT_COLS]):
            try:
                # evita travar em cardinalidade absurda
                nun = int(df[c].nunique(dropna=True))  # usa df original p/ cardinalidade real
                if nun > _MAX_CAT_CARDINALITY:
                    st.info(f"**{c}** tem alta cardinalidade ({nun} únicos). Pulando gráfico de categorias.")
                    continue

                vc = (
                    dff[c]
                    .astype(str)
                    .value_counts(dropna=False)
                    .head(20)
                    .reset_index()
                )
                vc.columns = [c, "count"]

                fig = px.bar(vc, x=c, y="count", title=f"Top categorias: {c}")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Não consegui plotar categórica **{c}**: {e}")
    else:
        st.info("Não há colunas categóricas para gerar gráficos.")


def build_report_figures(df: pd.DataFrame) -> list[bytes]:
    """
    Retorna lista de PNGs (bytes). Requer kaleido para fig.to_image().
    Otimizado para não travar com datasets grandes.
    """
    figs: list[bytes] = []
    if df is None or df.empty:
        return figs

    dff = _sample_df(df, max_rows=_MAX_PLOT_ROWS)

    num = dff.select_dtypes(include=[np.number])

    # Correlação (limitada)
    if num.shape[1] >= 2:
        try:
            corr_cols = list(num.columns[:_MAX_CORR_COLS])
            corr = dff[corr_cols].corr(numeric_only=True)
            fig = px.imshow(
                corr,
                text_auto=True,
                title=f"Matriz de Correlação (numéricas) — até {_MAX_CORR_COLS} colunas",
            )
            figs.append(fig.to_image(format="png", width=1200, height=900, scale=2))
        except Exception as e:
            # comum: kaleido ausente
            st.warning(f"Não consegui exportar correlação para PNG (verifique 'kaleido'): {e}")

    # Histogramas (limitado)
    for c in list(num.columns[:2]):
        try:
            fig = px.histogram(dff, x=c, marginal="box", nbins=40, title=f"Distribuição: {c}")
            figs.append(fig.to_image(format="png", width=1200, height=900, scale=2))
        except Exception as e:
            st.warning(f"Não consegui exportar histograma {c} para PNG (verifique 'kaleido'): {e}")

    return figs
