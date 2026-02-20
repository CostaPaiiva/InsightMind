import streamlit as st
import pandas as pd
import uuid

from core.loader import load_csv_smart
from core.profiler import make_quality_metrics, basic_summary
from core.visuals import render_visuals, build_report_figures
from core.insights import generate_auto_insights
from core.cleaning import clean_dataset, cleaning_plan_from_df
from core.report import build_html_report, build_pdf_report


# ----------------------------
# Cache pesado (ganho grande)
# ----------------------------
@st.cache_data(show_spinner=False)
def cached_load_csv(file) -> tuple[pd.DataFrame, dict]:
    # Cacheia leitura/parsing do CSV
    return load_csv_smart(file)


@st.cache_data(show_spinner=False)
def cached_quality(df: pd.DataFrame):
    return make_quality_metrics(df)


@st.cache_data(show_spinner=False)
def cached_summary(df: pd.DataFrame):
    return basic_summary(df)


@st.cache_data(show_spinner=False)
def cached_insights(df: pd.DataFrame):
    return generate_auto_insights(df, use_llm=False)


# ----------------------------
# App
# ----------------------------
st.set_page_config(page_title="InsightMind", layout="wide")
st.title("🧠 InsightMind — AutoDashboard + Limpeza + Relatório")


# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("⚙️ Configurações")
    max_rows_preview = st.slider("Linhas no preview", 10, 200, 50)
    st.markdown("---")
    file = st.file_uploader("📁 Envie um CSV", type=["csv"])


if not file:
    st.info("Envie um arquivo CSV para começar.")
    st.stop()


# ✅ leitura cacheada
df, meta = cached_load_csv(file)
st.session_state["df_raw"] = df  # sem copy() para não gastar memória


# Estado inicial
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "df_clean" not in st.session_state:
    # não cria df_clean aqui; só quando usuário aplicar limpeza
    pass


# Preview
st.subheader("🧾 Preview do dataset")
st.caption(
    f"Linhas: {df.shape[0]} | Colunas: {df.shape[1]} | Encoding: {meta.get('encoding')} | Sep: {meta.get('sep')}"
)

df_preview = df.head(max_rows_preview)
st.dataframe(df_preview, use_container_width=True)


tabs = st.tabs(["📌 Resumo", "📈 Gráficos", "✅ Diagnóstico", "🧼 Limpeza", "🧾 Relatório"])


# --- Resumo
with tabs[0]:
    colA, colB = st.columns([1, 1])

    with colA:
        st.markdown("### Resumo Estatístico")
        # ✅ cacheado
        summary_df = cached_summary(df)
        st.dataframe(summary_df.head(200), use_container_width=True)

    with colB:
        st.markdown("### Métricas de Qualidade")
        # ✅ cacheado
        qm = cached_quality(df)
        st.json(qm)


# --- Gráficos
with tabs[1]:
    st.markdown("### Visualizações Avançadas")
    st.caption("Para evitar lentidão, os gráficos só são gerados quando você clicar no botão.")

    df_plot = st.session_state.get("df_clean", df)

    if st.button("📈 Gerar gráficos"):
        try:
            render_visuals(df_plot)
        except Exception as e:
            st.error(f"Erro ao renderizar gráficos: {e}")
            st.exception(e)
    else:
        st.info("Clique em **📈 Gerar gráficos** para carregar as visualizações.")


# --- Diagnóstico
with tabs[2]:
    st.markdown("### ✅ Diagnóstico Automático do Dataset")
    st.caption("Análise automática: qualidade, riscos, insights e recomendações.")

    df_diag = st.session_state.get("df_clean", df)

    # ✅ cacheado
    qm_diag = cached_quality(df_diag)
    summary_diag = cached_summary(df_diag)
    insights_diag = cached_insights(df_diag)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("#### 📊 Métricas de Qualidade")
        st.json(qm_diag)

    with col2:
        st.markdown("#### 🧾 Resumo Estatístico (top 30)")
        st.dataframe(summary_diag.head(30), use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🔥 Principais Problemas (prioridade)")

    issues = []
    miss_rate = qm_diag.get("missing_rate", None)
    dup_rows = qm_diag.get("duplicate_rows", None)
    const_cols = qm_diag.get("constant_cols", None)
    high_missing_cols = qm_diag.get("high_missing_cols", None)

    if miss_rate is not None and miss_rate > 0:
        issues.append(("Missing elevado", f"Taxa de missing: {miss_rate}"))
    if dup_rows:
        issues.append(("Duplicadas", f"Linhas duplicadas: {dup_rows}"))
    if const_cols:
        issues.append(("Colunas constantes", f"{const_cols}"))
    if high_missing_cols:
        issues.append(("Colunas com missing alto", f"{high_missing_cols}"))

    if not issues:
        st.success("Nenhum problema crítico detectado nas métricas principais.")
    else:
        for title, detail in issues[:10]:
            st.warning(f"**{title}** — {detail}")

    st.markdown("---")
    st.markdown("#### 💡 Insights Automáticos")
    if not insights_diag:
        st.info("Sem insights automáticos relevantes.")
    else:
        for it in insights_diag[:15]:
            st.markdown(f"- {it}")

    st.markdown("---")
    st.markdown("#### ✅ Recomendações Práticas (automáticas)")
    st.markdown(
        "\n".join(
            [
                "- Trate missing nas colunas mais críticas (imputar/remover conforme o caso).",
                "- Remova duplicadas e colunas constantes (se existirem).",
                "- Padronize strings e valide datas (parse e consistência).",
                "- Revise outliers em numéricas (IQR/clip) se distorcem métricas.",
                "- Se houver colunas ID com alta cardinalidade, evite usar diretamente como feature.",
            ]
        )
    )


# --- Limpeza
with tabs[3]:
    st.markdown("### 🧼 Modo Limpar Dataset")
    st.caption("Pipeline automático + opções. Você pode baixar o CSV tratado no final.")

    plan_default = cleaning_plan_from_df(df)

    col1, col2, col3 = st.columns(3)
    with col1:
        remove_duplicates = st.checkbox("Remover duplicadas", value=plan_default["remove_duplicates"])
        trim_strings = st.checkbox("Padronizar strings (strip/lower)", value=plan_default["trim_strings"])
        parse_dates = st.checkbox("Tentar converter datas", value=plan_default["parse_dates"])
    with col2:
        drop_high_missing = st.checkbox("Remover colunas com missing alto", value=plan_default["drop_high_missing"])
        missing_threshold = st.slider(
            "Limiar missing p/ remover (%)",
            10,
            95,
            int(plan_default["missing_threshold"] * 100),
        )
        impute_numeric = st.selectbox("Imputação numérica", ["median", "mean", "none"], index=0)
    with col3:
        impute_categorical = st.selectbox("Imputação categórica", ["mode", "none"], index=0)
        drop_constant_cols = st.checkbox("Remover colunas constantes", value=plan_default["drop_constant_cols"])
        outlier_clip = st.checkbox("Clip de outliers (IQR)", value=plan_default["outlier_clip"])

    if st.button("Aplicar limpeza"):
        cleaned, log = clean_dataset(
            df=df,
            remove_duplicates=remove_duplicates,
            trim_strings=trim_strings,
            parse_dates=parse_dates,
            drop_high_missing=drop_high_missing,
            missing_threshold=missing_threshold / 100.0,
            impute_numeric=impute_numeric,
            impute_categorical=impute_categorical,
            drop_constant_cols=drop_constant_cols,
            outlier_clip=outlier_clip,
        )
        st.session_state["df_clean"] = cleaned
        st.session_state["clean_log"] = log
        st.success("Limpeza aplicada!")

        # ✅ opcional: limpar caches dependentes (quando df_clean muda, caches do df original não atrapalham;
        # mas se você quiser forçar recálculo, descomente):
        # st.cache_data.clear()

    if "df_clean" in st.session_state:
        st.markdown("#### 📄 Log da limpeza")
        for item in st.session_state.get("clean_log", []):
            st.write(f"- {item}")

        st.markdown("#### ✅ Preview do dataset tratado")
        st.dataframe(st.session_state["df_clean"].head(max_rows_preview), use_container_width=True)

        st.download_button(
            "⬇️ Baixar CSV tratado",
            data=st.session_state["df_clean"].to_csv(index=False).encode("utf-8"),
            file_name="dataset_tratado.csv",
            mime="text/csv",
        )


# --- Relatório
with tabs[4]:
    st.markdown("### 🧾 Relatório HTML/PDF (gráficos + insights)")
    st.caption("Gera um HTML interativo e um PDF (com imagens dos principais gráficos).")

    df_for_report = st.session_state.get("df_clean", df)

    # --- Checagem segura do profiling (sem derrubar o app)
    profiling_available = True
    profiling_error = None
    try:
        import pkg_resources  # noqa: F401
        from ydata_profiling import ProfileReport  # noqa: F401
    except Exception as e:
        profiling_available = False
        profiling_error = e

    include_profiling = st.checkbox(
        "Incluir profiling (HTML) do ydata-profiling",
        value=True,
        disabled=not profiling_available,
    )

    colA, colB = st.columns(2)
    with colA:
        if st.button("Gerar HTML"):
            with st.spinner("Montando HTML..."):
                # ✅ cacheado (rápido)
                qm_for_report = cached_quality(df_for_report)
                insights_for_report = cached_insights(df_for_report)

                try:
                    html_bytes = build_html_report(
                        df_for_report,
                        qm_for_report,
                        insights_for_report,
                        include_profiling=include_profiling,
                    )
                except Exception as e:
                    st.error(f"Erro ao gerar HTML: {e}")
                    st.exception(e)
                    html_bytes = None

            if html_bytes is not None:
                st.download_button(
                    "⬇️ Baixar relatório HTML",
                    data=html_bytes,
                    file_name="relatorio.html",
                    mime="text/html",
                )

    with colB:
        if st.button("Gerar PDF"):
            with st.spinner("Montando PDF..."):
                # ✅ cacheado + gera figs só no clique
                qm_for_report = cached_quality(df_for_report)
                insights_for_report = cached_insights(df_for_report)

                figs = build_report_figures(df_for_report)
                pdf_bytes = build_pdf_report(df_for_report, qm_for_report, insights_for_report, figs)

            st.download_button(
                "⬇️ Baixar relatório PDF",
                data=pdf_bytes,
                file_name="relatorio.pdf",
                mime="application/pdf",
            )