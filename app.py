import streamlit as st
import pandas as pd
import uuid


from core.loader import load_csv_smart
from core.profiler import make_quality_metrics, basic_summary
from core.visuals import render_visuals, build_report_figures
from core.insights import generate_auto_insights
from core.cleaning import clean_dataset, cleaning_plan_from_df
from core.report import build_html_report, build_pdf_report

@st.cache_data(show_spinner=False)
def cached_quality(df: pd.DataFrame):
    return make_quality_metrics(df)


@st.cache_data(show_spinner=False)
def cached_summary(df: pd.DataFrame):
    return basic_summary(df)


@st.cache_data(show_spinner=False)
def cached_insights(df: pd.DataFrame):
    return generate_auto_insights(df, use_llm=False)


st.set_page_config(page_title="InsightMind", layout="wide")
st.title("🧠 InsightMind — AutoDashboard com Chat IA + Limpeza + Relatório")


# ----------------------------
# Wrappers opcionais (não quebrar)
# ----------------------------
def respond_with_ollama(user_prompt: str, history: list[dict]) -> str:
    df = st.session_state.get("df_clean", st.session_state.get("df_raw"))
    if df is None:
        return "Envie um CSV para começarmos."

    return dataset_chat_answer(
        question=user_prompt,
        df=df,
        quality_metrics=make_quality_metrics(df),
        auto_insights=generate_auto_insights(df, use_llm=False),
        summary_table=basic_summary(df).head(30),
        provider="ollama",
    )


def respond_with_fallback(user_prompt: str, history: list[dict]) -> str:
    df = st.session_state.get("df_clean", st.session_state.get("df_raw"))
    if df is None:
        return "Envie um CSV para começarmos."

    return dataset_chat_answer(
        question=user_prompt,
        df=df,
        quality_metrics=make_quality_metrics(df),
        auto_insights=generate_auto_insights(df, use_llm=False),
        summary_table=basic_summary(df).head(30),
        provider="offline",
    )


def llm_respond_fn(user_prompt: str, history: list[dict]) -> str:
    try:
        return respond_with_ollama(user_prompt, history)
    except Exception:
        return respond_with_fallback(user_prompt, history)


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


df, meta = load_csv_smart(file)
st.session_state["df_raw"] = df.copy()

# Histórico do chat do tab (mensagens individuais)
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

st.subheader("🧾 Preview do dataset")
st.caption(
    f"Linhas: {df.shape[0]} | Colunas: {df.shape[1]} | Encoding: {meta.get('encoding')} | Sep: {meta.get('sep')}"
)
st.dataframe(df.head(max_rows_preview), use_container_width=True)

tabs = st.tabs(["📌 Resumo", "📈 Gráficos", "✅ Diagnóstico", "🧼 Limpeza", "🧾 Relatório"])


# --- Resumo
with tabs[0]:
    colA, colB = st.columns([1, 1])
    with colA:
        st.markdown("### Resumo Estatístico")
        summary_df = basic_summary(df)
        st.dataframe(summary_df, use_container_width=True)
    with colB:
        st.markdown("### Métricas de Qualidade")
        qm = make_quality_metrics(df)
        st.json(qm)


# --- Gráficos
with tabs[1]:
    st.markdown("### Visualizações Avançadas")
    # render_visuals(df)


# --- 1. INICIALIZAÇÃO (Coloque no início do script, fora das abas) ---
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# --- DENTRO DA ABA DE CHAT ---
with tabs[2]:
    st.markdown("### ✅ Diagnóstico Automático do Dataset")
    st.caption("Análise automática: qualidade, riscos, insights e recomendações.")

    df_diag = st.session_state.get("df_clean", df)

    # cache (você já usa algo parecido)
    qm = make_quality_metrics(df_diag)
    summary_df = basic_summary(df_diag)
    insights = generate_auto_insights(df_diag, use_llm=False)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 📊 Métricas de Qualidade")
        st.json(qm)

    with col2:
        st.markdown("#### 🧾 Resumo Estatístico (top 30)")
        st.dataframe(summary_df.head(30), use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🔥 Principais Problemas (prioridade)")

    # Exemplo de priorização (ajusta conforme seu qm real)
    issues = []

    # tenta usar chaves comuns (se existirem no seu make_quality_metrics)
    miss_rate = qm.get("missing_rate", None)
    dup_rows = qm.get("duplicate_rows", None)
    const_cols = qm.get("constant_cols", None)
    high_missing_cols = qm.get("high_missing_cols", None)

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
    if not insights:
        st.info("Sem insights automáticos relevantes.")
    else:
        for it in insights[:15]:
            st.markdown(f"- {it}")

    st.markdown("---")
    st.markdown("#### ✅ Recomendações Práticas (automáticas)")
    st.markdown(
        "\n".join([
            "- Trate missing nas colunas mais críticas (imputar/remover conforme o caso).",
            "- Remova duplicadas e colunas constantes (se existirem).",
            "- Padronize strings e valide datas (parse e consistência).",
            "- Revise outliers em numéricas (IQR/clip) se distorcem métricas.",
            "- Se houver colunas ID com alta cardinalidade, evite usar diretamente como feature."
        ])
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

    if "df_clean" in st.session_state:
        st.markdown("#### 📄 Log da limpeza")
        for item in st.session_state["clean_log"]:
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
    qm_for_report = make_quality_metrics(df_for_report)
    insights_for_report = generate_auto_insights(df_for_report, use_llm=False)

    include_profiling = st.checkbox("Incluir profiling (HTML) do ydata-profiling", value=True)

    colA, colB = st.columns(2)
    with colA:
        if st.button("Gerar HTML"):
            with st.spinner("Montando HTML..."):
                html_bytes = build_html_report(
                    df_for_report,
                    qm_for_report,
                    insights_for_report,
                    include_profiling=include_profiling,
                )
            st.download_button(
                "⬇️ Baixar relatório HTML",
                data=html_bytes,
                file_name="relatorio.html",
                mime="text/html",
            )

    with colB:
        if st.button("Gerar PDF"):
            with st.spinner("Montando PDF..."):
                figs = build_report_figures(df_for_report)
                pdf_bytes = build_pdf_report(df_for_report, qm_for_report, insights_for_report, figs)

            st.download_button(
                "⬇️ Baixar relatório PDF",
                data=pdf_bytes,
                file_name="relatorio.pdf",
                mime="application/pdf",
            )
