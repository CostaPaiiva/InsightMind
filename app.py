import streamlit as st
import pandas as pd

from core.loader import load_csv_smart
from core.profiler import make_quality_metrics, basic_summary
from core.visuals import render_visuals, build_report_figures
from core.insights import generate_auto_insights
from core.cleaning import clean_dataset, cleaning_plan_from_df
from core.llm_chat import dataset_chat_answer
from core.report import build_html_report, build_pdf_report

st.set_page_config(page_title="InsightMind", layout="wide")
st.title("🧠 InsightMind — AutoDashboard + Chat IA + Limpeza + Relatório")

with st.sidebar:
    st.header("⚙️ Configurações")
    use_llm = st.toggle("Ativar Chat IA (LLM)", value=True)
    
    llm_provider = st.selectbox(
        "Provedor do Chat IA",
        ["auto", "openai", "ollama", "offline"],
        index=0,
        help="auto: tenta OpenAI → Ollama → offline"
    )
    max_rows_preview = st.slider("Linhas no preview", 10, 200, 50)
    st.markdown("---")
    file = st.file_uploader("📁 Envie um CSV", type=["csv"])

if not file:
    st.info("Envie um arquivo CSV para começar.")
    st.stop()

df, meta = load_csv_smart(file)
st.session_state["df_raw"] = df.copy()

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

st.subheader("🧾 Preview do dataset")
st.caption(
    f"Linhas: {df.shape[0]} | Colunas: {df.shape[1]} | Encoding: {meta.get('encoding')} | Sep: {meta.get('sep')}"
)
st.dataframe(df.head(max_rows_preview), use_container_width=True)

tabs = st.tabs(["📌 Resumo", "📈 Gráficos", "💬 Chat IA", "🧼 Limpeza", "🧾 Relatório"])

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
    render_visuals(df)

# --- Chat IA
with tabs[2]:
    st.markdown("### 💬 Pergunte ao seu dataset")
    st.caption("Ex.: “O que esse dataset diz?”, “Quais problemas de qualidade existem?”, “O que devo melhorar?”")

    if not use_llm:
        st.warning("Ative o Chat IA no menu lateral para usar o LLM.")
    else:
        default_q = "O que esse dataset diz? Traga visão geral, achados importantes, problemas de qualidade e recomendações práticas."
        user_q = st.text_input("Sua pergunta", value="")

        col_btn1, col_btn2 = st.columns([1, 1])
        ask_custom = col_btn1.button("(Perguntar)")
        ask_default = col_btn2.button("✨ O que esse dataset diz?")

        if ask_custom or ask_default:
            q = user_q.strip() if ask_custom else default_q
            if ask_custom and not q:
                st.warning("Digite uma pergunta ou use o botão padrão.")
            else:
                with st.spinner("Gerando resposta..."):
                    answer = dataset_chat_answer(
                        question=q,
                        df=df,
                        quality_metrics=make_quality_metrics(df),
                        auto_insights=generate_auto_insights(df, use_llm=False),
                        summary_table=basic_summary(df).head(30),
                    )
                st.session_state["chat_history"].append({"q": q, "a": answer})

        st.markdown("---")
        st.markdown("### Histórico")
        for i, item in enumerate(reversed(st.session_state["chat_history"]), 1):
            st.markdown(f"**Q{i}:** {item['q']}")
            st.write(item["a"])
            st.markdown("---")

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
        missing_threshold = st.slider("Limiar missing p/ remover (%)", 10, 95, int(plan_default["missing_threshold"] * 100))
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

    figs = build_report_figures(df_for_report)

    colA, colB = st.columns(2)
    with colA:
        if st.button("Gerar HTML"):
            with st.spinner("Montando HTML..."):
                html_bytes = build_html_report(
                    df_for_report,
                    qm_for_report,
                    insights_for_report,
                    include_profiling=include_profiling
                )
            st.download_button("⬇️ Baixar relatório HTML", data=html_bytes, file_name="relatorio.html", mime="text/html")

    with colB:
        if st.button("Gerar PDF"):
            with st.spinner("Montando PDF..."):
                pdf_bytes = build_pdf_report(df_for_report, qm_for_report, insights_for_report, figs)
            st.download_button("⬇️ Baixar relatório PDF", data=pdf_bytes, file_name="relatorio.pdf", mime="application/pdf")