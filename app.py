import streamlit as st
import pandas as pd
import uuid

from core.loader import load_csv_smart
from core.profiler import make_quality_metrics, basic_summary
from core.visuals import render_visuals, build_report_figures
from core.insights import generate_auto_insights
from core.cleaning import clean_dataset, cleaning_plan_from_df
from core.llm_chat import dataset_chat_answer
from core.report import build_html_report, build_pdf_report
from core.offline_chat import offline_answer


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
    use_llm = st.toggle("Ativar Chat IA (LLM)", value=True)

    llm_provider = st.selectbox(
        "Provedor do Chat IA",
        ["auto", "openai", "ollama", "offline"],
        index=0,
        help="auto: tenta OpenAI → Ollama → offline",
    )

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
    # render_visuals(df)


# --- 1. INICIALIZAÇÃO ---
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

provider_effective = llm_provider if use_llm else "offline"

with tabs[2]:
    st.markdown("### 💬 Chat InsightMind")
    
    provider_effective = llm_provider if use_llm else "offline"

    # --- 1. ÁREA DE INPUT ---
    # O segredo: Não processar o histórico no mesmo frame que o input se possível
    prompt = st.chat_input("Digite sua pergunta aqui...")

    if st.button("🗑️ Limpar Conversa"):
        st.session_state["chat_history"] = []
        st.rerun()

    # --- 2. PROCESSAMENTO ---
    if prompt:
        # Adicionamos ao histórico (usando append para manter a ordem natural no State)
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        
        try:
            with st.spinner("Analisando..."):
                if provider_effective == "offline":
                    answer = offline_answer(prompt, df, st.session_state.get("qm_cached"), st.session_state.get("insights_cached"), st.session_state.get("summary_cached"))
                else:
                    answer = dataset_chat_answer(prompt, df, st.session_state.get("qm_cached"), st.session_state.get("insights_cached"), st.session_state.get("summary_cached"), provider_effective)
            
            st.session_state["chat_history"].append({"role": "assistant", "content": answer})
            st.rerun()
        except Exception as e:
            st.error(f"Erro: {e}")

    st.markdown("---")

    # --- 3. EXIBIÇÃO (A PARTE QUE DAVA ERRO) ---
    # Criamos um container fixo
    chat_placeholder = st.container()

    with chat_placeholder:
        # Pegamos as últimas 10 mensagens e INVERTEMOS para exibição
        # O segredo para não dar erro de Node é dar uma KEY ÚNICA para cada balão
        display_msg = st.session_state["chat_history"][-10:][::-1]
        
        for i, msg in enumerate(display_msg):
            # A key aqui é o segredo. Se o conteúdo mudar, a key muda e o React recria o nó
            # em vez de tentar remover o antigo.
            msg_key = f"msg_{len(st.session_state['chat_history'])}_{i}"
            with st.chat_message(msg["role"]):
                st.write(msg["content"])


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
