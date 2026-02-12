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

st.set_page_config(page_title="InsightMind", layout="wide")
st.title("🧠 InsightMind — AutoDashboard + Chat IA + Limpeza + Relatório")

# ----------------------------
# Chat UI (estrutura estável)
# ----------------------------
def _init_state():
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

def _add(role: str, content: str):
    st.session_state.chat_messages.append({
        "id": str(uuid.uuid4()),
        "role": role,
        "content": content
    })

def render_chat():
    chat_area = st.container()
    with chat_area:
        for m in st.session_state.chat_messages:
            with st.chat_message(m["role"]):
                # Evita chaves dinâmicas/instáveis: sem placeholders dentro do loop
                st.markdown(m["content"])

def chat_ui(llm_respond_fn):
    _init_state()

    render_chat()

    prompt = st.chat_input("Digite sua mensagem…")
    if prompt:
        _add("user", prompt)

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                answer = llm_respond_fn(prompt, st.session_state.chat_messages)
                st.markdown(answer)

        _add("assistant", answer)
        st.rerun()

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

    # fallback seguro (offline)
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

# Histórico do chat do tab (mantido)
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
# --- Chat IA
with tabs[2]:
    st.markdown("### 💬 Pergunte ao seu dataset")
    st.caption("Ex.: “O que esse dataset diz?”, “Quais problemas de qualidade existem?”, “O que devo melhorar?”")

    provider_effective = llm_provider if use_llm else "offline"

    default_q = "O que esse dataset diz? Traga visão geral, achados importantes, problemas de qualidade e recomendações práticas."

    # 1) Render estável primeiro (sempre no mesmo lugar)
    st.markdown("### Histórico (últimas 10)")
    history_box = st.container()

    with history_box:
        last_10 = st.session_state["chat_history"][-10:]
        for item in last_10:
            # Conteúdo com keys estáveis (importante!)
            with st.chat_message("user"):
                st.markdown(item["q"], key=f"q_{item['id']}")
            with st.chat_message("assistant"):
                st.markdown(item["a"], key=f"a_{item['id']}")
            st.divider()

    st.caption(f"Provedor em uso: **{provider_effective}**")
    st.markdown("---")

    # 2) Input e botões depois (não misturar com render de resposta)
    user_q = st.text_input("Sua pergunta", value="", key="chat_input_question")

    col_btn1, col_btn2 = st.columns([1, 1])
    ask_custom = col_btn1.button("(perguntar)", key="btn_ask_custom")
    ask_default = col_btn2.button("✨ O que esse dataset diz?", key="btn_ask_default")

    # 3) Se perguntou: calcula, salva e rerun (NÃO renderiza resposta aqui)
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
                    provider=provider_effective,
                )

            # id estável por item
            st.session_state["chat_history"].append({
                "id": str(uuid.uuid4()),
                "q": q,
                "a": answer
            })

            # limpa o input (opcional, mas ajuda estabilidade visual)
            st.session_state["chat_input_question"] = ""

            # importante: re-render limpo pelo histórico
            st.rerun()


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
