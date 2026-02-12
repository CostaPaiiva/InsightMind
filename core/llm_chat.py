import streamlit as st
import pandas as pd

from openai import OpenAI
from openai import RateLimitError, AuthenticationError, APIConnectionError, BadRequestError, APIStatusError

from core.fallback_chat import offline_answer

def _openai_client() -> OpenAI:
    api_key = st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY ausente")
    return OpenAI(api_key=api_key)

def _build_context(df: pd.DataFrame, quality_metrics: dict, auto_insights: list[str], summary_table: pd.DataFrame) -> dict:
    return {
        "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "columns": list(df.columns)[:200],
        "quality_metrics": quality_metrics,
        "auto_insights": auto_insights[:25],
        "summary_table_sample": summary_table.to_dict(orient="records"),
        "sample_rows": df.head(15).to_dict(orient="records"),
    }

def _prompt(system: str, question: str, context: dict) -> str:
    return f"{system}\n\nPergunta do usuário: {question}\n\nContexto do dataset (JSON):\n{context}"

def _answer_with_openai(question: str, context: dict) -> str:
    model = st.secrets.get("OPENAI_MODEL", "gpt-4.1-mini")

    system = (
        "Você é um analista de dados sênior. Responda em PT-BR, com objetividade. "
        "Baseie-se SOMENTE no contexto fornecido (não invente colunas/valores). "
        "Traga: (1) visão geral, (2) achados, (3) problemas de qualidade, (4) recomendações práticas. "
        "Se algo não estiver no contexto, diga que não dá para afirmar."
    )

    client = _openai_client()
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Pergunta: {question}\n\nContexto:\n{context}"},
        ],
    )
    return resp.output_text

def _answer_with_ollama(question: str, context: dict) -> str:
    # requer: pip install ollama + ollama instalado/rodando
    import ollama

    model = st.secrets.get("OLLAMA_MODEL", model = "llama3.2:1b")

    system = (
        "Você é um analista de dados sênior. Responda em PT-BR, com objetividade. "
        "Baseie-se SOMENTE no contexto fornecido (não invente colunas/valores). "
        "Traga: (1) visão geral, (2) achados, (3) problemas de qualidade, (4) recomendações práticas. "
        "Se algo não estiver no contexto, diga que não dá para afirmar."
    )

    prompt = f"{system}\n\nPergunta: {question}\n\nContexto do dataset (JSON):\n{context}"

    r = ollama.chat(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
    )
    return r["message"]["content"]

def dataset_chat_answer(
    question: str,
    df: pd.DataFrame,
    quality_metrics: dict,
    auto_insights: list[str],
    summary_table: pd.DataFrame,
    provider: str = "auto",  # "auto" | "openai" | "ollama" | "offline"
) -> str:
    """
    provider:
      - auto: tenta OpenAI; se falhar, tenta Ollama; se falhar, offline
      - openai: só OpenAI
      - ollama: só Ollama
      - offline: heurístico
    """
    context = _build_context(df, quality_metrics, auto_insights, summary_table)

    if provider == "offline":
        return offline_answer(question, df)

    if provider in ("auto", "openai"):
        try:
            return _answer_with_openai(question, context)
        except RateLimitError:
            if provider == "openai":
                return "⚠️ OpenAI: sem cota/créditos (429). Troque o provedor para Ollama ou Offline."
        except AuthenticationError:
            if provider == "openai":
                return "❌ OpenAI: API Key inválida/ausente. Configure OPENAI_API_KEY ou use Ollama/Offline."
        except (APIConnectionError, APIStatusError, BadRequestError, Exception):
            if provider == "openai":
                return "⚠️ OpenAI: erro na chamada. Troque para Ollama/Offline."

    if provider in ("auto", "ollama"):
        try:
            return _answer_with_ollama(question, context)
        except Exception:
            if provider == "ollama":
                return "⚠️ Ollama indisponível. Verifique se o Ollama está instalado/rodando e se o modelo foi baixado."

    return offline_answer(question, df)
