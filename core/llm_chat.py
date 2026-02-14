import json
import streamlit as st
import pandas as pd
from openai import OpenAI
from openai import RateLimitError, AuthenticationError, APIConnectionError, BadRequestError, APIStatusError
from core.offline_chat import offline_answer

def _openai_client() -> OpenAI:
    api_key = st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY ausente")
    return OpenAI(api_key=api_key)

def _build_context(df: pd.DataFrame, quality_metrics: dict, auto_insights: list[str], summary_table: pd.DataFrame) -> dict:
    summary_records = []
    if summary_table is not None and not summary_table.empty:
        summary_records = summary_table.head(30).to_dict(orient="records")
    return {
        "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "columns": list(df.columns)[:100],
        "quality_metrics": quality_metrics or {},
        "auto_insights": (auto_insights or [])[:20],
        "summary_table_sample": summary_records,
        "sample_rows": df.head(10).to_dict(orient="records"),
    }

def _system_prompt() -> str:
    return "Você é um analista sênior. Responda em PT-BR de forma estruturada. Use Markdown para tabelas."

def _answer_with_openai(question: str, context: dict) -> str:
    model = st.secrets.get("OPENAI_MODEL", "gpt-3.5-turbo") # ou seu modelo preferido
    client = _openai_client()
    context_json = json.dumps(context, ensure_ascii=False)

    # ✅ Corrigido para a API estável atual
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": f"Pergunta: {question}\nContexto: {context_json}"}
        ]
    )
    return resp.choices[0].message.content.strip()

def dataset_chat_answer(question: str, df: pd.DataFrame, quality_metrics: dict, auto_insights: list[str], summary_table: pd.DataFrame, provider: str = "auto") -> str:
    context = _build_context(df, quality_metrics, auto_insights, summary_table)
    
    if provider == "offline":
        return offline_answer(question, df, quality_metrics, auto_insights, summary_table)

    if provider in ("auto", "openai"):
        try:
            return _answer_with_openai(question, context)
        except Exception as e:
            if provider == "openai": return f"⚠️ Erro OpenAI: {e}"
    
    # Fallback para offline caso tudo falhe
    return offline_answer(question, df, quality_metrics, auto_insights, summary_table)