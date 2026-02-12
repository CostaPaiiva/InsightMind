import streamlit as st
import pandas as pd
from openai import OpenAI

def _client() -> OpenAI:
    api_key = st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY não encontrada em .streamlit/secrets.toml")
    return OpenAI(api_key=api_key)

def dataset_chat_answer(question: str, df: pd.DataFrame, quality_metrics: dict, auto_insights: list[str], summary_table: pd.DataFrame) -> str:
    model = st.secrets.get("OPENAI_MODEL", "gpt-4.1-mini")

    context = {
        "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "columns": list(df.columns)[:200],
        "quality_metrics": quality_metrics,
        "auto_insights": auto_insights[:25],
        "summary_table_sample": summary_table.to_dict(orient="records"),
        "sample_rows": df.head(15).to_dict(orient="records"),
    }

    system = (
        "Você é um analista de dados sênior. Responda em PT-BR, com objetividade. "
        "Baseie-se SOMENTE no contexto fornecido (não invente colunas/valores). "
        "Traga: (1) visão geral, (2) achados, (3) problemas de qualidade, (4) recomendações práticas. "
        "Se algo não estiver no contexto, diga que não dá para afirmar."
    )

    user = f"Pergunta do usuário: {question}\n\nContexto do dataset (JSON):\n{context}"
    client = _client()

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.output_text
