import streamlit as st
import pandas as pd
from openai import OpenAI
from openai import RateLimitError, AuthenticationError, APIConnectionError, BadRequestError, APIStatusError

def _client() -> OpenAI:
    api_key = st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY não encontrada em .streamlit/secrets.toml")
    return OpenAI(api_key=api_key)

def dataset_chat_answer(
    question: str,
    df: pd.DataFrame,
    quality_metrics: dict,
    auto_insights: list[str],
    summary_table: pd.DataFrame,
) -> str:
    """
    Retorna resposta do LLM. Se houver erro de cota/billing (429 insufficient_quota),
    retorna uma mensagem amigável e recomendações.
    """
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

    try:
        client = _client()
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.output_text

    except RateLimitError as e:
        # 429 pode ser quota estourada OU insufficient_quota (sem créditos)
        return (
            "⚠️ **Chat IA indisponível agora (OpenAI 429 / sem cota ou créditos).**\n\n"
            "O InsightMind vai continuar funcionando com **insights automáticos sem LLM**.\n\n"
            "**Como resolver:**\n"
            "1) Verifique se sua API Key é do projeto correto.\n"
            "2) Ative faturamento/créditos no painel da OpenAI (billing).\n"
            "3) Se você preferir custo zero, use um LLM local (ex.: Ollama) como fallback.\n"
        )

    except AuthenticationError:
        return (
            "❌ **Falha de autenticação (API Key inválida ou ausente).**\n"
            "Confira `OPENAI_API_KEY` em `.streamlit/secrets.toml`."
        )

    except (APIConnectionError, APIStatusError, BadRequestError) as e:
        return (
            "⚠️ **Erro ao chamar o serviço de IA.**\n\n"
            f"Detalhes: {type(e).__name__}\n"
            "Tente novamente mais tarde ou desative o Chat IA nas configurações."
        )
