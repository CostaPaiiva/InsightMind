# core/offline_chat.py
from __future__ import annotations
import re
import pandas as pd

# --- 1. FUNÇÕES DE SUPORTE (Devem vir antes da offline_answer) ---

def _basic_shape(df: pd.DataFrame) -> tuple[int, int]:
    return df.shape[0], df.shape[1]

def _detect_target(df: pd.DataFrame) -> list[str]:
    """Tenta identificar a coluna alvo por nomes comuns."""
    common_targets = [
        "target", "label", "alvo", "class", "classe", "status", "churn", 
        "price", "preco", "venda", "valor", "outcome", "resultado"
    ]
    found = [col for col in df.columns if any(t in str(col).lower() for t in common_targets)]
    return found if found else [df.columns[-1]]

def _get_correlations(df: pd.DataFrame):
    df_num = df.select_dtypes(include=['number'])
    if df_num.shape[1] < 2: return {}
    corr = df_num.corr().unstack()
    high_corr = corr[(abs(corr) > 0.7) & (abs(corr) < 1.0)].drop_duplicates()
    return high_corr.head(5).to_dict()

# --- 2. FUNÇÃO PRINCIPAL ---

def offline_answer(
    question: str,
    df: pd.DataFrame,
    quality_metrics: dict | None = None,
    auto_insights: list | str | None = None,
    summary_table: pd.DataFrame | None = None,
) -> str:
    q = (question or "").strip().lower()

    # Se a pergunta for sobre duplicados
    if any(word in q for word in ["duplicado", "repetido", "duplicate"]):
        dups = df.duplicated().sum()
        if dups == 0:
            return "✅ **Limpeza**: Não foram encontradas linhas duplicadas."
        return f"⚠️ **Atenção**: Existem **{dups} linhas duplicadas**."

    # Se a pergunta for sobre resumo estatístico
    if any(word in q for word in ["resumo", "estatistica", "describe"]):
        desc = df.describe().T
        try:
            return "📊 **Resumo Estatístico**:\n\n" + desc.to_markdown()
        except:
            return "📊 **Resumo Estatístico**:\n\n```\n" + str(desc) + "\n```"

    # RESPOSTA PADRÃO (Onde estava dando o erro)
    n_rows, n_cols = _basic_shape(df)
    
    # Agora a função _detect_target já foi definida acima, então não dará erro
    targets = _detect_target(df)
    target_col = targets[0]
    
    return (
        f"📊 **Conjunto de dados com {n_rows} linhas e {n_cols} colunas.**\n\n"
        f"🎯 **Coluna Alvo provável**: `{target_col}`\n\n"
        "**Consulte sobre:**\n"
        "- 'Existem valores **duplicados**?'\n"
        "- 'Quais são os valores **nulos**?'\n"
        "- 'Me mostre o **resumo estatístico**.'"
    )