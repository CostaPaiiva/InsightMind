# core/offline_chat.py
from __future__ import annotations
import re
import pandas as pd

def _top_missing(df: pd.DataFrame, n: int = 8) -> pd.Series:
    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    return miss.head(n)

def _basic_shape(df: pd.DataFrame) -> tuple[int, int]:
    return df.shape[0], df.shape[1]

def _esc_col(col: str) -> str:
    return str(col)

def _dtype_report(df: pd.DataFrame, col: str) -> str:
    if col not in df.columns:
        return f"Não encontrei a coluna **{_esc_col(col)}** no dataset."
    s = df[col]
    dtype = str(s.dtype)
    if pd.api.types.is_integer_dtype(s):
        return f"A coluna **{_esc_col(col)}** está como **inteiro** ({dtype})."
    if pd.api.types.is_float_dtype(s):
        s_num = pd.to_numeric(s, errors="coerce").dropna()
        if len(s_num) > 0 and (s_num % 1 == 0).all():
            return (f"A coluna **{_esc_col(col)}** está como **float** ({dtype}), "
                    f"mas os valores parecem **inteiros**.")
        return f"A coluna **{_esc_col(col)}** está como **float** ({dtype})."
    if pd.api.types.is_bool_dtype(s):
        return f"A coluna **{_esc_col(col)}** está como **booleano** ({dtype})."
    if pd.api.types.is_datetime64_any_dtype(s):
        return f"A coluna **{_esc_col(col)}** está como **data/hora** ({dtype})."
    return f"A coluna **{_esc_col(col)}** não está numérica (dtype: **{dtype}**)."

def _parse_dtype_question(q: str) -> str | None:
    q = q.strip().lower()
    m = re.search(r"\b(tipo|dtype)\s+(da|do)\s+coluna\s+([a-zA-Z0-9_]+)\b", q)
    if m: return m.group(3)
    m = re.search(r"\bcoluna\s+([a-zA-Z0-9_]+)\s+(tem|é)\s+(inteiro|int|float|decimal)\b", q)
    if m: return m.group(1)
    if "coluna id" in q and ("inteiro" in q or "int" in q or "float" in q): return "id"
    return None

def offline_answer(
    question: str,
    df: pd.DataFrame,
    quality_metrics: dict | None = None,
    auto_insights: list | str | None = None,
    summary_table: pd.DataFrame | None = None,
) -> str:
    q_raw = (question or "").strip()
    q = q_raw.lower()

    # 1) Verifica se é pergunta de DTYPE (Tipo de coluna)
    col_asked = _parse_dtype_question(q)
    if col_asked:
        if col_asked not in df.columns:
            lower_map = {str(c).lower(): str(c) for c in df.columns}
            if col_asked.lower() in lower_map:
                col_asked = lower_map[col_asked.lower()]
        return _dtype_report(df, col_asked)

    # 2) Prepara os blocos de dados
    n_rows, n_cols = _basic_shape(df)
    overview = [f"📊 **Análise Rápida**: O dataset possui {n_rows} linhas e {n_cols} colunas."]
    
    # 3) Lógica de decisão por intenção (Melhorada)
    
    # INTENÇÃO: VALORES AUSENTES
    if any(word in q for word in ["missing", "ausent", "nulo", "vazio", "nan", "null"]):
        missing_top = _top_missing(df, n=8)
        lines = ["🔎 **Análise de Dados Ausentes**:"]
        if len(missing_top) == 0:
            lines.append("- Não foram encontrados valores nulos! ✅")
        else:
            for col, rate in missing_top.items():
                lines.append(f"- Coluna `{col}`: {rate*100:.1f}% de dados faltantes.")
        return "\n".join(lines)

    # INTENÇÃO: ESTATÍSTICA / RESUMO
    if any(word in q for word in ["resumo", "estatistica", "describe", "summary", "média", "media"]):
        lines = ["📈 **Resumo Estatístico das Numéricas**:"]
        if summary_table is not None:
            try:
                lines.append(summary_table.head(10).to_markdown())
            except:
                lines.append("```\n" + str(summary_table.head(10)) + "\n```")
        return "\n".join(lines)

    # INTENÇÃO: INSIGHTS
    if any(word in q for word in ["insight", "achado", "descobri", "interessante", "dica"]):
        lines = ["💡 **Insights Automáticos Detectados**:"]
        if isinstance(auto_insights, list) and auto_insights:
            for it in auto_insights[:6]:
                lines.append(f"- {it}")
        else:
            lines.append("- Analisando os dados, não detectei padrões anômalos óbvios ainda.")
        return "\n".join(lines)

    # RESPOSTA PADRÃO (Caso ele não entenda a pergunta específica)
    return (
        "🤔 Não entendi exatamente o que você quer saber sobre os dados.\n\n"
        "**Tente perguntar sobre:**\n"
        "- 'Quais colunas têm valores **nulos**?'\n"
        "- 'Me mostre o **resumo** estatístico.'\n"
        "- 'Qual o **tipo** da coluna [nome_da_coluna]?'\n"
        "- 'Quais são os **insights**?'"
    )