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
    # mantém nome de coluna seguro p/ exibição
    return str(col)


def _dtype_report(df: pd.DataFrame, col: str) -> str:
    if col not in df.columns:
        return f"Não encontrei a coluna **{_esc_col(col)}** no dataset."

    s = df[col]
    dtype = str(s.dtype)

    # Inteiro (inclui pandas nullable Int64)
    if pd.api.types.is_integer_dtype(s):
        return f"A coluna **{_esc_col(col)}** está como **inteiro** ({dtype})."

    # Float (checa se é 'inteiro disfarçado' por causa de NaN)
    if pd.api.types.is_float_dtype(s):
        s_num = pd.to_numeric(s, errors="coerce").dropna()
        if len(s_num) > 0 and (s_num % 1 == 0).all():
            return (
                f"A coluna **{_esc_col(col)}** está como **float** ({dtype}), "
                f"mas os valores parecem **inteiros** (geralmente isso acontece por causa de valores ausentes)."
            )
        return f"A coluna **{_esc_col(col)}** está como **float** ({dtype})."

    # Boolean / datetime / object etc.
    if pd.api.types.is_bool_dtype(s):
        return f"A coluna **{_esc_col(col)}** está como **booleano** ({dtype})."
    if pd.api.types.is_datetime64_any_dtype(s):
        return f"A coluna **{_esc_col(col)}** está como **data/hora** ({dtype})."

    return (
        f"A coluna **{_esc_col(col)}** não está numérica (dtype: **{dtype}**). "
        f"Pode estar como texto, mistura de tipos ou precisando de conversão."
    )


def _parse_dtype_question(q: str) -> str | None:
    """
    Detecta perguntas do tipo:
    - "coluna id tem inteiro ou float?"
    - "a coluna idade é int ou float?"
    - "tipo da coluna customer_id"
    Retorna o nome da coluna, se detectar.
    """
    q = q.strip().lower()

    # "tipo da coluna X" / "qual o tipo da coluna X"
    m = re.search(r"\b(tipo|dtype)\s+(da|do)\s+coluna\s+([a-zA-Z0-9_]+)\b", q)
    if m:
        return m.group(3)

    # "coluna X é int/float?"
    m = re.search(r"\bcoluna\s+([a-zA-Z0-9_]+)\s+(tem|é)\s+(inteiro|int|float|decimal)\b", q)
    if m:
        return m.group(1)

    # caso específico: "coluna id tem inteiro ou float?"
    if "coluna id" in q and ("inteiro" in q or "int" in q or "float" in q):
        return "id"

    return None


def offline_answer(
    question: str,
    df: pd.DataFrame,
    quality_metrics: dict,
    auto_insights: list | str,
    summary_table: pd.DataFrame,
) -> str:
    q_raw = (question or "").strip()
    q = q_raw.lower()

    # 1) Perguntas sobre tipo/dtype de coluna (ex.: "coluna id é int ou float?")
    col_asked = _parse_dtype_question(q)
    if col_asked:
        # tenta casar por nome exato; se não achar, tenta case-insensitive
        if col_asked not in df.columns:
            lower_map = {str(c).lower(): str(c) for c in df.columns}
            if col_asked.lower() in lower_map:
                col_asked = lower_map[col_asked.lower()]
        return _dtype_report(df, col_asked)

    # 2) Blocos base
    n_rows, n_cols = _basic_shape(df)
    missing_top = _top_missing(df, n=8)

    overview = [
        "**Visão geral do dataset**",
        f"- Linhas: **{n_rows}**",
        f"- Colunas: **{n_cols}**",
    ]

    quality_block = ["", "**Qualidade (sinais principais)**"]
    if isinstance(quality_metrics, dict) and quality_metrics:
        for k in ["missing_rate", "duplicate_rows", "constant_cols", "high_missing_cols", "n_rows", "n_cols"]:
            if k in quality_metrics:
                quality_block.append(f"- {k}: **{quality_metrics[k]}**")
    else:
        quality_block.append("- (Métricas de qualidade indisponíveis no momento.)")

    missing_block = ["", "**Colunas com mais valores ausentes (top)**"]
    if len(missing_top) == 0:
        missing_block.append("- Nenhuma coluna com missing.")
    else:
        for col, rate in missing_top.items():
            missing_block.append(f"- {col}: **{rate*100:.1f}%**")

    insights_block = ["", "**Insights automáticos**"]
    if isinstance(auto_insights, list):
        if not auto_insights:
            insights_block.append("- (Sem insights automáticos relevantes.)")
        else:
            for it in auto_insights[:8]:
                insights_block.append(f"- {it}")
    else:
        insights_block.append(str(auto_insights))

    recs = [
        "",
        "**Recomendações práticas**",
        "- Trate missing nas colunas mais críticas (imputação/remoção dependendo do caso).",
        "- Remova duplicadas e colunas constantes, se existirem.",
        "- Padronize strings e valide colunas de data (parse e consistência).",
        "- Revise outliers em numéricas (IQR/clip) se estiverem distorcendo métricas.",
        "- Se alguma coluna deveria ser numérica/data e está como texto, converta e valide.",
    ]

    # 3) Respostas por intenção
    if re.search(r"\b(missing|ausent|nulo|null|nan)\b", q):
        return "\n".join(overview + missing_block + recs)

    if re.search(r"\b(qualidade|quality|problema|erro|inconsist)\b", q):
        return "\n".join(overview + quality_block + missing_block + recs)

    if re.search(r"\b(resumo|summary|estat)\b", q):
        lines = ["**Resumo estatístico (amostra)**"]
        try:
            lines.append(summary_table.head(15).to_string())
        except Exception:
            lines.append(str(summary_table.head(15)))
        return "\n".join(overview + [""] + lines + recs)

    if re.search(r"\b(insight|achad|descobr)\b", q):
        return "\n".join(overview + insights_block + recs)

    # 4) Default
    return "\n".join(overview + quality_block + missing_block + insights_block + recs)
