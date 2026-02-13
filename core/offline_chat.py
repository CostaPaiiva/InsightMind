# core/offline_chat.py
from __future__ import annotations
import re
import pandas as pd

def _top_missing(df: pd.DataFrame, n: int = 8):
    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    return miss.head(n)

def _basic_shape(df: pd.DataFrame):
    return df.shape[0], df.shape[1]

def offline_answer(
    question: str,
    df: pd.DataFrame,
    quality_metrics: dict,
    auto_insights: list | str,
    summary_table: pd.DataFrame,
) -> str:
    q = (question or "").strip().lower()

    n_rows, n_cols = _basic_shape(df)
    missing_top = _top_missing(df, n=8)

    # blocos base
    overview = [
        f"**Visão geral do dataset**",
        f"- Linhas: **{n_rows}**",
        f"- Colunas: **{n_cols}**",
    ]

    quality_block = ["", "**Qualidade (sinais principais)**"]
    # tenta ser robusto a diferentes formatos do seu qm
    if isinstance(quality_metrics, dict):
        for k in ["missing_rate", "duplicate_rows", "constant_cols", "high_missing_cols", "n_rows", "n_cols"]:
            if k in quality_metrics:
                quality_block.append(f"- {k}: **{quality_metrics[k]}**")

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
    ]

    # respostas “por intenção” (offline)
    if re.search(r"\b(missing|ausent|nulo|null|nan)\b", q):
        return "\n".join(overview + missing_block + recs)

    if re.search(r"\b(qualidade|quality|problema|erro|inconsist)\b", q):
        return "\n".join(overview + quality_block + missing_block + recs)

    if re.search(r"\b(resumo|summary|estat)\b", q):
        # usa o summary_table como texto compacto
        lines = ["**Resumo estatístico (amostra)**"]
        try:
            lines.append(summary_table.head(15).to_string())
        except Exception:
            lines.append(str(summary_table.head(15)))
        return "\n".join(overview + [""] + lines + recs)

    if re.search(r"\b(insight|achad|descobr)\b", q):
        return "\n".join(overview + insights_block + recs)

    # default: “o que esse dataset diz?”
    return "\n".join(overview + quality_block + missing_block + insights_block + recs)
