import pandas as pd
from core.insights import generate_auto_insights
from core.suggestions import generate_suggestions
from core.profiler import make_quality_metrics

def fallback_answer(df: pd.DataFrame) -> str:
    qm = make_quality_metrics(df)
    ins = generate_auto_insights(df, use_llm=False)
    sug = generate_suggestions(df)

    text = []
    text.append(f"Visão geral: {qm['linhas']} linhas e {qm['colunas']} colunas.")
    text.append(f"Missing total: {qm['missing_total']} ({qm['missing_%']:.1f}%). Duplicadas: {qm['linhas_duplicadas']}.")
    text.append("\nPrincipais insights:")
    for i in ins[:8]:
        text.append(f"- {i}")
    text.append("\nSugestões de melhoria:")
    for s in sug[:8]:
        text.append(f"- {s}")
    return "\n".join(text)
