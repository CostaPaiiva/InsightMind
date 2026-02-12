import pandas as pd
from core.insights import generate_auto_insights
from core.profiler import make_quality_metrics, basic_summary

def offline_answer(question: str, df: pd.DataFrame) -> str:
    qm = make_quality_metrics(df)
    ins = generate_auto_insights(df, use_llm=False)
    summ = basic_summary(df).head(25)

    lines = []
    lines.append("🟡 **Modo offline (sem LLM disponível)**")
    lines.append("")
    lines.append(f"**Pergunta:** {question}")
    lines.append("")
    lines.append(f"**Visão geral:** {qm['linhas']} linhas, {qm['colunas']} colunas.")
    lines.append(f"**Qualidade:** missing_total={qm['missing_total']} ({qm['missing_%']:.1f}%), duplicadas={qm['linhas_duplicadas']}.")
    lines.append("")
    lines.append("**Principais insights automáticos:**")
    for x in ins[:8]:
        lines.append(f"- {x}")
    lines.append("")
    lines.append("**Resumo (amostra das colunas):**")
    for _, row in summ.iterrows():
        lines.append(f"- {row['coluna']} | {row['tipo']} | missing {row['% missing']:.1f}% | únicos {row['n_unique']} | ex: {row['exemplo']}")
    lines.append("")
    lines.append("")
    return "\n".join(lines)
