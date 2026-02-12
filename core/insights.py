import numpy as np
import pandas as pd

def generate_auto_insights(df: pd.DataFrame, use_llm: bool = False):
    insights = []
    n_rows, n_cols = df.shape
    insights.append(f"Dataset com {n_rows} linhas e {n_cols} colunas.")

    miss = df.isna().mean().sort_values(ascending=False)
    top_miss = miss[miss > 0].head(5)
    if len(top_miss) > 0:
        insights.append("Colunas com mais missing: " + ", ".join([f"{c} ({miss[c]*100:.1f}%)" for c in top_miss.index]))
    else:
        insights.append("Não há valores ausentes (missing) relevantes.")

    num = df.select_dtypes(include=[np.number])
    if num.shape[1] >= 2:
        corr = num.corr().abs()
        corr.values[np.tril_indices_from(corr)] = 0
        best = corr.stack().sort_values(ascending=False).head(3)
        for (a, b), v in best.items():
            insights.append(f"Correlação forte entre {a} e {b}: |r|={v:.2f}. Avalie colinearidade/causalidade.")

    if use_llm:
        insights.append("LLM ativado: plugue prompts adicionais para insights naturais.")
    return insights
