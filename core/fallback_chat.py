import re
import numpy as np
import pandas as pd
from core.insights import generate_auto_insights
from core.profiler import make_quality_metrics, basic_summary

TARGET_KEYWORDS = [
    "coluna alvo", "alvo", "target", "label", "y", "variável alvo", "variavel alvo",
    "prever", "predizer", "previsao", "previsão"
]

def _looks_like_target_question(q: str) -> bool:
    q = q.lower().strip()
    return any(k in q for k in TARGET_KEYWORDS)

def _target_candidates(df: pd.DataFrame) -> list[str]:
    """
    Heurística:
    - Evita colunas ID (id, uuid, code, codigo)
    - Evita colunas muito únicas (quase um identificador)
    - Prioriza colunas com menos cardinalidade (classificação) e também numéricas plausíveis (regressão)
    """
    n = len(df)
    cols = list(df.columns)

    def is_id_like(name: str) -> bool:
        name = name.lower()
        return any(tok in name for tok in ["id", "uuid", "cpf", "cnpj", "codigo", "code", "hash"])

    candidates = []
    for c in cols:
        if is_id_like(c):
            continue
        nunique = df[c].nunique(dropna=True)
        # muito único => parece ID
        if n > 0 and (nunique / n) > 0.95:
            continue
        candidates.append((c, nunique))

    if not candidates:
        return []

    # ordenar: primeiro baixa cardinalidade (bom p/ classificação),
    # depois numéricas com cardinalidade razoável (regressão)
    low_card = [c for c, u in candidates if u <= 20]
    mid_card = [c for c, u in candidates if 20 < u <= 200]
    high_card = [c for c, u in candidates if u > 200]

    ranked = low_card + mid_card + high_card
    return ranked[:6]

def offline_answer(question: str, df: pd.DataFrame) -> str:
    qm = make_quality_metrics(df)
    ins = generate_auto_insights(df, use_llm=False)
    summ = basic_summary(df).head(25)

    # ✅ Caso específico: pergunta de alvo/target
    if _looks_like_target_question(question):
        cands = _target_candidates(df)
        lines = []
        lines.append("🟡 **Modo offline (sem LLM disponível)**")
        lines.append("")
        lines.append(f"**Pergunta:** {question}")
        lines.append("")
        lines.append("✅ **Resposta direta:** não existe uma “coluna alvo” automática — você escolhe a coluna que quer prever (y).")
        if cands:
            lines.append("")
            lines.append("**Sugestões de colunas que podem ser alvo (candidatas):**")
            for c in cands:
                dtype = str(df[c].dtype)
                nunique = int(df[c].nunique(dropna=True))
                lines.append(f"- **{c}** (tipo: {dtype}, únicos: {nunique})")
            lines.append("")
            lines.append("Se você me disser o objetivo (classificação ou regressão) e qual resultado quer prever, eu te digo a melhor.")
        else:
            lines.append("")
            lines.append("Não encontrei candidatos claros (muitas colunas parecem ID/únicas).")
        return "\n".join(lines)

    # ✅ Resposta genérica offline (mantém)
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
    lines.append("💡 Para respostas em linguagem natural: use Ollama (local) ou OpenAI (com billing).")
    return "\n".join(lines)
