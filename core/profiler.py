import pandas as pd
import numpy as np

def basic_summary(df: pd.DataFrame) -> pd.DataFrame:
    info = []
    for c in df.columns:
        s = df[c]
        example = ""
        s_non = s.dropna()
        if len(s_non) > 0:
            example = str(s_non.iloc[0])
        info.append({
            "coluna": c,
            "tipo": str(s.dtype),
            "% missing": float(s.isna().mean() * 100),
            "n_unique": int(s.nunique(dropna=True)),
            "exemplo": example
        })
    return pd.DataFrame(info)

def make_quality_metrics(df: pd.DataFrame) -> dict:
    total_cells = df.shape[0] * df.shape[1]
    missing = int(df.isna().sum().sum())
    dup_rows = int(df.duplicated().sum())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols]

    constant_cols = []
    for c in df.columns:
        if df[c].nunique(dropna=True) <= 1:
            constant_cols.append(c)

    return {
        "linhas": int(df.shape[0]),
        "colunas": int(df.shape[1]),
        "missing_total": missing,
        "missing_%": float(missing / total_cells * 100) if total_cells else 0.0,
        "linhas_duplicadas": dup_rows,
        "colunas_numericas": numeric_cols,
        "colunas_categoricas": cat_cols,
        "colunas_constantes": constant_cols
    }
