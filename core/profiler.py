import pandas as pd
import numpy as np

# Ajustes de performance
_MAX_UNIQUE_SAMPLE_ROWS = 50000  # amostra p/ nunique em datasets grandes


def _first_non_null_example(s: pd.Series) -> str:
    """Pega o primeiro valor não-nulo sem fazer dropna completo (mais rápido)."""
    try:
        mask = s.notna().to_numpy()
        if mask.any():
            idx = int(np.argmax(mask))
            return str(s.iloc[idx])
    except Exception:
        pass
    return ""


def basic_summary(df: pd.DataFrame) -> pd.DataFrame:
    info = []
    n_rows = int(df.shape[0])

    # Para n_unique, em dataset muito grande, amostramos
    if n_rows > _MAX_UNIQUE_SAMPLE_ROWS:
        df_unique = df.sample(_MAX_UNIQUE_SAMPLE_ROWS, random_state=42)
    else:
        df_unique = df

    for c in df.columns:
        s = df[c]

        example = _first_non_null_example(s)

        # missing rápido
        miss_pct = float(s.isna().mean() * 100)

        # nunique pode ser caro; usa amostra se dataset for grande
        try:
            n_unique = int(df_unique[c].nunique(dropna=True))
        except Exception:
            n_unique = int(s.astype(str).nunique(dropna=True))

        info.append(
            {
                "coluna": c,
                "tipo": str(s.dtype),
                "% missing": miss_pct,
                "n_unique": n_unique,
                "exemplo": example,
            }
        )

    return pd.DataFrame(info)


def make_quality_metrics(df: pd.DataFrame) -> dict:
    n_rows, n_cols = int(df.shape[0]), int(df.shape[1])
    total_cells = n_rows * n_cols

    # missing total
    missing = int(df.isna().sum().sum()) if total_cells else 0

    # duplicadas
    dup_rows = int(df.duplicated().sum()) if n_rows else 0

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols]

    # colunas constantes: em dataset grande, amostra para acelerar
    if n_rows > _MAX_UNIQUE_SAMPLE_ROWS:
        df_const = df.sample(_MAX_UNIQUE_SAMPLE_ROWS, random_state=42)
    else:
        df_const = df

    constant_cols = []
    for c in df.columns:
        try:
            if df_const[c].nunique(dropna=True) <= 1:
                constant_cols.append(c)
        except Exception:
            # fallback seguro
            try:
                if df_const[c].astype(str).nunique(dropna=True) <= 1:
                    constant_cols.append(c)
            except Exception:
                pass

    return {
        "linhas": n_rows,
        "colunas": n_cols,
        "missing_total": missing,
        "missing_%": float(missing / total_cells * 100) if total_cells else 0.0,
        "linhas_duplicadas": dup_rows,
        "colunas_numericas": numeric_cols,
        "colunas_categoricas": cat_cols,
        "colunas_constantes": constant_cols,
    }
