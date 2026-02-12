import pandas as pd
import numpy as np

def cleaning_plan_from_df(df: pd.DataFrame) -> dict:
    miss = df.isna().mean()
    return {
        "remove_duplicates": True,
        "trim_strings": True,
        "parse_dates": True,
        "drop_high_missing": bool((miss > 0.6).any()),
        "missing_threshold": 0.6,
        "drop_constant_cols": True,
        "outlier_clip": False,
    }

def _try_parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    obj_cols = out.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        s = out[c]
        parsed = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
        if parsed.notna().mean() >= 0.7 and parsed.nunique(dropna=True) > 5:
            out[c] = parsed
    return out

def _trim_strings(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    obj_cols = out.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        out[c] = out[c].astype(str).str.strip()
        out[c] = out[c].replace({"nan": np.nan, "None": np.nan})
        out[c] = out[c].str.lower()
    return out

def _drop_constant_cols(df: pd.DataFrame):
    out = df.copy()
    dropped = []
    for c in list(out.columns):
        if out[c].nunique(dropna=True) <= 1:
            dropped.append(c)
            out = out.drop(columns=[c])
    return out, dropped

def _impute(df: pd.DataFrame, impute_numeric: str, impute_categorical: str):
    out = df.copy()
    log = []

    num_cols = out.select_dtypes(include=[np.number]).columns
    cat_cols = [c for c in out.columns if c not in num_cols]

    if impute_numeric in ("median", "mean"):
        for c in num_cols:
            if out[c].isna().any():
                val = out[c].median() if impute_numeric == "median" else out[c].mean()
                out[c] = out[c].fillna(val)
        log.append(f"Imputação numérica aplicada: {impute_numeric}.")

    if impute_categorical == "mode":
        for c in cat_cols:
            if out[c].isna().any():
                mode = out[c].mode(dropna=True)
                if len(mode) > 0:
                    out[c] = out[c].fillna(mode.iloc[0])
        log.append("Imputação categórica aplicada: mode.")

    return out, log

def _clip_outliers_iqr(df: pd.DataFrame):
    out = df.copy()
    log = []
    num_cols = out.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        s = out[c].dropna()
        if len(s) < 20:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        before = out[c].copy()
        out[c] = out[c].clip(lo, hi)
        changed = (before != out[c]).sum()
        if changed > 0:
            log.append(f"Outliers clipados em {c}: {changed} valores ajustados.")
    return out, log

def clean_dataset(
    df: pd.DataFrame,
    remove_duplicates: bool,
    trim_strings: bool,
    parse_dates: bool,
    drop_high_missing: bool,
    missing_threshold: float,
    impute_numeric: str,
    impute_categorical: str,
    drop_constant_cols: bool,
    outlier_clip: bool,
):
    out = df.copy()
    log = []

    if remove_duplicates:
        d0 = out.shape[0]
        out = out.drop_duplicates()
        d1 = out.shape[0]
        if d1 != d0:
            log.append(f"Removidas duplicadas: {d0 - d1} linhas.")

    if trim_strings:
        out = _trim_strings(out)
        log.append("Strings padronizadas (strip/lower).")

    if parse_dates:
        out = _try_parse_dates(out)
        log.append("Tentativa de conversão de datas aplicada.")

    if drop_high_missing:
        miss = out.isna().mean()
        to_drop = miss[miss >= missing_threshold].index.tolist()
        if to_drop:
            out = out.drop(columns=to_drop)
            log.append(f"Colunas removidas por missing >= {missing_threshold:.0%}: {', '.join(to_drop)}")

    if drop_constant_cols:
        out, dropped = _drop_constant_cols(out)
        if dropped:
            log.append(f"Colunas constantes removidas: {', '.join(dropped)}")

    out, impute_log = _impute(out, impute_numeric, impute_categorical)
    log.extend(impute_log)

    if outlier_clip:
        out, out_log = _clip_outliers_iqr(out)
        log.extend(out_log)

    if not log:
        log.append("Nenhuma alteração aplicada.")
    return out, log
