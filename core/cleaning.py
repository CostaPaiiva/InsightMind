# Importa a biblioteca pandas para manipulação de dados em DataFrames
import pandas as pd
# Importa a biblioteca numpy para operações numéricas
import numpy as np

# Função que gera um plano de limpeza a partir de um DataFrame
def cleaning_plan_from_df(df: pd.DataFrame) -> dict:
    # Calcula a proporção de valores ausentes em cada coluna
    miss = df.isna().mean()
    # Retorna um dicionário com regras de limpeza
    return {
        "remove_duplicates": True,  # Remover linhas duplicadas
        "trim_strings": True,       # Remover espaços extras e padronizar strings
        "parse_dates": True,        # Tentar converter colunas em datas
        "drop_high_missing": bool((miss > 0.6).any()),  # Remover colunas com mais de 60% de valores ausentes
        "missing_threshold": 0.6,   # Limite de valores ausentes
        "drop_constant_cols": True, # Remover colunas constantes
        "outlier_clip": False,      # Não aplicar clipping de outliers por padrão
    }

# Função que tenta converter colunas de texto em datas
def _try_parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()  # Cria uma cópia do DataFrame
    obj_cols = out.select_dtypes(include=["object"]).columns  # Seleciona colunas de tipo objeto (strings)
    for c in obj_cols:  # Itera sobre cada coluna de texto
        s = out[c]
        parsed = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)  # Tenta converter em datas
        # Se pelo menos 70% dos valores forem válidos e houver diversidade suficiente
        if parsed.notna().mean() >= 0.7 and parsed.nunique(dropna=True) > 5:
            out[c] = parsed  # Substitui a coluna original pela versão convertida
    return out

# Função que padroniza strings
def _trim_strings(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    obj_cols = out.select_dtypes(include=["object"]).columns  # Seleciona colunas de texto
    for c in obj_cols:
        out[c] = out[c].astype(str).str.strip()  # Remove espaços extras
        out[c] = out[c].replace({"nan": np.nan, "None": np.nan})  # Converte "nan" e "None" em valores nulos
        out[c] = out[c].str.lower()  # Converte para minúsculas
    return out

# Função que remove colunas constantes
def _drop_constant_cols(df: pd.DataFrame):
    out = df.copy()
    dropped = []  # Lista de colunas removidas
    for c in list(out.columns):
        if out[c].nunique(dropna=True) <= 1:  # Se a coluna tiver apenas um valor único
            dropped.append(c)
            out = out.drop(columns=[c])  # Remove a coluna
    return out, dropped

# Função que realiza imputação de valores ausentes
def _impute(df: pd.DataFrame, impute_numeric: str, impute_categorical: str):
    out = df.copy()
    log = []  # Registro das operações realizadas

    num_cols = out.select_dtypes(include=[np.number]).columns  # Colunas numéricas
    cat_cols = [c for c in out.columns if c not in num_cols]   # Colunas categóricas

    # Imputação numérica (média ou mediana)
    if impute_numeric in ("median", "mean"):
        for c in num_cols:
            if out[c].isna().any():
                val = out[c].median() if impute_numeric == "median" else out[c].mean()
                out[c] = out[c].fillna(val)
        log.append(f"Imputação numérica aplicada: {impute_numeric}.")

    # Imputação categórica (moda)
    if impute_categorical == "mode":
        for c in cat_cols:
            if out[c].isna().any():
                mode = out[c].mode(dropna=True)
                if len(mode) > 0:
                    out[c] = out[c].fillna(mode.iloc[0])
        log.append("Imputação categórica aplicada: mode.")

    return out, log

# Função que aplica clipping de outliers usando IQR
def _clip_outliers_iqr(df: pd.DataFrame):
    out = df.copy()
    log = []
    num_cols = out.select_dtypes(include=[np.number]).columns  # Seleciona colunas numéricas
    for c in num_cols:
        s = out[c].dropna()
        if len(s) < 20:  # Ignora colunas com poucos valores
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)  # Quartis
        iqr = q3 - q1
        if iqr == 0:
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr  # Limites inferior e superior
        before = out[c].copy()
        out[c] = out[c].clip(lo, hi)  # Ajusta valores fora do intervalo
        changed = (before != out[c]).sum()
        if changed > 0:
            log.append(f"Outliers clipados em {c}: {changed} valores ajustados.")
    return out, log

# Função principal que aplica todas as etapas de limpeza
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

    # Remover duplicadas
    if remove_duplicates:
        d0 = out.shape[0]
        out = out.drop_duplicates()
        d1 = out.shape[0]
        if d1 != d0:
            log.append(f"Removidas duplicadas: {d0 - d1} linhas.")

    # Padronizar strings
    if trim_strings:
        out = _trim_strings(out)
        log.append("Strings padronizadas (strip/lower).")

    # Converter datas
    if parse_dates:
        out = _try_parse_dates(out)
        log.append("Tentativa de conversão de datas aplicada.")

    # Remover colunas com muitos valores ausentes
    if drop_high_missing:
        miss = out.isna().mean()
        to_drop = miss[miss >= missing_threshold].index.tolist()
        if to_drop:
            out = out.drop(columns=to_drop)
            log.append(f"Colunas removidas por missing >= {missing_threshold:.0%}: {', '.join(to_drop)}")

    # Remover colunas constantes
    if drop_constant_cols:
        out, dropped = _drop_constant_cols(out)
        if dropped:
            log.append(f"Colunas constantes removidas: {', '.join(dropped)}")

    # Imputação de valores ausentes
    out, impute_log = _impute(out, impute_numeric, impute_categorical)
    log.extend(impute_log)

    # Clipping de outliers
    if outlier_clip:
        out, out_log = _clip_outliers_iqr(out)
        log.extend(out_log)

    # Caso nenhuma alteração tenha sido feita
    if not log:
        log.append("Nenhuma alteração aplicada.")
    return out, log
