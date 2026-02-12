import pandas as pd
from io import BytesIO

def load_csv_smart(uploaded_file):
    raw = uploaded_file.read()
    meta = {}

    for enc in ["utf-8", "latin1", "cp1252"]:
        try:
            raw.decode(enc)
            meta["encoding"] = enc
            break
        except UnicodeDecodeError:
            continue
    else:
        meta["encoding"] = "utf-8"

    seps = [",", ";", "\t", "|"]
    best = None
    best_cols = 0

    for sep in seps:
        try:
            df_try = pd.read_csv(BytesIO(raw), sep=sep, encoding=meta["encoding"])
            if df_try.shape[1] > best_cols:
                best_cols = df_try.shape[1]
                best = (sep, df_try)
        except Exception:
            pass

    if best is None:
        df = pd.read_csv(BytesIO(raw), encoding=meta["encoding"])
        meta["sep"] = "auto"
    else:
        meta["sep"] = best[0]
        df = best[1]

    df.columns = [c.strip() for c in df.columns]
    return df, meta
