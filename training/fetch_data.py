"""Descarga Heart Disease Cleveland (UCI id=45) y guarda data/heart_disease.csv."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
UCI_HEART_ID = 45


def main():
    from ucimlrepo import fetch_ucirepo

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data = fetch_ucirepo(id=UCI_HEART_ID)
    X = data.data.features.copy()
    tgt = data.data.targets
    if "num" in tgt.columns:
        num = tgt["num"]
    else:
        num = tgt.iloc[:, 0]
    out = X.copy()
    out["target"] = (pd.to_numeric(num, errors="coerce").fillna(0) > 0).astype(int)
    path = DATA_DIR / "heart_disease.csv"
    out.to_csv(path, index=False)
    print(f"Guardado: {path} ({len(out)} filas)")


if __name__ == "__main__":
    main()
