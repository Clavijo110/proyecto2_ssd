"""Genera data/heart_disease.csv sintético (offline) para demo cuando no hay red/UCI."""
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(ROOT / "data")))


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    n = 303
    age = rng.integers(29, 81, size=n)
    sex = rng.integers(0, 2, size=n)
    cp = rng.integers(0, 4, size=n)
    trestbps = rng.integers(94, 201, size=n)
    chol = rng.integers(126, 565, size=n)
    fbs = rng.integers(0, 2, size=n)
    restecg = rng.integers(0, 3, size=n)
    thalach = rng.integers(71, 203, size=n)
    exang = rng.integers(0, 2, size=n)
    oldpeak = np.clip(rng.exponential(1.0, size=n), 0, 6.2)
    slope = rng.integers(0, 3, size=n)
    ca = rng.integers(0, 4, size=n)
    thal = rng.choice(np.array([3, 6, 7]), size=n)
    z = (
        -2.0
        + 0.035 * (age - 50)
        + 0.75 * sex
        + 0.45 * cp
        - 0.012 * thalach
        + 0.55 * exang
        + 0.35 * oldpeak
    )
    p = 1.0 / (1.0 + np.exp(-z))
    target = (rng.binomial(1, p) | rng.binomial(1, 0.08, size=n)).astype(int)
    df = pd.DataFrame(
        {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,
            "target": target,
        }
    )
    path = DATA_DIR / "heart_disease.csv"
    df.to_csv(path, index=False)
    print(f"Demo CSV: {path} ({len(df)} filas, target balance {df['target'].mean():.2f})")


if __name__ == "__main__":
    main()
