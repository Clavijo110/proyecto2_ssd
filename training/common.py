"""
Datos y preprocesamiento — UCI Heart Disease Cleveland (id=45).
Clase positiva: enfermedad cardíaca (num > 0 en el dataset original).
No se usa SQL: solo pandas/sklearn sobre CSV o ucimlrepo.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(ROOT / "data")))
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(ROOT / "ai-service" / "models")))
METRICS_PATH = MODELS_DIR / "metrics.json"
MANIFEST_PATH = MODELS_DIR / "compare_manifest.json"

RANDOM_STATE = 42
UCI_HEART_ID = 45


def _binary_target_from_num(series: pd.Series) -> pd.Series:
    v = pd.to_numeric(series, errors="coerce").fillna(0)
    return (v > 0).astype(int)


def _load_raw_xy():
    csv_path = DATA_DIR / "heart_disease.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if "target" in df.columns:
            y = df["target"].astype(int)
            X = df.drop(columns=["target"])
        elif "num" in df.columns:
            y = _binary_target_from_num(df["num"])
            X = df.drop(columns=["num"])
        else:
            raise ValueError("heart_disease.csv requiere columna 'target' (0/1) o 'num' (0–4)")
    else:
        from ucimlrepo import fetch_ucirepo

        data = fetch_ucirepo(id=UCI_HEART_ID)
        X = data.data.features.copy()
        tgt = data.data.targets
        if "num" in tgt.columns:
            y = _binary_target_from_num(tgt["num"])
        else:
            y = _binary_target_from_num(tgt.iloc[:, 0])

    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def prepare_train_test(test_size: float = 0.2):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    X_raw, y = _load_raw_xy()

    try:
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_raw,
            y,
            test_size=test_size,
            stratify=y,
            random_state=RANDOM_STATE,
        )
    except ValueError:
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_raw,
            y,
            test_size=test_size,
            stratify=None,
            random_state=RANDOM_STATE,
        )

    pre = build_preprocessor(X_train_raw)
    pre.fit(X_train_raw)
    X_train = pre.transform(X_train_raw)
    X_test = pre.transform(X_test_raw)

    feature_names = [f"f{i}" for i in range(X_train.shape[1])]
    joblib.dump(pre, MODELS_DIR / "preprocessor.pkl")
    with open(MODELS_DIR / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f)

    n_test = X_test.shape[0]
    manifest = {f"p-{i}": {"row": i} for i in range(n_test)}
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    joblib.dump(
        {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": np.asarray(y_train),
            "y_test": np.asarray(y_test),
            "feature_names": feature_names,
        },
        MODELS_DIR / "data_bundle.joblib",
    )

    return (
        X_train,
        X_test,
        np.asarray(y_train),
        np.asarray(y_test),
        feature_names,
        pre,
    )


def load_or_prepare():
    bundle_path = MODELS_DIR / "data_bundle.joblib"
    pre_path = MODELS_DIR / "preprocessor.pkl"
    if bundle_path.exists() and pre_path.exists():
        b = joblib.load(bundle_path)
        pre = joblib.load(pre_path)
        return (
            b["X_train"],
            b["X_test"],
            b["y_train"],
            b["y_test"],
            b["feature_names"],
            pre,
        )
    return prepare_train_test()


def append_metric(record: dict):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    if METRICS_PATH.exists():
        rows = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    rows = [r for r in rows if r.get("model") != record.get("model")]
    rows.append(record)
    METRICS_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    prepare_train_test()
    print("OK: preprocessor.pkl, feature_names.json, compare_manifest.json, data_bundle.joblib")
