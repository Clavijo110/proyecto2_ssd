"""
TabPFN — Heart Disease (muestras ~303): entrena con todo el train sin subsample agresivo.
"""
import time

import joblib
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from tabpfn import TabPFNClassifier

from common import MODELS_DIR, RANDOM_STATE, append_metric, load_or_prepare


def main():
    X_train, X_test, y_train, y_test, _, _ = load_or_prepare()

    max_train = 3000
    if len(X_train) > max_train:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(X_train), size=max_train, replace=False)
        X_tr = X_train[idx]
        y_tr = y_train[idx]
    else:
        X_tr, y_tr = X_train, y_train

    clf = TabPFNClassifier(device="cpu")

    t0 = time.perf_counter()
    clf.fit(X_tr.astype(np.float32), y_tr)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    probs = clf.predict_proba(X_test.astype(np.float32))[:, 1]
    preds = clf.predict(X_test.astype(np.float32))
    infer_time = (time.perf_counter() - t1) / max(len(X_test), 1)

    f1 = float(f1_score(y_test, preds))
    auc = float(roc_auc_score(y_test, probs))

    joblib.dump(clf, MODELS_DIR / "tabpfn_heart.pkl")
    append_metric(
        {
            "model": "tabpfn",
            "f1": f1,
            "auc_roc": auc,
            "train_time_s": train_time,
            "infer_time_per_sample_s": infer_time,
            "note": f"trained_on_n={len(X_tr)}",
        }
    )
    print(f"tabpfn F1={f1:.4f} AUC={auc:.4f} -> tabpfn_heart.pkl")


if __name__ == "__main__":
    main()
