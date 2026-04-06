"""
Sustituto ligero de TabPFN cuando no hay torch (misma API predict_proba para el ai-service).
Solo para entorno demo/local; en producción use train_tabpfn.py con tabpfn instalado.
"""
import time

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score, roc_auc_score

from common import MODELS_DIR, append_metric, load_or_prepare


def main():
    X_train, X_test, y_train, y_test, _, _ = load_or_prepare()
    clf = ExtraTreesClassifier(
        n_estimators=120, max_depth=8, random_state=42, class_weight="balanced"
    )
    t0 = time.perf_counter()
    clf.fit(X_train.astype(np.float64), y_train)
    train_time = time.perf_counter() - t0
    t1 = time.perf_counter()
    probs = clf.predict_proba(X_test.astype(np.float64))[:, 1]
    preds = clf.predict(X_test.astype(np.float64))
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
            "note": "stub ExtraTrees (sin paquete tabpfn)",
        }
    )
    print(f"tabpfn_stub F1={f1:.4f} AUC={auc:.4f} -> tabpfn_heart.pkl")


if __name__ == "__main__":
    main()
