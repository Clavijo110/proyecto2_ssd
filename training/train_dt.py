"""Árbol de decisión — Heart Disease UCI (id=45)."""
import time

import joblib
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from common import MODELS_DIR, append_metric, load_or_prepare


def main():
    X_train, X_test, y_train, y_test, _, _ = load_or_prepare()

    t0 = time.perf_counter()
    dt = DecisionTreeClassifier(
        criterion="gini",
        max_depth=6,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
    )
    dt.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    probs = dt.predict_proba(X_test)[:, 1]
    preds = dt.predict(X_test)
    _ = dt.predict(X_test[:1])
    infer_time = (time.perf_counter() - t1) / max(len(X_test), 1)

    f1 = float(f1_score(y_test, preds))
    auc = float(roc_auc_score(y_test, probs))

    joblib.dump(dt, MODELS_DIR / "dt_heart.pkl")
    append_metric(
        {
            "model": "decision_tree",
            "f1": f1,
            "auc_roc": auc,
            "train_time_s": train_time,
            "infer_time_per_sample_s": infer_time,
        }
    )
    print(f"decision_tree F1={f1:.4f} AUC={auc:.4f} -> dt_heart.pkl")


if __name__ == "__main__":
    main()
