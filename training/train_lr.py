"""Regresión logística L2 — Heart Disease UCI."""
import time

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import MODELS_DIR, append_metric, load_or_prepare


def main():
    X_train, X_test, y_train, y_test, _, _ = load_or_prepare()

    lr_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    C=1.0,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    t0 = time.perf_counter()
    lr_pipeline.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    probs = lr_pipeline.predict_proba(X_test)[:, 1]
    preds = lr_pipeline.predict(X_test)
    infer_time = (time.perf_counter() - t1) / max(len(X_test), 1)

    f1 = float(f1_score(y_test, preds))
    auc = float(roc_auc_score(y_test, probs))

    joblib.dump(lr_pipeline, MODELS_DIR / "lr_heart.pkl")
    append_metric(
        {
            "model": "logistic_regression",
            "f1": f1,
            "auc_roc": auc,
            "train_time_s": train_time,
            "infer_time_per_sample_s": infer_time,
        }
    )
    print(f"logistic_regression F1={f1:.4f} AUC={auc:.4f} -> lr_heart.pkl")


if __name__ == "__main__":
    main()
