"""Gradient Boosting calibrado — Heart Disease UCI."""
import time

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score

from common import MODELS_DIR, append_metric, load_or_prepare


def main():
    X_train, X_test, y_train, y_test, _, _ = load_or_prepare()

    gbm = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=10,
        subsample=0.8,
        max_features="sqrt",
        random_state=42,
    )

    cv_folds = 5 if len(X_train) >= 120 else 3
    t0 = time.perf_counter()
    gbm_cal = CalibratedClassifierCV(
        estimator=gbm, method="isotonic", cv=cv_folds
    )
    gbm_cal.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    probs = gbm_cal.predict_proba(X_test)[:, 1]
    preds = gbm_cal.predict(X_test)
    infer_time = (time.perf_counter() - t1) / max(len(X_test), 1)

    f1 = float(f1_score(y_test, preds))
    auc = float(roc_auc_score(y_test, probs))
    brier = float(brier_score_loss(y_test, probs))

    joblib.dump(gbm_cal, MODELS_DIR / "gbm_heart.pkl")
    append_metric(
        {
            "model": "gbm",
            "f1": f1,
            "auc_roc": auc,
            "brier": brier,
            "train_time_s": train_time,
            "infer_time_per_sample_s": infer_time,
        }
    )
    print(f"gbm F1={f1:.4f} AUC={auc:.4f} Brier={brier:.4f} -> gbm_heart.pkl")


if __name__ == "__main__":
    main()
