"""KNN con pipeline StandardScaler — Heart Disease UCI."""
import time

import joblib
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import MODELS_DIR, append_metric, load_or_prepare


def main():
    X_train, X_test, y_train, y_test, _, _ = load_or_prepare()

    knn_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=7,
                    weights="distance",
                    metric="euclidean",
                    algorithm="ball_tree",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    t0 = time.perf_counter()
    knn_pipeline.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    probs = knn_pipeline.predict_proba(X_test)[:, 1]
    preds = knn_pipeline.predict(X_test)
    infer_time = (time.perf_counter() - t1) / max(len(X_test), 1)

    f1 = float(f1_score(y_test, preds))
    auc = float(roc_auc_score(y_test, probs))

    joblib.dump(knn_pipeline, MODELS_DIR / "knn_heart.pkl")
    append_metric(
        {
            "model": "knn",
            "f1": f1,
            "auc_roc": auc,
            "train_time_s": train_time,
            "infer_time_per_sample_s": infer_time,
        }
    )
    print(f"knn F1={f1:.4f} AUC={auc:.4f} -> knn_heart.pkl")


if __name__ == "__main__":
    main()
