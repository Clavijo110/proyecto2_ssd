"""XGBoost — Heart Disease UCI; guarda JSON nativo."""
import json
import time

import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from common import MODELS_DIR, append_metric, load_or_prepare


def main():
    X_train, X_test, y_train, y_test, feature_names, _ = load_or_prepare()
    fn = list(feature_names)

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=fn)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=fn)

    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "eta": 0.05,
        "max_depth": 5,
        "min_child_weight": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "lambda": 2.0,
        "alpha": 0.5,
        "scale_pos_weight": scale_pos_weight,
        "tree_method": "hist",
        "seed": 42,
    }

    t0 = time.perf_counter()
    evals_result = {}
    model_xgb = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, "train"), (dtest, "eval")],
        early_stopping_rounds=20,
        evals_result=evals_result,
        verbose_eval=False,
    )
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    probs = model_xgb.predict(dtest)
    preds = (probs > 0.5).astype(int)
    infer_time = (time.perf_counter() - t1) / max(len(X_test), 1)

    f1 = float(f1_score(y_test, preds))
    auc = float(roc_auc_score(y_test, probs))
    ap = float(average_precision_score(y_test, probs))

    out_json = MODELS_DIR / "xgb_heart.json"
    model_xgb.save_model(str(out_json))
    with open(MODELS_DIR / "xgb_feature_names.json", "w", encoding="utf-8") as f:
        json.dump(fn, f)

    append_metric(
        {
            "model": "xgboost",
            "f1": f1,
            "auc_roc": auc,
            "auc_pr": ap,
            "train_time_s": train_time,
            "infer_time_per_sample_s": infer_time,
        }
    )
    print(f"xgboost F1={f1:.4f} AUC={auc:.4f} AUC-PR={ap:.4f} -> xgb_heart.json")


if __name__ == "__main__":
    main()
