"""
Microservicio FastAPI — 6 modelos ML (enfermedad cardíaca, UCI Heart id=45).

Seguridad:
- Sin SQL en este servicio (inferencia ML + HTTP a FHIR); no hay concatenación SQL.
- Rate limit: 60 peticiones / minuto / IP (ventana deslizante), respeta X-Forwarded-For detrás de Nginx.
- Validación estricta (Pydantic): modelo enumerado, features numéricas finitas, patient_id con patrón fijo.
- Cabeceras de seguridad (HSTS opcional si HTTPS delante).
- CORS restringido por ALLOWED_ORIGINS.
- Cliente FHIR httpx con verificación TLS configurable.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

import httpx
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Path as PathParam, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

_MODEL_DEFAULT = Path(__file__).resolve().parent / "models"
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(_MODEL_DEFAULT)))
_port = os.getenv("PORT")
_default_fhir = (
    f"http://127.0.0.1:{_port}/fhir" if _port else "http://fhir-server:8080/fhir"
)
FHIR_BASE_URL = os.getenv("FHIR_BASE_URL", _default_fhir).rstrip("/")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "")
FHIR_TLS_VERIFY = os.getenv("FHIR_TLS_VERIFY", "true").lower() in ("1", "true", "yes")
ROOT_PATH = os.getenv("ROOT_PATH", "").rstrip("/") or ""

_raw_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,https://localhost:3443,https://127.0.0.1:3443",
)
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

_trusted_hosts = os.getenv(
    "TRUSTED_HOSTS", "localhost,127.0.0.1,ai-service,salud-frontend"
).strip()
# TRUSTED_HOSTS=* desactiva TrustedHost (útil en Docker con varios Host headers)
TRUSTED_HOST_LIST = (
    [h.strip() for h in _trusted_hosts.split(",") if h.strip()]
    if _trusted_hosts and _trusted_hosts != "*"
    else []
)

RATE_LIMIT_WINDOW_S = 60
RATE_LIMIT_MAX = 60
_rate_buckets: Dict[str, List[float]] = defaultdict(list)

HSTS_MAX_AGE = int(os.getenv("HSTS_MAX_AGE", "31536000"))

app = FastAPI(
    title="Salud Digital IA — ML Service",
    description="Seis algoritmos ML (Heart Disease UCI) + comparativa y RiskAssessment FHIR R4",
    version="1.1.0",
    root_path=ROOT_PATH,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

if TRUSTED_HOST_LIST:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=TRUSTED_HOST_LIST)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept", "Authorization"],
)


def _client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()[:45]
    if request.client:
        return request.client.host
    return "unknown"


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    p = request.url.path.rstrip("/") or "/"
    if p.endswith("/health") or p == "/health":
        return await call_next(request)
    ip = _client_ip(request)
    now = time.time()
    bucket = _rate_buckets[ip]
    bucket[:] = [t for t in bucket if now - t < RATE_LIMIT_WINDOW_S]
    if len(bucket) >= RATE_LIMIT_MAX:
        return JSONResponse(
            status_code=429,
            content={
                "detail": "Límite de tasa excedido: máximo 60 solicitudes por minuto por IP.",
                "retry_after_s": RATE_LIMIT_WINDOW_S,
            },
            headers={"Retry-After": str(RATE_LIMIT_WINDOW_S)},
        )
    bucket.append(now)
    response = await call_next(request)
    return response


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = (
        "accelerometer=(), camera=(), geolocation=(), microphone=(), payment=()"
    )
    response.headers["Cache-Control"] = "no-store, max-age=0"
    csp = (
        "default-src 'none'; frame-ancestors 'none'; "
        "base-uri 'none'; form-action 'none'"
    )
    response.headers["Content-Security-Policy"] = csp
    if request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https":
        response.headers["Strict-Transport-Security"] = (
            f"max-age={HSTS_MAX_AGE}; includeSubDomains"
        )
    return response


class ModelName(str, Enum):
    decision_tree = "decision_tree"
    knn = "knn"
    gbm = "gbm"
    logistic_regression = "logistic_regression"
    xgboost = "xgboost"
    tabpfn = "tabpfn"


MODEL_KEYS = [m.value for m in ModelName]


class PredictRequest(BaseModel):
    model: ModelName
    features: List[float] = Field(..., min_length=1, max_length=4096)

    @field_validator("features")
    @classmethod
    def finite_floats(cls, v: List[float]) -> List[float]:
        for i, x in enumerate(v):
            if not isinstance(x, (int, float)) or not math.isfinite(float(x)):
                raise ValueError(f"features[{i}] debe ser un número finito")
        return v


_cache: Dict[str, Any] = {}


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _sanitize_fhir_token(value: str, max_len: int = 64) -> str:
    """Evita caracteres peligrosos en identificadores enviados a FHIR (no sustituye autenticación)."""
    s = re.sub(r"[^A-Za-z0-9._:-]", "-", value.strip())[:max_len]
    if not s:
        s = "unknown"
    return s


def _get_row_from_patient_id(patient_id: str) -> int:
    manifest = _load_json(MODELS_DIR / "compare_manifest.json")
    if not manifest or patient_id not in manifest:
        raise HTTPException(
            status_code=404,
            detail="patient_id no válido o no existe en el manifiesto (use p-0, p-1, …).",
        )
    return int(manifest[patient_id]["row"])


def _load_preprocessor():
    path = MODELS_DIR / "preprocessor.pkl"
    if not path.exists():
        return None
    return joblib.load(path)


def _load_bundle():
    key = "data_bundle"
    if key in _cache:
        return _cache[key]
    path = MODELS_DIR / "data_bundle.joblib"
    if not path.exists():
        return None
    _cache[key] = joblib.load(path)
    return _cache[key]


def _get_sklearn_model(filename: str):
    key = f"skl:{filename}"
    if key not in _cache:
        path = MODELS_DIR / filename
        if not path.exists():
            return None
        _cache[key] = joblib.load(path)
    return _cache[key]


def _get_xgb():
    key = "xgb"
    if key not in _cache:
        import xgboost as xgb

        p = MODELS_DIR / "xgb_heart.json"
        fn_path = MODELS_DIR / "xgb_feature_names.json"
        if not p.exists() or not fn_path.exists():
            return None
        booster = xgb.Booster()
        booster.load_model(str(p))
        fn = json.loads(fn_path.read_text(encoding="utf-8"))
        _cache[key] = (booster, fn)
    return _cache[key]


def _predict_vector(model_key: str, x: np.ndarray) -> tuple[float, int]:
    if model_key == "decision_tree":
        m = _get_sklearn_model("dt_heart.pkl")
        if m is None:
            raise HTTPException(503, "Modelo decision_tree no disponible")
        prob = float(m.predict_proba(x)[0, 1])
        cls = int(m.predict(x)[0])
        return prob, cls

    if model_key == "knn":
        m = _get_sklearn_model("knn_heart.pkl")
        if m is None:
            raise HTTPException(503, "Modelo knn no disponible")
        prob = float(m.predict_proba(x)[0, 1])
        cls = int(m.predict(x)[0])
        return prob, cls

    if model_key == "gbm":
        m = _get_sklearn_model("gbm_heart.pkl")
        if m is None:
            raise HTTPException(503, "Modelo gbm no disponible")
        prob = float(m.predict_proba(x)[0, 1])
        cls = int(m.predict(x)[0])
        return prob, cls

    if model_key == "logistic_regression":
        m = _get_sklearn_model("lr_heart.pkl")
        if m is None:
            raise HTTPException(503, "Modelo logistic_regression no disponible")
        prob = float(m.predict_proba(x)[0, 1])
        cls = int(m.predict(x)[0])
        return prob, cls

    if model_key == "xgboost":
        loaded = _get_xgb()
        if loaded is None:
            raise HTTPException(503, "Modelo xgboost no disponible")
        booster, fn = loaded
        import xgboost as xgb

        d = xgb.DMatrix(x, feature_names=fn)
        prob = float(booster.predict(d)[0])
        cls = int(prob > 0.5)
        return prob, cls

    if model_key == "tabpfn":
        m = _get_sklearn_model("tabpfn_heart.pkl")
        if m is None:
            raise HTTPException(503, "Modelo tabpfn no disponible")
        xf = x.astype(np.float32)
        prob = float(m.predict_proba(xf)[0, 1])
        cls = int(m.predict(xf)[0])
        return prob, cls

    raise HTTPException(400, "modelo no soportado")


def _maybe_mlflow_log(model_key: str, prob: float):
    if not MLFLOW_TRACKING_URI:
        return
    try:
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        with mlflow.start_run(run_name="inference", nested=True):
            mlflow.log_metric(f"prob_{model_key}", prob)
    except Exception:
        pass


def _build_risk_assessment(
    patient_id: str, model_name: str, probability: float
) -> dict:
    safe_id = _sanitize_fhir_token(patient_id.replace("/", "-"))
    return {
        "resourceType": "RiskAssessment",
        "status": "final",
        "identifier": [
            {
                "system": "http://salud-digital-ia.local/ml-model",
                "value": _sanitize_fhir_token(f"{model_name}-{safe_id}", 80),
            }
        ],
        "subject": {"reference": f"Patient/{safe_id}"},
        "code": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": "65842-8",
                    "display": "Predicted risk of disease progression",
                }
            ],
            "text": "Riesgo de enfermedad cardíaca (predicción ML)",
        },
        "prediction": [
            {
                "outcome": {
                    "text": f"Prob. enfermedad cardíaca — {model_name}",
                },
                "probabilityDecimal": round(float(probability), 4),
            }
        ],
    }


@app.get("/health")
def health():
    bundle = _load_bundle() is not None
    pre = _load_preprocessor() is not None
    files = {k: False for k in MODEL_KEYS}
    mapping = {
        "decision_tree": "dt_heart.pkl",
        "knn": "knn_heart.pkl",
        "gbm": "gbm_heart.pkl",
        "logistic_regression": "lr_heart.pkl",
        "xgboost": "xgb_heart.json",
        "tabpfn": "tabpfn_heart.pkl",
    }
    for k, f in mapping.items():
        if f.endswith(".json"):
            files[k] = (MODELS_DIR / f).exists() and (
                MODELS_DIR / "xgb_feature_names.json"
            ).exists()
        else:
            files[k] = (MODELS_DIR / f).exists()
    models_ready = all(files.values())
    return {
        "status": "ok" if bundle and pre and models_ready else "degraded",
        "dataset": "UCI Heart Disease (id=45)",
        "data_bundle": bundle,
        "preprocessor": pre,
        "models": files,
    }


@app.get("/models")
def list_models():
    return {
        "models": MODEL_KEYS,
        "feature_names": _load_json(MODELS_DIR / "feature_names.json"),
    }


@app.get("/metrics")
def metrics_table():
    path = MODELS_DIR / "metrics.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/predict")
def predict(body: PredictRequest):
    fnames = _load_json(MODELS_DIR / "feature_names.json")
    if not fnames or len(body.features) != len(fnames):
        raise HTTPException(
            400,
            detail=f"Dimensión incorrecta: se esperaban {len(fnames or [])} features.",
        )
    x = np.asarray(body.features, dtype=float).reshape(1, -1)
    model_key = body.model.value
    prob, cls = _predict_vector(model_key, x)
    _maybe_mlflow_log(model_key, prob)
    return {
        "model": model_key,
        "probability_positive": prob,
        "predicted_class": cls,
        "label": "enfermedad cardíaca" if cls == 1 else "sin signos de enfermedad (clase negativa)",
    }


@app.get("/compare/{patient_id}")
def compare(
    patient_id: str = PathParam(..., pattern=r"^p-\d{1,8}$"),
    push_fhir: bool = Query(False, description="Publicar RiskAssessment por modelo en HAPI"),
):
    row = _get_row_from_patient_id(patient_id)
    bundle = _load_bundle()
    if bundle is None:
        raise HTTPException(
            503,
            "Artefactos de entrenamiento no encontrados. Ejecute training/prepare.py y los train_*.py.",
        )
    X_test = bundle["X_test"]
    if row < 0 or row >= len(X_test):
        raise HTTPException(400, "Índice de fila fuera de rango")
    x = X_test[row : row + 1]

    results = {}
    t0 = time.perf_counter()
    for key in MODEL_KEYS:
        try:
            p, c = _predict_vector(key, x)
            results[key] = {
                "probability_positive": p,
                "predicted_class": c,
                "label": (
                    "enfermedad cardíaca"
                    if c == 1
                    else "sin signos de enfermedad (clase negativa)"
                ),
            }
            _maybe_mlflow_log(key, p)
        except HTTPException as e:
            results[key] = {"error": e.detail}

    out: Dict[str, Any] = {
        "patient_id": patient_id,
        "row": row,
        "predictions": results,
        "elapsed_s": round(time.perf_counter() - t0, 4),
    }

    if push_fhir:
        fhir_results = []
        with httpx.Client(timeout=30.0, verify=FHIR_TLS_VERIFY) as client:
            for key in MODEL_KEYS:
                if "error" in results.get(key, {}):
                    continue
                prob = results[key]["probability_positive"]
                resource = _build_risk_assessment(patient_id, key, prob)
                try:
                    r = client.post(
                        f"{FHIR_BASE_URL}/RiskAssessment",
                        json=resource,
                        headers={"Content-Type": "application/fhir+json"},
                    )
                    fhir_results.append(
                        {
                            "model": key,
                            "status_code": r.status_code,
                            "body": r.text[:500] if r.text else None,
                        }
                    )
                except Exception as ex:
                    fhir_results.append({"model": key, "error": str(ex)})
        out["fhir_push"] = fhir_results

    return out


@app.get("/manifest")
def manifest():
    data = _load_json(MODELS_DIR / "compare_manifest.json")
    if not data:
        raise HTTPException(404, "compare_manifest.json no generado")
    sample = list(data.keys())[:20]
    return {"total_ids": len(data), "sample_ids": sample}


@app.get("/")
def root():
    return {
        "service": "ai-service",
        "dataset": "UCI Heart Disease Cleveland (id=45)",
        "docs": f"{ROOT_PATH}/docs" if ROOT_PATH else "/docs",
        "rate_limit": f"{RATE_LIMIT_MAX} solicitudes / {RATE_LIMIT_WINDOW_S}s por IP",
        "endpoints": [
            "GET /health",
            "GET /models",
            "GET /metrics",
            "POST /predict",
            "GET /compare/{{patient_id}}?push_fhir=true",
            "GET /manifest",
        ],
    }
