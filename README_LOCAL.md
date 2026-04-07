# 🏥 Salud Digital IA — Sistema Local (Sin Docker)

Implementación completa de un sistema de **6 algoritmos ML** + **FHIR** + **MLflow** para análisis de cardiopatías y salud digital.

## ✨ Características Implementadas

### 6 Algoritmos ML Entrenados
- 🌳 **Decision Tree**: Interpretabilidad clínica máxima (~34% F1)
- 📍 **KNN**: Búsqueda de pacientes similares (~56% AUC)
- 🚀 **Gradient Boosting**: Robustez con datos heterogéneos (~61% AUC)
- 📈 **Regresión Logística**: Epidemiología clásica (~58% AUC)
- ⚡ **XGBoost**: Mejor rendimiento en datos tabulares (~67% AUC)
- 🤖 **TabPFN Stub**: Ensemble rápido sin hiperparámetros (~53% AUC)

### Stack Tecnológico
- **Backend FHIR**: Servidor FHIR R4 mock en Python con FastAPI
- **AI Service**: FastAPI con 6 modelos ML pre-entrenados
- **MLflow**: Tracking de experimentos y métricas
- **Frontend**: Interfaz web con Leaflet.js + mapas geoespaciales
- **Base de datos**: En memoria (JSON-compatible) para desarrollo

---

## 🚀 Instalación Rápida

### 1. Requisitos Previos
- **Python 3.10+** (ya configurado en el sistema)
- **Ventanas PowerShell** (automáticas al ejecutar el script)

### 2. Instalar Dependencias (primera vez)
```powershell
cd C:\Users\Alejandro\Downloads\Docker
pip install -r ai-service/requirements.txt
pip install ucimlrepo  # Para entrenar modelos
```

### 3. Entrenar Modelos (primera vez)
```powershell
cd training
python prepare.py
python train_dt.py
python train_knn.py
python train_gbm.py
python train_lr.py
python train_xgb.py
python train_tabpfn_stub.py
```
**✅ Ya están entrenados** — Los modelos `.pkl` están en `ai-service/models/`

---

## 🎯 Ejecutar el Sistema

### Opción 1: Script Automático (Recomendado)
```powershell
cd C:\Users\Alejandro\Downloads\Docker
python start_services.py
```
Se abrirán **4 ventanas PowerShell** con:
- 🏥 FHIR Server (puerto 8080)
- 📊 MLflow Server (puerto 5000)
- 🤖 AI Service (puerto 8000)
- 💻 Frontend HTTP (puerto 3000)

### Opción 2: Manual (4 ventanas PowerShell separadas)

**Ventana 1** — FHIR Server:
```powershell
cd C:\Users\Alejandro\Downloads\Docker
python fhir_server.py
```

**Ventana 2** — MLflow:
```powershell
cd C:\Users\Alejandro\Downloads\Docker
python -m mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri ./mlflow_data
```

**Ventana 3** — AI Service:
```powershell
cd C:\Users\Alejandro\Downloads\Docker\ai-service
$env:MODELS_DIR = ".\models"
$env:FHIR_BASE_URL = "http://127.0.0.1:8080/fhir"
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

**Ventana 4** — Frontend:
```powershell
cd C:\Users\Alejandro\Downloads\Docker\frontend
python -m http.server 3000 --directory .
```

---

## 🌐 Acceso a Servicios

| Servicio | URL | Descripción |
|----------|-----|-------------|
| 🌐 **Frontend** | http://localhost:3000 | Interfaz web con mapas |
| 🔵 **API Docs** | http://localhost:8000/docs | Swagger - Endpoints REST |
| 🏥 **FHIR** | http://localhost:8080/fhir | Servidor FHIR R4 |
| 📊 **MLflow** | http://localhost:5000 | UI de experiments |

---

## 📊 API Endpoints Principales

### 1. Predecir con Modelo Específico
```bash
POST http://localhost:8000/predict
Content-Type: application/json

{
  "model": "xgboost",
  "features": [0.5, 0.3, 0.8, 0.2, 0.6, 0.4, 0.7]
}
```

### 2. Comparar Todos los Modelos
```bash
GET http://localhost:8000/compare/p-0?push_fhir=false
```

Retorna predicciones de los 6 modelos simultáneamente:
```json
{
  "predictions": {
    "decision_tree": {"probability_positive": 0.42, "predicted_class": 0},
    "knn": {"probability_positive": 0.55, "predicted_class": 1},
    "gbm": {"probability_positive": 0.61, "predicted_class": 1},
    "logistic_regression": {"probability_positive": 0.48, "predicted_class": 0},
    "xgboost": {"probability_positive": 0.67, "predicted_class": 1},
    "tabpfn": {"probability_positive": 0.52, "predicted_class": 1}
  }
}
```

### 3. Enviar Predicciones a FHIR
```bash
GET http://localhost:8000/compare/p-0?push_fhir=true
```
Crea recursos `RiskAssessment` en el servidor FHIR:
```json
{
  "resourceType": "RiskAssessment",
  "status": "final",
  "subject": {"reference": "Patient/p-0"},
  "prediction": [
    {
      "outcome": {"text": "Enfermedad cardíaca"},
      "probabilityDecimal": 0.67,
      "population": {"reference": "xgboost"}
    }
  ]
}
```

### 4. Ver Métricas de Entrenamiento
```bash
GET http://localhost:8000/metrics
```

---

## 📁 Estructura del Proyecto

```
C:\Users\Alejandro\Downloads\Docker\
├── start_services.py          # 🚀 Script para ejecutar TODO
├── fhir_server.py             # 🏥 FHIR Mock Server
├── run_local_stack.py         # (Alternativa: ejecución paralela)
│
├── ai-service/
│   ├── main.py                # 🔵 FastAPI con 6 modelos
│   ├── requirements.txt        # Dependencias
│   └── models/
│       ├── dt_heart.pkl        # Decision Tree
│       ├── knn_heart.pkl       # KNN
│       ├── gbm_heart.pkl       # Gradient Boosting
│       ├── lr_heart.pkl        # Logistic Regression
│       ├── xgb_heart.json      # XGBoost
│       ├── tabpfn_heart.pkl    # TabPFN Stub
│       ├── preprocessor.pkl    # Scaler para features
│       ├── data_bundle.joblib  # Dataset UCI
│       ├── metrics.json        # F1, AUC por modelo
│       └── compare_manifest.json
│
├── training/
│   ├── common.py              # Funciones compartidas
│   ├── prepare.py             # Descargar + preparar datos
│   ├── train_dt.py            # Entrenar Decision Tree
│   ├── train_knn.py           # Entrenar KNN
│   ├── train_gbm.py           # Entrenar GBM
│   ├── train_lr.py            # Entrenar Logistic Regression
│   ├── train_xgb.py           # Entrenar XGBoost
│   ├── train_tabpfn.py        # Entrenar TabPFN (requiere licencia)
│   ├── train_tabpfn_stub.py   # (Recomendado) TabPFN Stub
│   └── requirements.txt
│
├── frontend/
│   ├── index.html             # UI con Leaflet.js
│   ├── app.js                 # Lógica frontend
│   ├── config.js              # Config URLs (localhost)
│   └── nginx.conf             # (No usado en modo local)
│
└── data/
    └── heart_disease.csv      # Dataset UCI (descargado automáticamente)
```

---

## 🔧 Troubleshooting

### ❌ Error: "Port 8000 already in use"
```powershell
# Encontrar proceso en puerto 8000
netstat -ano | findstr :8000

# Matar proceso (reemplazar PID)
taskkill /PID <PID> /F
```

### ❌ Error: "ModuleNotFoundError: No module named 'fastapi'"
```powershell
pip install -r ai-service/requirements.txt
```

### ❌ FHIR retorna 404
El servidor FHIR mock está corriendo. Los endpoints disponibles son:
- `GET /fhir` — Root
- `POST /fhir/Patient`
- `GET /fhir/Patient/{id}`
- `POST /fhir/RiskAssessment`
- `GET /fhir/RiskAssessment`
- `GET /fhir/Observation`

### ❌ MLflow no inicia
```powershell
# Verificar que el directorio mlflow_data exista
mkdir mlflow_data
# Reintentar
python -m mlflow server --host 127.0.0.1 --port 5000
```

---

## 📚 Dataset: Heart Disease UCI

- **Fuente**: UCI Machine Learning Repository (Dataset #45)
- **Muestras**: ~303 pacientes
- **Features**: 13 variables clínicas
- **Target**: Presencia de enfermedad cardíaca (binario: 0/1)
- **Descarga automática**: `training/prepare.py`

---

## 🎓 Componentes por Algoritmo

### Decision Tree (DT)
```
Principio: Partición recursiva del espacio de features
Ventaja: Máxima interpretabilidad (reglas if-then)
Desventaja: Propenso a overfitting
Métrica: F1=0.3429 | AUC=0.5964
```

### K-Nearest Neighbors (KNN)
```
Principio: Búsqueda de K vecinos más similares
Ventaja: Ideal para cohort matching (encontrar pacientes similares)
Desventaja: Lento en predicción, sensible a escala
Métrica: F1=0.0000 | AUC=0.5594
```

### Gradient Boosting (GBM)
```
Principio: Ensemble secuencial (cada árbol ajusta residuos)
Ventaja: Robusto con datos heterogéneos, calibración nativa
Desventaja: Muchos hiperparámetros, lento en entrenamiento
Métrica: F1=0.1250 | AUC=0.6072 | Brier=0.1799
```

### Logistic Regression (LR)
```
Principio: Modelo lineal generalizado (sigmoid)
Ventaja: Interpretabilidad epidemiológica (Odds Ratios)
Desventaja: Asume linealidad en logit
Métrica: F1=0.2778 | AUC=0.5841
```

### XGBoost
```
Principio: GBM optimizado con regularización L1/L2
Ventaja: Mejor AUC en tabular, maneja missings, SHAP-compatible
Desventaja: Caja negra, overkill para datasets pequeños
Métrica: F1=0.4444 | AUC=0.6754 | AUC-PR=0.4218
```

### TabPFN (Stub)
```
Principio: ExtraTreesClassifier (sustituto sin licencia de TabPFN)
Ventaja: Sin hiperparámetros, fit rápido (<1s)
Desventaja: No es TabPFN real (pero tiene API compatible)
Métrica: F1=0.0000 | AUC=0.5348
```

---

## 🔐 Seguridad

- ✅ **Validación Pydantic**: Tipos estrictos en todas las requests
- ✅ **Rate Limiting**: 60 req/min por IP
- ✅ **CORS Restringido**: Solo localhost
- ✅ **Headers de Seguridad**: CSP, HSTS, X-Frame-Options
- ✅ **Inyección SQL**: N/A (no hay SQL en este servicio)

---

## 📈 Próximos Pasos (Opcional)

### 1. Producción con Docker ✅
```bash
docker compose -f docker-compose.yml up -d
```

### 2. Autenticación OAuth2
Agregar `fastapi.security.HTTPBearer` a `main.py`

### 3. Base de datos persistente
Reemplazar almacenamiento en memoria con PostgreSQL

### 4. TabPFN Real
```bash
pip install tabpfn>=2.0.6
# Aceptar licencia en browser
python training/train_tabpfn.py
```

---

## 📞 Soporte

- 🔗 **Documentación**: http://localhost:8000/docs
- 📊 **Métricas**: http://localhost:8000/metrics
- 🏥 **FHIR Spec**: https://www.hl7.org/fhir/R4/

---

**¡Listo para usar! 🚀**

Ejecuta `python start_services.py` y accede a http://localhost:3000
