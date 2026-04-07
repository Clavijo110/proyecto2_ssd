# 🚀 Despliegue en Render

El código está en GitHub y listo para desplegar en Render.

## ⚡ Pasos Rápidos

### 1. Crear servicio en Render
- Ir a https://render.com
- Login con GitHub
- Click en **"New +"** > **"Web Service"**
- Conectar el repositorio `proyecto2_ssd`

### 2. Configuración en Render
```
Name:           salud-digital-ia
Environment:    Python
Build Command:  pip install -r requirements.txt
Start Command:  uvicorn app:app --host 0.0.0.0 --port $PORT
```

### 3. Deployment
- Render automáticamente hará push en cada cambio
- El servicio se despliega en: **https://salud-digital-ia.onrender.com**

---

## 📍 URLs del Servicio Desplegado

| Componente | URL |
|-----------|-----|
| Frontend | `/` |
| API Docs | `/api/docs` |
| API Health | `/api/health`|
| FHIR Root | `/fhir` |
| FHIR Docs | `/fhir/docs` |
| Compare Models | `/api/compare/p-0` |
| Metrics | `/api/metrics` |

---

## 🔧 Archivos Clave

- `app.py` — App principal unificada
- `requirements.txt` — Dependencias
- `Procfile` — Comando de arranque
- `render.yaml` — Configuración Render (opcional)
- `frontend/` — UI estática
- `ai-service/models/` — Modelos ML pre-entrenados
- `README_RENDER.md` — Instrucciones detalladas

---

## ✅ Estado del Despliegue

- ✅ Código en GitHub: https://github.com/Clavijo110/proyecto2_ssd
- ✅ Commit: `e3d7844` (Deploy to Render)
- ✅ Branch: `main`
- 🔄 Listo para desplegar en Render

---

## 💡 Características

- 🌳 6 algoritmos ML (Decision Tree, KNN, GBM, LR, XGBoost, TabPFN)
- 🏥 Servidor FHIR R4 integrado
- 🔵 FastAPI con docs automáticos
- 💻 Frontend con Leaflet.js
- 📊 Métricas y comparativa de modelos
- 🔐 Rate limiting y validación CORS

---

## 🎯 Próximo Paso

Abre https://render.com, conecta tu GitHub y sigue los pasos de arriba.
El primer despliegue tarda ~5 minutos. La app estará lista después.
