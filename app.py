import json
import os
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent

# Load the AI service app from ai-service/main.py
spec_ai = spec_from_file_location("ai_service_main", str(BASE_DIR / "ai-service" / "main.py"))
ai_module = module_from_spec(spec_ai)
spec_ai.loader.exec_module(ai_module)

# Load the FHIR mock server from fhir_server.py
spec_fhir = spec_from_file_location("fhir_server", str(BASE_DIR / "fhir_server.py"))
fhir_module = module_from_spec(spec_fhir)
spec_fhir.loader.exec_module(fhir_module)

app = FastAPI(
    title="Salud Digital IA Cloud",
    description="Aplicación unificada de Salud Digital IA con frontend, ML y FHIR",
    version="1.0.0",
)

# Montar API y FHIR en el mismo servicio
app.mount("/api", ai_module.app)
app.mount("/fhir", fhir_module.app)

@app.get("/fhir")
def fhir_root_redirect():
    return RedirectResponse(url="/fhir/")

# Servir frontend estático desde / o /ui
app.mount(
    "/",
    StaticFiles(directory=str(BASE_DIR / "frontend"), html=True),
    name="frontend",
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "Salud Digital IA Cloud",
        "api": "/api/health",
        "fhir": "/fhir",
    }
