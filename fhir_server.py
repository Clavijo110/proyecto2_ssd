"""
Servidor FHIR R4 mock local — compatible con salud-digital-ia
Endpoints básicos para almacenar y recuperar RiskAssessment, Observation, Patient
"""
import json
import uuid
from datetime import datetime
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="FHIR R4 Mock Server",
    description="Servidor FHIR local de prueba para Salud Digital IA",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# Almacenamiento en memoria
_storage: Dict[str, Dict[str, Any]] = {
    "Patient": {},
    "RiskAssessment": {},
    "Observation": {},
}


class ResourceType(BaseModel):
    resourceType: str


class Patient(BaseModel):
    resourceType: str = "Patient"
    id: str = None
    identifier: list = None
    name: list = None
    gender: str = None
    birthDate: str = None


class RiskAssessment(BaseModel):
    resourceType: str = "RiskAssessment"
    id: str = None
    status: str = "final"
    subject: Dict = None
    prediction: list = None
    period: Dict = None


class Observation(BaseModel):
    resourceType: str = "Observation"
    id: str = None
    status: str = "final"
    code: Dict = None
    subject: Dict = None
    valueQuantity: Dict = None
    effectiveDateTime: str = None


class Bundle(BaseModel):
    resourceType: str = "Bundle"
    type: str = "searchset"
    total: int = 0
    entry: list = None


@app.get("/", tags=["FHIR"])
async def fhir_root():
    """Raíz FHIR con información del servidor"""
    return {
        "resourceType": "CapabilityStatement",
        "status": "active",
        "date": datetime.utcnow().isoformat(),
        "software": {"name": "Salud Digital FHIR Mock", "version": "1.0.0"},
        "rest": [
            {
                "mode": "server",
                "resources": [
                    {"type": "Patient"},
                    {"type": "RiskAssessment"},
                    {"type": "Observation"},
                ],
            }
        ],
    }


@app.post("/Patient", tags=["Patient"])
async def create_patient(patient: Patient):
    """Crear un nuevo paciente"""
    if not patient.id:
        patient.id = str(uuid.uuid4())
    _storage["Patient"][patient.id] = patient.dict()
    return {**patient.dict(), "meta": {"versionId": "1"}}


@app.get("/Patient/{patient_id}", tags=["Patient"])
async def get_patient(patient_id: str):
    """Recuperar un paciente por ID"""
    if patient_id not in _storage["Patient"]:
        raise HTTPException(status_code=404, detail="Paciente no encontrado")
    return _storage["Patient"][patient_id]


@app.post("/RiskAssessment", tags=["RiskAssessment"])
async def create_risk_assessment(assessment: RiskAssessment):
    """Crear una evaluación de riesgo (predicción ML como FHIR)"""
    if not assessment.id:
        assessment.id = str(uuid.uuid4())
    assessment.period = {
        "start": datetime.utcnow().isoformat(),
    }
    _storage["RiskAssessment"][assessment.id] = assessment.dict()
    return {**assessment.dict(), "meta": {"versionId": "1"}}


@app.get("/RiskAssessment", tags=["RiskAssessment"])
async def search_risk_assessments(
    subject: str = Query(None, description="patient_id para filtrar"),
):
    """Buscar evaluaciones de riesgo"""
    assessments = list(_storage["RiskAssessment"].values())
    if subject:
        assessments = [
            a
            for a in assessments
            if a.get("subject", {}).get("reference", "").endswith(subject)
        ]
    return {
        "resourceType": "Bundle",
        "type": "searchset",
        "total": len(assessments),
        "entry": [{"resource": a} for a in assessments],
    }


@app.get("/RiskAssessment/{assessment_id}", tags=["RiskAssessment"])
async def get_risk_assessment(assessment_id: str):
    """Recuperar una evaluación de riesgo por ID"""
    if assessment_id not in _storage["RiskAssessment"]:
        raise HTTPException(
            status_code=404, detail="Evaluación de riesgo no encontrada"
        )
    return _storage["RiskAssessment"][assessment_id]


@app.post("/Observation", tags=["Observation"])
async def create_observation(observation: Observation):
    """Crear una observación clínica"""
    if not observation.id:
        observation.id = str(uuid.uuid4())
    if not observation.effectiveDateTime:
        observation.effectiveDateTime = datetime.utcnow().isoformat()
    _storage["Observation"][observation.id] = observation.dict()
    return {**observation.dict(), "meta": {"versionId": "1"}}


@app.get("/Observation", tags=["Observation"])
async def search_observations(subject: str = Query(None)):
    """Buscar observaciones"""
    observations = list(_storage["Observation"].values())
    if subject:
        observations = [
            o
            for o in observations
            if o.get("subject", {}).get("reference", "").endswith(subject)
        ]
    return {
        "resourceType": "Bundle",
        "type": "searchset",
        "total": len(observations),
        "entry": [{"resource": o} for o in observations],
    }


@app.get("/Observation/{obs_id}", tags=["Observation"])
async def get_observation(obs_id: str):
    """Recuperar una observación por ID"""
    if obs_id not in _storage["Observation"]:
        raise HTTPException(status_code=404, detail="Observación no encontrada")
    return _storage["Observation"][obs_id]


@app.post("/validate", tags=["Validation"])
async def validate_resource(resource: dict):
    """Validar un recurso FHIR (básico)"""
    if "resourceType" not in resource:
        raise HTTPException(status_code=400, detail="Falta resourceType")
    return {
        "resourceType": "OperationOutcome",
        "issue": [],
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check del servidor FHIR"""
    return {
        "status": "UP",
        "service": "FHIR R4 Mock",
        "timestamp": datetime.utcnow().isoformat(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="info")
