# Salud Digital IA — entrena modelos (si aplica) y levanta el stack completo.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "=== 1/2 Entrenando modelos (Heart Disease, perfil train) ===" -ForegroundColor Cyan
docker compose --profile train build train-models
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
docker compose --profile train run --rm train-models
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n=== 2/2 Levantando servicios (FHIR, MLflow, AI, Frontend) ===" -ForegroundColor Cyan
docker compose up --build -d
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`nListo. Abre:" -ForegroundColor Green
Write-Host "  Dashboard HTTP   http://localhost:3000"
Write-Host "  Dashboard HTTPS  https://localhost:3443  (certificado autofirmado)"
Write-Host "  FHIR HAPI        http://localhost:8080"
Write-Host "  MLflow           http://localhost:5000"
Write-Host "  API ML (proxy)   http://localhost:3000/ml-api/docs"
