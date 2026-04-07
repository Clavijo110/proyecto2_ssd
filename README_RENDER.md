# Despliegue en Render — Salud Digital IA

Esta aplicación unifica frontend, backend ML y servidor FHIR en un solo servicio Python.

## Archivos de despliegue

- `app.py`: aplica FastAPI y monta:
  - `/api` → AI Service (6 modelos ML)
  - `/fhir` → servidor FHIR local
  - `/` → frontend estático
- `requirements.txt`: dependencias Python
- `Procfile`: comando de inicio para Render
- `render.yaml`: configuración de servicio Render

## Deploy en Render

1. Subir este repositorio a GitHub.
2. Crear un nuevo servicio en Render de tipo **Web Service**.
3. Seleccionar el repositorio.
4. Usar estos valores:
   - `Environment`: Python
   - `Build Command`: `pip install -r requirements.txt`
   - `Start Command`: `uvicorn app:app --host 0.0.0.0 --port $PORT`

- No es necesario forzar `FHIR_BASE_URL` en Render: el servicio usa el puerto `PORT` asignado automáticamente y resuelve `http://127.0.0.1:$PORT/fhir`.

## Rutas importantes después del despliegue

- Frontend: `/`
- API docs: `/api/docs`
- FHIR root: `/fhir`
- AI health: `/api/health`
- Metrics: `/api/metrics`
- Predict: `/api/predict`
- Compare: `/api/compare/{patient_id}`

## Notas

- La UI ahora usa `/api` para el backend y `/fhir` para el servidor FHIR.
- El sitio se sirve desde la misma aplicación, por lo que no hay dependencias Docker externas.
- Render gestionará el servidor y expondrá la aplicación públicamente.
