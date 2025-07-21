start cmd /k "uvicorn compute_service:app --host 0.0.0.0 --port 8001"
timeout /t 3
ssh -R 80:localhost:8001 serveo.net