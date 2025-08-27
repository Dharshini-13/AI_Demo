# IMDB Sentiment Demo - Full (DVC + MLflow + FastAPI)

This demo contains:
- DVC pipeline (`dvc.yaml`) for `prepare` and `train` stages
- MLflow local experiment (if MLflow is available it will log a run)
- FastAPI app (`app.py`) served with Uvicorn for predictions
- A small sample dataset in `data/reviews.csv`
- `model.pkl` (trained) and metrics in `metrics/accuracy.json`

How to run locally (Windows PowerShell):
1. python -m venv .venv
2. .\.venv\Scripts\Activate.ps1   (or use cmd: .venv\Scripts\activate.bat)
3. pip install -r requirements.txt
4. python src/prepare_data.py
5. python src/train.py   # produces model.pkl and metrics/accuracy.json
6. uvicorn app:app --reload
7. Open http://127.0.0.1:8000/docs to test the API

Notes:
- MLflow logging will run if `mlflow` is installed; otherwise training still works.
- DVC is configured via `dvc.yaml` — after installing dvc you can use `dvc repro`, `dvc add`, `dvc push` etc.
