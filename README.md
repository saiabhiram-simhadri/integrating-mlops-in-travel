# Integrating MLOps in Travel

An early-stage project combining machine learning experiments on travel data with a Flask API for flight price prediction. This README describes the current repository and will evolve as the project develops.

## Current scope

- Flight price prediction and gender classification notebooks.
- Local flight, hotel, and user datasets.
- Saved flight price prediction and hotel recommender artifacts.
- A flight price API supporting single and batch predictions.
- A standalone MLflow experiment tracking demo for flight price models.
- Docker packaging and Kubernetes deployment, service, and autoscaling manifests for the flight price API.

The hotel recommender artifact is included, but there is currently no hotel recommendation API. Automated training orchestration, CI/CD, and model monitoring are not yet configured in this repository.

## Repository layout

```text
.
├── data/
│   ├── flights.csv
│   ├── hotels.csv
│   └── users.csv
├── notebooks/
│   ├── Flight_Price_Prediction.ipynb
│   └── Gender_Classification.ipynb
├── models/
│   ├── flight_price_pipeline.joblib
│   └── travel_hotel_recommender.joblib
├── flight_price_api/
│   ├── app.py
│   ├── models/
│   │   └── flight_price_pipeline.joblib
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── .dockerignore
│   └── k8s_deployment.yaml
├── .gitignore
├── pyproject.toml
├── flight_price_experiments.py
└── README.md
```

Datasets and model artifacts are intended to be committed to this repository. The API directory contains a separate copy of the flight model for its Docker build; keep it synchronized with the root model when replacing artifacts.

## Local setup

Use Python 3.11 or later. Python 3.11 matches the Docker image. From the repository root, create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e "."
```

For notebook tools and plotting libraries:

```powershell
python -m pip install -e ".[notebooks]"
jupyter lab
```

Notebook paths and any additional experiment-specific dependencies may need adjustment for your environment.

## Model experiment tracking with MLflow

Install the tracking extra and run the demo from the repository root:

```powershell
python -m pip install -e ".[tracking]"
python flight_price_experiments.py
```

For a faster demonstration, use a reproducible sample:

```powershell
python flight_price_experiments.py --max-rows 2000
```

The script follows the flight notebook's numerical baseline: `distance`, `flight_month`, and `is_weekend`, with an 80/20 random split and seed 47. Scaling is fitted inside each training pipeline. It compares Ridge, Random Forest, and two LightGBM learning rates. Categorical variables and target-derived features such as `price_per_km` are excluded.

Each invocation creates four separate model runs in the `flight-price-experiments` experiment. Each run logs model parameters, training/validation MAE, RMSE and R², split settings, features, source code, and the training/validation row indices. The script trains fresh pipelines so validation rows are excluded from fitting.

Serialization follows the notebook: `joblib.dump(pipeline, path)` saves the full fitted pipeline, and `joblib.load(path)` reloads it. The script checks that reloaded predictions match. Each run saves its pipeline and exact library requirements in `experiment_outputs/<run_id>/` and logs those files under the run's `api_model` artifacts. The API supports this raw pipeline format. A comparison CSV in `experiment_outputs/` identifies the lowest validation RMSE; these are model-selection scores, not independent final test scores.

Metadata is stored in root `mlflow.db` by default. Use `MLFLOW_TRACKING_URI` to select another backend. Outputs are eligible for Git tracking under the current `.gitignore`. Existing API models are preserved. Edit the settings near the top of the script to change paths, the experiment name, seed, or validation ratio; use `--max-rows` for a quick demonstration.

View the runs on port 5001 (leaving port 5000 available for the API):

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Open `http://localhost:5001`, select the experiment, and compare the model runs. Use `MLFLOW_TRACKING_URI` for another tracking backend. Additional options are shown by `python flight_price_experiments.py --help`.

## Run the flight price API

From the repository root:

```powershell
python flight_price_api/app.py
```

The development server listens on port `5000`. By default, the model is loaded from `models/flight_price_pipeline.joblib` relative to the working directory. Set `MODEL_PATH` before starting the service to use a different artifact:

```powershell
$env:MODEL_PATH = "C:\path\to\flight_price_pipeline.joblib"
python flight_price_api/app.py
```

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/` | GET | Service information |
| `/health` | GET | Returns HTTP 200 when the model is loaded, otherwise 503 |
| `/predict` | POST | Accepts one JSON object or a list of objects and returns predictions |

Check model availability:

```powershell
Invoke-RestMethod -Uri http://localhost:5000/health
```

Prediction records must match the saved pipeline's input features. The API accepts `route` directly or derives it by joining `from` and `to` with `_`. If omitted, `flight_month` defaults to `6`, `flight_dayofweek` to `2`, and `is_weekend` is derived from the day of week. Additional fields depend on the trained pipeline; consult the flight prediction notebook for its feature definitions.

A successful prediction response has this shape (values are illustrative):

```json
{
  "status": "success",
  "count": 1,
  "predictions": [123.45]
}
```

## Docker

Build from the API directory's context, which includes its model copy:

```powershell
docker build -t flight-price-api ./flight_price_api
docker run --rm -p 5000:5000 flight-price-api
```

The container runs the API with Gunicorn on port `5000`.

## Kubernetes

Review `flight_price_api/k8s_deployment.yaml` and set its image to an image available to your cluster before applying:

```powershell
kubectl apply -f flight_price_api/k8s_deployment.yaml
```

The manifest defines a deployment, a NodePort service on `30080`, and CPU-based autoscaling from two to five replicas. Autoscaling requires a working cluster metrics API. Actual service access depends on your cluster setup.

## Development status

`pyproject.toml` includes optional development dependencies and Ruff/pytest settings. Automated tests have not yet been added. Model accuracy, notebook reproducibility, and deployment behavior should be validated for the target environment before relying on predictions.


## Hotel recommendation dashboard

Install the dashboard dependencies and launch the app from the project root:

```powershell
python -m pip install -e ".[dashboard]"
python -m streamlit run app_streamlit.py
```

The app loads `data/hotels.csv` and `data/users.csv` using paths relative to the
script. **Similar hotels** compares hotel stay patterns using cosine similarity.
**For a traveler** loads `models/travel_hotel_recommender.joblib` and ranks
unseen destinations using similar travelers, then finds hotels within the budget.
Scores describe similarity or destination preference, not booking probabilities.

Run the dashboard tests with:

```powershell
python -m unittest discover -s tests -p test_app_streamlit.py -v
```
