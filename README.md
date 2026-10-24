# Integrating MLOps in Travel

A travel MLOps project combining flight price prediction, gender classification, and travel recommendations with MLflow experiment tracking, Airflow training orchestration, a Flask prediction API, and a Streamlit dashboard. Docker, Kubernetes, and Jenkins support containerization and deployment workflows.

## Features

- Flight price prediction, gender classification, and travel recommendation notebooks.
- Local flight, hotel, and user datasets.
- Saved flight price prediction and hotel recommender artifacts.
- A flight price API supporting single and batch predictions.
- A standalone MLflow experiment tracking demo for flight price models.
- A Streamlit dashboard for hotel and traveler recommendations.
- An Airflow DAG that validates flight data before running model comparisons.
- Unit tests for the dashboard and Airflow workflow.
- Docker packaging and Kubernetes deployment, service, and autoscaling manifests for the flight price API.
- A Jenkins pipeline definition for linting, import checks, container publishing, and Kubernetes deployment.

Hotel recommendations are available through the Streamlit dashboard. Airflow orchestrates data validation and model comparison, while Jenkins defines the CI/CD workflow.

## Repository layout

```text
.
├── data/
│   ├── flights.csv
│   ├── hotels.csv
│   └── users.csv
├── notebooks/
│   ├── Flight_Price_Prediction.ipynb
│   ├── Gender_Classification.ipynb
│   └── Travel_Recommendation.ipynb
├── models/
│   ├── flight_price_pipeline.joblib
│   ├── gender_classifier.joblib
│   └── travel_hotel_recommender.joblib
├── experiment_tracking/
│   ├── flight_price_experiments.py
│   ├── requirements_mflow.txt
│   ├── mlflow.db
│   └── mlruns/                       # MLflow run artifacts
├── dags/
│   ├── travel_pipeline_dag.py
│   └── requirements_dags.txt
├── flight_price_api/
│   ├── app.py
│   ├── models/
│   │   └── flight_price_pipeline.joblib
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── .dockerignore
│   └── k8s_deployment.yaml
├── tests/
│   ├── test_app_streamlit.py
│   └── test_travel_pipeline_dag.py
├── app_streamlit.py
├── Dockerfile.airflow
├── Dockerfile.airflow.dockerignore
├── Jenkinsfile
├── .gitignore
└── README.md
```

Datasets and model artifacts are intended to be committed to this repository. The API directory contains a separate copy of the flight model for its Docker build; keep it synchronized with the root model when replacing artifacts.

## Local setup

The API Docker image uses Python 3.11. From the repository root, create a virtual environment and install the API dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r flight_price_api/requirements.txt
```

For notebook tools and plotting libraries:

```powershell
python -m pip install jupyterlab matplotlib seaborn shap
jupyter lab
```

Install additional dependencies as needed for each notebook. Use separate virtual environments for the API, MLflow training, and Airflow workflows.

## Model experiment tracking with MLflow

Install the tracking requirements in a separate virtual environment and run the demo from the repository root:

```powershell
python -m pip install -r experiment_tracking/requirements_mflow.txt
python experiment_tracking/flight_price_experiments.py
```

For a faster demonstration, use a reproducible sample:

```powershell
python experiment_tracking/flight_price_experiments.py --max-rows 2000
```

The script follows the flight notebook's numerical baseline: `distance`, `flight_month`, and `is_weekend`, with an 80/20 random split and seed 47. Scaling is fitted inside each training pipeline. It compares Ridge, Random Forest, and two LightGBM learning rates. Categorical variables and target-derived features such as `price_per_km` are excluded.

Each invocation creates a parent comparison run and four nested model runs in the `flight-price-experiments` experiment. The parent records the seed, validation size, best model, and best run ID. Each model run logs estimator parameters and training/validation MAE, RMSE, and R². The script trains fresh pipelines so validation rows are excluded from fitting.

Each model run logs the fitted Joblib pipeline and its library versions under `api_model/`. The parent run stores `model_comparison.csv`, ordered by validation RMSE. Artifacts are stored under `experiment_tracking/mlruns/` for newly created experiments; existing experiments retain their configured artifact locations. The API supports the raw pipeline format. Training does not automatically replace either committed API model copy. Validation scores are for model selection, not independent final test scores.

Metadata is stored in `experiment_tracking/mlflow.db` by default, using an absolute path independent of the working directory. Use `--tracking-uri` or `MLFLOW_TRACKING_URI` to select another backend. The script prints the tracking URI and a matching UI command at startup.

View the runs on port 5001 (leaving port 5000 available for the API):

```powershell
mlflow ui --backend-store-uri sqlite:///experiment_tracking/mlflow.db --port 5001
```

Run this command from the repository root. Open `http://localhost:5001`, select `flight-price-experiments`, and expand the parent comparison runs to see the model runs. A bare `mlflow ui` can open a different database in the current directory and show no experiments. If a UI is already running, stop it with Ctrl+C and restart it with the correct `--backend-store-uri`. When training uses another backend, point the UI at that same backend. Additional options are shown by `python experiment_tracking/flight_price_experiments.py --help`.

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

## Airflow training workflow

`dags/travel_pipeline_dag.py` defines `travel_flight_price_mlops_workflow`, scheduled daily with catchup disabled and one active run at a time:

```text
validate_incoming_data -> retrain_and_track_model
```

Validation checks required columns, missing values, dates, positive finite distances, non-negative finite prices, and a minimum of 10 records. It passes the resolved dataset path to the training task through XCom. The training task launches `experiment_tracking/flight_price_experiments.py` as a subprocess and records its results in MLflow.

| Environment variable | Purpose |
| --- | --- |
| `TRAVEL_PROJECT_ROOT` | Override the project root resolved from the DAG location |
| `FLIGHTS_DATA_PATH` | Override the input CSV; relative paths resolve from the project root |
| `TRAINING_PYTHON` | Select the training interpreter; defaults to Airflow's interpreter |
| `MLFLOW_TRACKING_URI` | Override the default project SQLite backend |
| `MLFLOW_EXPERIMENT_NAME` | Override `flight-price-experiments` for DAG training |
| `FLIGHTS_MAX_ROWS` | Optionally sample records during training |

Run Airflow in Linux or a container, with `dags/requirements_dags.txt` installed and a separate training interpreter containing `experiment_tracking/requirements_mflow.txt`. Keep the project files accessible to both tasks.

`Dockerfile.airflow` describes an Airflow container with a separate training environment based on Python 3.12.

## Jenkins configuration

`Jenkinsfile` defines checkout, flake8 checks, an application import check, Docker build/push, and Kubernetes rollout.

Jenkins requires a Docker-capable agent, Python dependencies and flake8, Docker Hub credentials named `dockerhub-credentials`, and configured Kubernetes access.


## Hotel recommendation dashboard

Install the dashboard dependencies and launch the app from the project root:

```powershell
python -m pip install -r flight_price_api/requirements.txt streamlit
python -m streamlit run app_streamlit.py
```

The app loads `data/hotels.csv` and `data/users.csv` using paths relative to the
script. **Similar hotels** compares hotel stay patterns using cosine similarity.
**For a traveler** loads `models/travel_hotel_recommender.joblib` and ranks
unseen destinations using similar travelers, then finds hotels within the budget.
Scores describe similarity or destination preference, not booking probabilities.

## Tests

Run the dashboard tests after installing its dependencies:

```powershell
python -m unittest discover -s tests -p test_app_streamlit.py -v
```

Run the Airflow tests in an environment with Airflow and the training dependencies configured:

```powershell
python -m unittest discover -s tests -p test_travel_pipeline_dag.py -v
```

The Airflow tests check task ordering, invalid price rejection, and a complete `dag.test()` run that executes training. They skip when Airflow is unavailable. Set `FLIGHTS_MAX_ROWS=2000` for a smaller training sample. A complete DAG test writes MLflow runs and model artifacts.
