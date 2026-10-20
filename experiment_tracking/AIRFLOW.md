# Test the travel Airflow DAG

Airflow runs in Linux through Docker Desktop. The Windows virtual environment is
used for local ML development; Airflow and training dependencies are installed in
the Docker image. A separate training virtual environment keeps MLflow's library
versions separate from Airflow's dependencies.

From the project root, build the image:

```powershell
docker build --load -f Dockerfile.airflow -t travel-airflow:local .
```

Initialize a temporary Airflow database and run the tests:

```powershell
docker run --rm -e _AIRFLOW_DB_MIGRATE=true -e FLIGHTS_MAX_ROWS=2000 travel-airflow:local python -m unittest discover -s tests -v
```

This checks task order, rejects an invalid fare, and executes both tasks through
Airflow. Validation reads the full dataset; training uses a reproducible 2,000-row
sample. Omit `-e FLIGHTS_MAX_ROWS=2000` to train on the full dataset.

The test container is removed afterward. Its Airflow database, MLflow runs, and
saved models are temporary and do not replace your local experiment artifacts.

The DAG runs daily and uses these optional environment variables:

- `FLIGHTS_DATA_PATH`: dataset path inside the container (default: `/opt/project/data/flights.csv`).
- `TRAINING_PYTHON`: interpreter for model training (configured by the image).
- `FLIGHTS_MAX_ROWS`: optional training sample size.
- `MLFLOW_TRACKING_URI`: optional MLflow backend override.

Build the image again after changing the DAG or experiment script.

If the DAG is copied into Airflow's own DAGs folder, set `TRAVEL_PROJECT_ROOT` to
the project directory as seen by the Airflow worker. Relative `FLIGHTS_DATA_PATH`
values are resolved from that project root. The validation task passes its absolute
dataset path to training through Airflow XCom. `MLFLOW_EXPERIMENT_NAME` optionally
changes the experiment name; the default database remains
`experiment_tracking/mlflow.db`. The workflow ends after comparison and artifact
tracking; deployment uses your existing API setup separately.
