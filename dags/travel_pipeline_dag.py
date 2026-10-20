"""Validate flight data, then run the existing model comparison script."""

import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator

# The project root is one level above dags/. Override it if the DAG is copied elsewhere.
ROOT = Path(os.environ.get("TRAVEL_PROJECT_ROOT") or Path(__file__).resolve().parent.parent).resolve()
DATA_PATH = ROOT / "data" / "flights.csv"
TRAINING_SCRIPT = ROOT / "experiment_tracking" / "flight_price_experiments.py"
TRACKING_URI = "sqlite:///" + (ROOT / "experiment_tracking" / "mlflow.db").as_posix()


def validate_travel_dataset() -> str:
    """Validate the dataset and pass its absolute path to the training task."""
    data_file = Path(os.environ.get("FLIGHTS_DATA_PATH", str(DATA_PATH)))
    if not data_file.is_absolute():
        data_file = ROOT / data_file
    data_file = data_file.resolve()
    flights = pd.read_csv(data_file)

    # Match the columns and value rules used by load_flights in the training script.
    required = ["date", "from", "to", "distance", "price"]
    missing = set(required) - set(flights.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if flights[required].isna().any().any():
        raise ValueError("Required flight columns contain missing values.")
    flights["date"] = pd.to_datetime(flights["date"], format="%m/%d/%Y", errors="raise")
    if flights["date"].isna().any():
        raise ValueError("Flight dates must contain valid, non-missing values.")
    distance = pd.to_numeric(flights["distance"], errors="raise")
    price = pd.to_numeric(flights["price"], errors="raise")
    if not distance.between(0, float("inf"), inclusive="neither").all():
        raise ValueError("Distance must be positive and finite.")
    if not (price.ge(0) & price.lt(float("inf"))).all():
        raise ValueError("Price must be non-negative and finite.")
    if len(flights) < 10:
        raise ValueError("At least 10 flight records are required.")

    print(f"Validated {len(flights)} flight records: {data_file}")
    return str(data_file)


def retrain_and_track_model(data_file: str):
    """Run the four model experiments and save their results in MLflow."""
    python_path = os.environ.get("TRAINING_PYTHON", sys.executable)
    command = [
        python_path, str(TRAINING_SCRIPT),
        "--data", data_file,
        "--tracking-uri", os.environ.get("MLFLOW_TRACKING_URI") or TRACKING_URI,
        "--experiment-name", os.environ.get("MLFLOW_EXPERIMENT_NAME") or "flight-price-experiments",
    ]
    if os.environ.get("FLIGHTS_MAX_ROWS"):
        command.extend(["--max-rows", os.environ["FLIGHTS_MAX_ROWS"]])
    subprocess.run(command, cwd=ROOT, check=True)
    print("Model comparison complete. Metrics and Joblib pipelines are saved in MLflow.")


default_args = {
    "owner": "travel_mlops_team",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="travel_flight_price_mlops_workflow",
    default_args=default_args,
    description="Validate flight data, compare models, and track results with MLflow.",
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "travel", "regression"],
) as dag:
    task_validate = PythonOperator(
        task_id="validate_incoming_data",
        python_callable=validate_travel_dataset,
    )
    task_retrain = PythonOperator(
        task_id="retrain_and_track_model",
        python_callable=retrain_and_track_model,
        op_kwargs={"data_file": "{{ ti.xcom_pull(task_ids='validate_incoming_data') }}"},
    )

    task_validate.set_downstream(task_retrain)
