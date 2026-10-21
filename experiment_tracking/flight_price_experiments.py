"""Train flight price models and compare their validation scores with MLflow."""

import argparse
import os
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
FEATURES = ["distance", "flight_month", "is_weekend"]
DEFAULT_TRACKING_URI = "sqlite:///" + (SCRIPT_DIR / "mlflow.db").as_posix()


def save_pipeline(pipeline):
    """Save the full pipeline with Joblib and log it with its library versions."""
    with TemporaryDirectory() as folder:
        model_path = Path(folder) / "flight_price_pipeline.joblib"
        joblib.dump(pipeline, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="api_model")

    packages = ["scikit-learn", "lightgbm", "pandas", "numpy", "scipy", "joblib"]
    requirements = "\n".join(f"{package}=={version(package)}" for package in packages)
    mlflow.log_text(requirements + "\n", "api_model/requirements.txt")


def load_flights(path, max_rows, seed):
    """Check the data and prepare calendar and route features."""
    flights = pd.read_csv(path)
    required = ["date", "from", "to", "distance", "price"]
    missing = set(required) - set(flights.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if flights[required].isna().any().any():
        raise ValueError("Required flight columns contain missing values.")

    flights["date"] = pd.to_datetime(flights["date"], format="%m/%d/%Y")
    flights[["distance", "price"]] = flights[["distance", "price"]].astype(float)
    if not np.isfinite(flights[["distance", "price"]]).all().all():
        raise ValueError("Distance and price must contain finite numbers.")
    if (flights["distance"] <= 0).any() or (flights["price"] < 0).any():
        raise ValueError("Distance must be positive and price must be non-negative.")

    # Calendar features and the direction of travel.
    flights["flight_month"] = flights["date"].dt.month
    flights["flight_dayofweek"] = flights["date"].dt.dayofweek
    flights["is_weekend"] = flights["flight_dayofweek"].isin([5, 6]).astype(int)
    flights["route"] = flights["from"].str.strip() + "_" + flights["to"].str.strip()

    if max_rows is not None:
        if max_rows < 10:
            raise ValueError("--max-rows must be at least 10.")
        flights = flights.sample(n=min(max_rows, len(flights)), random_state=seed)
    if len(flights) < 10:
        raise ValueError("At least 10 flight records are required.")

    # Pipelines select FEATURES; extra engineered columns are ready for later experiments.
    # Keep price-derived columns and identifiers out of the model inputs.
    inputs = FEATURES + ["flight_dayofweek", "route"]
    return flights[inputs], flights["price"]


def model_candidates(seed):
    """Compare three model types and two LightGBM learning rates."""
    return {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=100, max_depth=12, random_state=seed, n_jobs=2
        ),
        "lightgbm_baseline": LGBMRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=6,
            num_leaves=31, random_state=seed, verbosity=-1, n_jobs=2,
        ),
        "lightgbm_lower_learning_rate": LGBMRegressor(
            n_estimators=100, learning_rate=0.05, max_depth=6,
            num_leaves=31, random_state=seed, verbosity=-1, n_jobs=2,
        ),
    }


def regression_metrics(actual, predicted, prefix):
    """Calculate the three regression scores used for comparison."""
    return {
        f"{prefix}_mae": float(mean_absolute_error(actual, predicted)),
        f"{prefix}_rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        f"{prefix}_r2": float(r2_score(actual, predicted)),
    }


def run_experiments(args):
    """Train each model, log its results, and display the comparison."""
    # 1. Set up tracking and split the data once for all models.
    mlflow.set_tracking_uri(args.tracking_uri)
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    if mlflow.get_tracking_uri().startswith("sqlite:"):
        print(f'mlflow ui --backend-store-uri "{mlflow.get_tracking_uri()}" --port 5050')
    if mlflow.get_experiment_by_name(args.experiment_name) is None:
        mlflow.create_experiment(
            args.experiment_name,
            artifact_location=(SCRIPT_DIR / "mlruns" / args.experiment_name).as_uri(),
        )
    mlflow.set_experiment(args.experiment_name)
    X, y = load_flights(args.data, args.max_rows, args.seed)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=args.validation_size, random_state=args.seed
    )

    results = []
    with mlflow.start_run(run_name="flight_price_model_comparison"):
        mlflow.log_params({"seed": args.seed, "validation_size": args.validation_size})
        for name, model in model_candidates(args.seed).items():
            # 2. Fit preprocessing on training rows only, inside a fresh pipeline.
            preprocessor = ColumnTransformer([("num", StandardScaler(), FEATURES)])
            pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", model)])

            with mlflow.start_run(run_name=name, nested=True) as run:
                mlflow.log_params(model.get_params())
                pipeline.fit(X_train, y_train)

                # 3. Log training and validation scores, then save the fitted pipeline.
                metrics = regression_metrics(y_train, pipeline.predict(X_train), "train")
                metrics.update(regression_metrics(y_valid, pipeline.predict(X_valid), "validation"))
                mlflow.log_metrics(metrics)
                save_pipeline(pipeline)
                results.append({"model": name, "run_id": run.info.run_id, **metrics})
                print(f"{name}: validation RMSE = {metrics['validation_rmse']:.2f}")

        # 4. Compare validation results and record the best run.
        comparison = pd.DataFrame(results).sort_values("validation_rmse")
        best = comparison.iloc[0]
        mlflow.log_params({"best_model": best["model"], "best_run_id": best["run_id"]})
        mlflow.log_text(comparison.to_csv(index=False), "model_comparison.csv")
        print(comparison[["model", "validation_mae", "validation_rmse", "validation_r2"]]
              .to_string(index=False))
        print("Validation scores are for model selection, not final test performance.")
    return comparison


def parse_args():
    """Read optional settings from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "flights.csv")
    parser.add_argument("--experiment-name", default="flight-price-experiments")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI")
                        or DEFAULT_TRACKING_URI)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--max-rows", type=int, help="Optional sample size for a quick demo")
    return parser.parse_args()


if __name__ == "__main__":
    run_experiments(parse_args())
