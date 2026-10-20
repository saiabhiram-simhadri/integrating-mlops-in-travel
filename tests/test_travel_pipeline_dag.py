"""Run inside the Airflow Docker image: python -m unittest discover -s tests -v."""

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

if importlib.util.find_spec('airflow') is None:
    raise unittest.SkipTest('Run these tests inside the Airflow Docker image.')

import pandas as pd
import pendulum

from dags.travel_pipeline_dag import dag, validate_travel_dataset


class TravelPipelineTests(unittest.TestCase):
    def test_task_order(self):
        self.assertEqual(set(dag.task_ids), {"validate_incoming_data", "retrain_and_track_model"})
        self.assertEqual(
            dag.get_task("validate_incoming_data").downstream_task_ids,
            {"retrain_and_track_model"},
        )

    def test_rejects_invalid_price(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "invalid_flights.csv"
            pd.DataFrame([{
                "from": "A", "to": "B", "flightType": "economic",
                "price": -10, "distance": 450, "agency": "Example", "date": "01/01/2026",
            }]).to_csv(path, index=False)
            with patch.dict(os.environ, {"FLIGHTS_DATA_PATH": str(path)}):
                with self.assertRaisesRegex(ValueError, "Price must be non-negative"):
                    validate_travel_dataset()

    def test_complete_dag(self):
        run = dag.test(execution_date=pendulum.datetime(2026, 10, 20, tz="UTC"))
        self.assertEqual(run.state, "success")
        self.assertTrue(all(task.state == "success" for task in run.get_task_instances()))


if __name__ == "__main__":
    unittest.main()


