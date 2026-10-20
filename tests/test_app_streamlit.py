"""Run with: python -m unittest discover -s tests -p test_app_streamlit.py -v."""

import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from app_streamlit import recommend_for_traveler, recommend_similar_hotels

APP = Path(__file__).resolve().parents[1] / "app_streamlit.py"


class RecommendationTests(unittest.TestCase):
    def setUp(self):
        self.stats = pd.DataFrame({
            "name": ["Hotel A", "Hotel B", "Hotel C"],
            "location": ["A", "B", "C"], "avg_daily_rate": [100, 100, 200],
        })
        self.model = {
            "train_matrix": pd.DataFrame([[2, 0, 0], [1, 4, 0], [2, 0, 6]],
                                         index=[0, 1, 2], columns=["A", "B", "C"]),
            "user_similarity_df": pd.DataFrame([[1, .8, .2], [.8, 1, 0], [.2, 0, 1]],
                                               index=[0, 1, 2], columns=[0, 1, 2]),
            "k_neighbors": 2,
        }

    def test_weighted_destinations_exclude_history(self):
        result = recommend_for_traveler(0, self.model, self.stats, (0, 300), 8)
        self.assertEqual(result["name"].tolist(), ["Hotel B", "Hotel C"])
        self.assertAlmostEqual(result.iloc[0]["score"], 3.2)
        self.assertAlmostEqual(result.iloc[1]["score"], 1.2)

    def test_budget_is_applied_before_top_k(self):
        result = recommend_for_traveler(0, self.model, self.stats, (150, 250), 1)
        self.assertEqual(result["name"].tolist(), ["Hotel C"])
        similarity = pd.DataFrame([[1, .9, .5], [.9, 1, .3], [.5, .3, 1]],
                                  index=self.stats["name"], columns=self.stats["name"])
        result = recommend_similar_hotels("Hotel A", similarity, self.stats, (150, 250), 1)
        self.assertEqual(result["name"].tolist(), ["Hotel C"])

    def test_no_neighbors_or_no_unseen_destinations(self):
        self.model["user_similarity_df"].loc[0] = [1, 0, 0]
        self.assertTrue(recommend_for_traveler(0, self.model, self.stats, (0, 300), 4).empty)
        self.model["user_similarity_df"].loc[0] = [1, .8, .2]
        self.model["train_matrix"].loc[0] = [1, 1, 1]
        self.assertTrue(recommend_for_traveler(0, self.model, self.stats, (0, 300), 4).empty)


class DashboardTests(unittest.TestCase):
    def test_loads_from_another_working_directory(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as folder:
            try:
                os.chdir(folder)
                app = AppTest.from_file(str(APP), default_timeout=30).run()
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                self.assertGreater(len(app.dataframe[0].value), 0)
            finally:
                os.chdir(previous)
        selected = app.sidebar.selectbox[0].value
        self.assertNotIn(selected, app.dataframe[0].value["Hotel"].tolist())
        app.sidebar.slider[0].set_value(1).run()
        self.assertEqual(len(app.dataframe[0].value), 1)
        low = app.sidebar.slider[1].min
        app.sidebar.slider[1].set_value((low + .01, low + .02)).run()
        self.assertFalse(app.exception)
        self.assertGreater(len(app.info), 0)

    def test_saved_model_mode_and_empty_history_result(self):
        app = AppTest.from_file(str(APP), default_timeout=30).run()
        app.sidebar.radio[0].set_value("For a traveler").run()
        app.sidebar.selectbox[0].select(1).run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertGreater(len(app.dataframe[0].value), 0)
        app.sidebar.selectbox[0].select(0).run()
        self.assertFalse(app.exception)
        self.assertGreater(len(app.info), 0)


if __name__ == "__main__":
    unittest.main()

