"""Explore similar hotels or recommendations from the saved traveler model."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "models" / "travel_hotel_recommender.joblib"


@st.cache_data
def load_data():
    """Load project data and calculate hotel similarity from stay durations."""
    hotels = pd.read_csv(DATA_DIR / "hotels.csv")
    users = pd.read_csv(DATA_DIR / "users.csv")
    interactions = hotels.pivot_table(
        index="userCode", columns="name", values="days", aggfunc="sum", fill_value=0
    )
    similarity = pd.DataFrame(
        cosine_similarity(interactions.T), index=interactions.columns, columns=interactions.columns
    )
    stats = hotels.groupby(["name", "place"]).agg(
        avg_daily_rate=("price", "mean"),
        total_bookings=("days", "count"),
        avg_stay_days=("days", "mean"),
    ).reset_index().rename(columns={"place": "location"})
    return hotels, users, similarity, stats


@st.cache_resource
def load_recommender():
    """Load the user-similarity and destination matrices saved with Joblib."""
    return joblib.load(MODEL_PATH)


def recommend_similar_hotels(selected_hotel, similarity, stats, budget, top_k):
    scores = similarity[selected_hotel].drop(selected_hotel).rename("score")
    recommendations = stats.merge(scores, left_on="name", right_index=True)
    recommendations = recommendations[
        recommendations["avg_daily_rate"].between(*budget) & recommendations["score"].gt(0)
    ]
    return recommendations.sort_values("score", ascending=False).head(top_k)


def recommend_for_traveler(user_code, model, stats, budget, top_k):
    """Rank unseen destinations using stays from the nearest similar travelers."""
    matrix = model["train_matrix"]
    neighbors = model["user_similarity_df"].loc[user_code].drop(user_code)
    neighbors = neighbors[neighbors > 0].nlargest(model["k_neighbors"])
    if neighbors.empty:
        return pd.DataFrame()
    scores = matrix.loc[neighbors.index].mul(neighbors, axis=0).sum() / neighbors.sum()
    scores = scores[matrix.loc[user_code] == 0].rename("score")
    recommendations = stats.merge(scores, left_on="location", right_index=True)
    recommendations = recommendations[
        recommendations["avg_daily_rate"].between(*budget) & recommendations["score"].gt(0)
    ]
    return recommendations.sort_values("score", ascending=False).head(top_k)


def main():
    st.set_page_config(page_title="Travel & Hotel Recommendations", page_icon="🏨", layout="wide")
    st.title("Travel & Hotel Recommendations")
    st.caption("Find similar hotels or discover destinations using similar travelers' stays.")
    try:
        hotels, users, similarity, stats = load_data()
    except (OSError, ValueError, KeyError) as error:
        st.error(f"Could not load hotel data from {DATA_DIR}: {error}")
        st.stop()

    st.sidebar.header("Recommendation filters")
    mode = st.sidebar.radio("Recommend by", ["Similar hotels", "For a traveler"])
    if mode == "Similar hotels":
        selected_hotel = st.sidebar.selectbox("Preferred hotel", sorted(similarity.columns))
    else:
        try:
            model = load_recommender()
        except (OSError, ValueError, KeyError) as error:
            st.error(f"Could not load the saved recommender from {MODEL_PATH}: {error}")
            st.stop()
        user_ids = model["train_matrix"].index.tolist()
        names = users.set_index("code")["name"].to_dict()
        user_code = st.sidebar.selectbox(
            "Traveler", user_ids, format_func=lambda code: f"{code} - {names.get(code, 'Traveler')}"
        )

    top_k = st.sidebar.slider("Number of recommendations", 1, 8, 4)
    min_rate, max_rate = float(stats["avg_daily_rate"].min()), float(stats["avg_daily_rate"].max())
    budget = (min_rate, max_rate)
    if min_rate < max_rate:
        budget = st.sidebar.slider("Daily budget ($)", min_rate, max_rate, budget)

    if mode == "Similar hotels":
        recommendations = recommend_similar_hotels(selected_hotel, similarity, stats, budget, top_k)
        score_label = "Hotel similarity"
    else:
        recommendations = recommend_for_traveler(user_code, model, stats, budget, top_k)
        score_label = "Destination score"
        st.caption("New destinations are based on the saved training history; scores are not probabilities.")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Recommended hotels")
        if recommendations.empty:
            st.info("No recommendations match this selection. Try another hotel, traveler, or budget.")
        else:
            display = recommendations.rename(columns={
                "name": "Hotel", "location": "Destination", "score": score_label,
                "avg_daily_rate": "Avg daily rate ($)", "avg_stay_days": "Avg stay (days)",
                "total_bookings": "Bookings",
            })
            st.dataframe(display.round(3), width="stretch", hide_index=True)
    with right:
        st.subheader("Average daily rates")
        chart = stats.assign(property=stats["name"] + " - " + stats["location"])
        st.bar_chart(chart.set_index("property")["avg_daily_rate"])

    st.divider()
    a, b, c, d = st.columns(4)
    a.metric("Total travelers", f"{users['code'].nunique():,}")
    b.metric("Hotel bookings", f"{len(hotels):,}")
    c.metric("Hotel properties", len(stats))
    d.metric("Avg booked nightly rate", f"${hotels['price'].mean():.2f}")


if __name__ == "__main__":
    main()
