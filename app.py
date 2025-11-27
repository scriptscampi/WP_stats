import streamlit as st
import pandas as pd
from wp_dashboard import build_team_season_features, run_clustering  # <-- import your functions

st.set_page_config(page_title="NFL Archetype Clusters", layout="wide")
st.title("NFL Team Archetype Clusters")

year = st.sidebar.number_input("Season year", min_value=1999, max_value=2035, value=2024, step=1)
n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 5)

if st.sidebar.button("Run clustering"):
    with st.spinner(f"Building features for {year}…"):
        features = build_team_season_features(year)

    with st.spinner(f"Clustering into {n_clusters} archetypes…"):
        res = run_clustering(features, n_clusters=n_clusters, random_state=42)
        teams = res["teams"]
        cluster_summary = res["cluster_summary"]

    st.subheader("Team Archetypes")
    st.dataframe(teams)

    st.subheader("Cluster Summary")
    st.dataframe(cluster_summary)
else:
    st.info("Choose a year and number of clusters, then click **Run clustering**.")
