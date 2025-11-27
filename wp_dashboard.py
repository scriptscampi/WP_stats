#!/usr/bin/env python3
"""

Cluster NFL teams by season into style/archetype groups using nflreadpy.

Outputs (to --outdir):
- team_season_features_<year>.csv  : raw engineered features per team
- team_archetypes_<year>.csv       : cluster + label per team
- cluster_summary_<year>.csv       : centroid means by cluster
- team_archetypes_<year>.png       : PCA scatter with labels



Requires: nflreadpy, pandas, numpy, scikit-learn, matplotlib
"""
import sys
import argparse
import os
from typing import List, Dict

import numpy as np
import pandas as pd
import nflreadpy as nfl
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# ----------------------- Helpers -----------------------

def _col_exists(df: pd.DataFrame, name: str) -> bool:
    return name in df.columns

def compute_wp_for_posteam(pbp: pd.DataFrame) -> pd.Series:
    """
    Build a per-row Win Probability for the 'posteam' (offense) if WP columns exist.
    Fallback returns NaNs if absent.
    """
    cols = pbp.columns
    if "home_wp_post" in cols and "away_wp_post" in cols and {"home_team", "away_team", "posteam"}.issubset(cols):
        wp_home = pbp["home_wp_post"].astype(float)
        wp_away = pbp["away_wp_post"].astype(float)
        # posteam is offense; if offense == home -> use home_wp, else away_wp
        is_home_off = pbp["posteam"] == pbp["home_team"]
        team_wp = np.where(is_home_off, wp_home, wp_away)
        return pd.Series(team_wp, index=pbp.index)
    elif "wp" in cols:
        # Some datasets have a generic wp column (offense perspective may vary)
        return pbp["wp"].astype(float)
    else:
        return pd.Series(np.nan, index=pbp.index)


def safe_success(col: pd.Series) -> pd.Series:
    # Normalize common truthy values to 0/1 for success-rate mean
    return col.fillna(0).astype(float)


# ----------------------- Feature Engineering -----------------------

def build_team_season_features(year: int) -> pd.DataFrame:
    pbp = nfl.import_pbp_data(years=[year])

    # Regular season only when flagged
    if _col_exists(pbp, "season_type"):
        pbp = pbp[pbp["season_type"].str.upper() == "REG"].copy()

    # Basic safety: keep only real plays (some rows are timeouts, penalties-only, etc.)
    # We’ll still allow those where yards_gained/EPA exist for robustness.
    if _col_exists(pbp, "play_id"):
        pbp = pbp[pbp["play_id"].notna()].copy()

    # Offense-side aggregates (posteam)
    off = (
        pbp.groupby("posteam", as_index=False)
           .agg({
               "epa": "mean" if _col_exists(pbp, "epa") else "mean",
               "yards_gained": "mean" if _col_exists(pbp, "yards_gained") else "mean",
               "play_id": "count" if _col_exists(pbp, "play_id") else "size",
           })
           .rename(columns={
               "posteam": "team",
               "epa": "off_epa",
               "yards_gained": "off_ypp",
               "play_id": "off_plays",
           })
    )

    # Success rate (avg of success flag)
    if _col_exists(pbp, "success"):
        succ_off = pbp.groupby("posteam", as_index=False)["success"].mean()
        succ_off.rename(columns={"posteam": "team", "success": "off_success"}, inplace=True)
        off = off.merge(succ_off, on="team", how="left")
    else:
        off["off_success"] = np.nan

    # Turnovers (INT + fumbles lost) per offensive play
    int_col = "interception"
    fum_col = "fumble_lost" if "fumble_lost" in pbp.columns else ("fumble" if "fumble" in pbp.columns else None)
    tmp = pbp.copy()
    tmp[int_col] = tmp[int_col] if int_col in tmp.columns else 0
    tmp[fum_col] = tmp[fum_col] if fum_col in tmp.columns else 0
    to_off = (
        tmp.groupby("posteam", as_index=False)
           .agg({int_col: "sum", fum_col: "sum"})
           .rename(columns={"posteam": "team"})
    )
    to_off["off_turnovers"] = to_off[int_col].fillna(0) + to_off[fum_col].fillna(0)
    off = off.merge(to_off[["team", "off_turnovers"]], on="team", how="left")
    off["off_tov_rate"] = off["off_turnovers"] / off["off_plays"].replace(0, np.nan)

    # Pass rate & pace
    if "pass" in pbp.columns:
        pass_rate = pbp.groupby("posteam", as_index=False)["pass"].mean().rename(columns={"posteam": "team", "pass": "pass_rate"})
        off = off.merge(pass_rate, on="team", how="left")
    else:
        off["pass_rate"] = np.nan

    # Defensive aggregates (defteam)
    deff = (
        pbp.groupby("defteam", as_index=False)
           .agg({
               "epa": "mean",
               "yards_gained": "mean",
               "play_id": "count" if _col_exists(pbp, "play_id") else "size",
           })
           .rename(columns={
               "defteam": "team",
               "epa": "def_epa_allowed",
               "yards_gained": "def_ypp_allowed",
               "play_id": "def_plays_faced",
           })
    )
    if _col_exists(pbp, "success"):
        succ_def = pbp.groupby("defteam", as_index=False)["success"].mean()
        succ_def.rename(columns={"defteam": "team", "success": "def_success_allowed"}, inplace=True)
        deff = deff.merge(succ_def, on="team", how="left")
    else:
        deff["def_success_allowed"] = np.nan

    # Win Probability volatility (std over season of posteam WP)
    pbp = pbp.copy()
    pbp["posteam_wp"] = compute_wp_for_posteam(pbp)
    if pbp["posteam_wp"].notna().any():
        wp_std = pbp.groupby("posteam")["posteam_wp"].std().reset_index().rename(columns={"posteam": "team", "posteam_wp": "wp_std"})
        off = off.merge(wp_std, on="team", how="left")
    else:
        off["wp_std"] = np.nan

    # Merge offense + defense views
    feat = off.merge(deff, on="team", how="outer")

    # Weekly data for wins / points for / against
    weekly = nfl.import_weekly_data(years=[year])
    # Regular season filter if exists
    if _col_exists(weekly, "season_type"):
        weekly = weekly[weekly["season_type"].str.upper() == "REG"].copy()

    wk_team = weekly.groupby("team", as_index=False).agg({
        "result": lambda s: int((s == "W").sum()) if s.notna().any() else 0,
        "points": "sum",
        "points_allowed": "sum",
        "game_id": "nunique",
    }).rename(columns={
        "result": "wins",
        "points": "points_for",
        "points_allowed": "points_against",
        "game_id": "games",
    })

    feat = feat.merge(wk_team, left_on="team", right_on="team", how="left")

    # Derived diffs & pace
    feat["epa_diff"] = feat["off_epa"] - feat["def_epa_allowed"]
    feat["success_diff"] = feat["off_success"] - feat["def_success_allowed"]
    feat["ypp_diff"] = feat["off_ypp"] - feat["def_ypp_allowed"]
    feat["pace_plays_per_game"] = feat["off_plays"] / feat["games"].replace(0, np.nan)

    # Clean feature order
    cols = [
        "team", "games", "wins",
        "points_for", "points_against",
        "off_plays", "def_plays_faced", "pace_plays_per_game",
        "off_epa", "off_success", "off_ypp", "pass_rate", "off_turnovers", "off_tov_rate",
        "def_epa_allowed", "def_success_allowed", "def_ypp_allowed",
        "epa_diff", "success_diff", "ypp_diff",
        "wp_std",
    ]
    # Ensure all exist
    for c in cols:
        if c not in feat.columns:
            feat[c] = np.nan

    return feat[cols].sort_values("team").reset_index(drop=True)


# ----------------------- Clustering -----------------------

def auto_label_cluster(centroid: pd.Series, league_means: pd.Series) -> str:
    """
    Create a heuristic label from cluster centroid values relative to league means.
    Simple, transparent rules; tweak as needed.
    """
    def z(v, m, s):  # z-score helper
        return 0 if pd.isna(v) or pd.isna(m) or s == 0 or pd.isna(s) else (v - m) / s

    # We’ll compute z-scores on key axes
    axes = ["epa_diff", "off_epa", "def_epa_allowed", "off_tov_rate", "wins", "pass_rate", "wp_std"]
    stats = {}
    for a in axes:
        mean = league_means[a]["mean"]
        stdv = league_means[a]["std"]
        stats[a] = z(centroid[a], mean, stdv)

    # Lower def_epa_allowed is better; invert for readability
    z_def_good = -stats["def_epa_allowed"]

    # Rule-based labeling
    if stats["epa_diff"] > 0.8 and z_def_good > 0.6 and stats["off_tov_rate"] < -0.3 and stats["wins"] > 0.6:
        return "Elite Balanced"
    if stats["off_epa"] > 0.8 and z_def_good < 0.0:
        return "Shootout Offense"
    if z_def_good > 0.9 and stats["off_epa"] < 0.0:
        return "Defensive Grinder"
    if stats["epa_diff"] < -0.6 and stats["wins"] < -0.3:
        return "Offensive Struggler"
    if abs(stats["epa_diff"]) < 0.2 and (stats["wp_std"] > 0.5 or stats["off_tov_rate"] > 0.4):
        return "Chaotic Wildcard"
    if stats["pass_rate"] > 0.8 and stats["off_epa"] > 0.2:
        return "Air Raid Lean"
    if stats["pass_rate"] < -0.8 and z_def_good > 0.2:
        return "Ground Control"
    # Fallbacks
    if stats["epa_diff"] > 0.4:
        return "Efficient Positive"
    if stats["epa_diff"] < -0.4:
        return "Inefficient Negative"
    return "Balanced Middle"


def run_clustering(features: pd.DataFrame, n_clusters: int, random_state: int = 42) -> Dict[str, pd.DataFrame]:
    # Choose features for clustering
    cluster_cols = [
        "epa_diff", "off_epa", "def_epa_allowed",
        "off_success", "def_success_allowed", "success_diff",
        "off_ypp", "def_ypp_allowed", "ypp_diff",
        "off_tov_rate", "pass_rate", "pace_plays_per_game",
        "wins", "wp_std",
    ]
    X = features[cluster_cols].astype(float).fillna(features[cluster_cols].median(numeric_only=True))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = km.fit_predict(X_scaled)

    out = features.copy()
    out["cluster"] = labels

    # Cluster summaries
    centroids = pd.DataFrame(km.cluster_centers_, columns=cluster_cols)
    # Transform back to original scale for readable means
    inv_centroids = pd.DataFrame(scaler.inverse_transform(centroids), columns=cluster_cols)
    cluster_summary = inv_centroids.copy()
    cluster_summary["cluster"] = np.arange(n_clusters)

    # League stats for labeling
    league_means = features[cluster_cols].agg(["mean", "std"]).T

    # Label clusters
    labels_map = {}
    for i in range(n_clusters):
        lbl = auto_label_cluster(cluster_summary.loc[i], league_means)
        labels_map[i] = lbl

    out["archetype"] = out["cluster"].map(labels_map)
    cluster_summary["archetype"] = cluster_summary["cluster"].map(labels_map)

    return {
        "teams": out,
        "cluster_summary": cluster_summary[["cluster", "archetype"] + cluster_cols],
        "X_scaled": X_scaled,
        "scaler": scaler,
    }


def save_outputs(year: int, outdir: str, features: pd.DataFrame, teams: pd.DataFrame, cluster_summary: pd.DataFrame):
    os.makedirs(outdir, exist_ok=True)
    f1 = os.path.join(outdir, f"team_season_features_{year}.csv")
    f2 = os.path.join(outdir, f"team_archetypes_{year}.csv")
    f3 = os.path.join(outdir, f"cluster_summary_{year}.csv")
    features.to_csv(f1, index=False)
    teams.to_csv(f2, index=False)
    cluster_summary.to_csv(f3, index=False)
    return f1, f2, f3


def plot_pca(year: int, outdir: str, X_scaled: np.ndarray, teams_df: pd.DataFrame):
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    teams_df = teams_df.copy()
    teams_df["pca1"] = coords[:, 0]
    teams_df["pca2"] = coords[:, 1]

    plt.figure(figsize=(9, 7))
    for label, grp in teams_df.groupby("archetype"):
        plt.scatter(grp["pca1"], grp["pca2"], label=label, s=80, alpha=0.8)
    # annotate team codes
    for _, r in teams_df.iterrows():
        plt.text(r["pca1"], r["pca2"], r["team"], fontsize=8, ha="center", va="center")

    plt.title(f"Team-Season Archetypes — {year} (PCA)")
    plt.xlabel("PC1 (Offense–Defense / Efficiency)")
    plt.ylabel("PC2 (Pace / Chaos / Style)")
    plt.legend(frameon=True, fontsize=8, loc="best")
    plt.tight_layout()

    out_png = os.path.join(outdir, f"team_archetypes_{year}.png")
    plt.savefig(out_png, dpi=150)
    plt.close()
    return out_png


# ----------------------- Main -----------------------

def main():
    if "streamlit" in sys.argv[0]:
        pass
    else:
        ap = argparse.ArgumentParser(description="Cluster NFL teams into season-long archetypes.")
        ap.add_argument("--year", type=int, required=True, dedfault=2025 help="Season year, e.g., 2025")
        ap.add_argument("--clusters", type=int, default=5, help="Number of K-Means clusters (default: 5)")
        ap.add_argument("--outdir", type=str, default=".", help="Output directory (default: current)")
        args = ap.parse_args()
        
        print(f"[INFO] Building team-season features for {args.year} …")
        features = build_team_season_features(args.year)
        
        print(f"[INFO] Clustering into {args.clusters} archetypes …")
        res = run_clustering(features, n_clusters=args.clusters, random_state=42)
        teams = res["teams"]
        cluster_summary = res["cluster_summary"]
        
        print(f"[INFO] Saving CSV outputs …")
        f1, f2, f3 = save_outputs(args.year, args.outdir, features, teams, cluster_summary)
        print(f"  • Features: {f1}")
        print(f"  • Teams   : {f2}")
        print(f"  • Summary : {f3}")
        
        print(f"[INFO] Rendering PCA plot …")
        png_path = plot_pca(args.year, args.outdir, res["X_scaled"], teams)
        print(f"  • Plot    : {png_path}")
        
        # Console preview
        preview_cols = ["team", "wins", "epa_diff", "off_epa", "def_epa_allowed", "off_tov_rate", "pass_rate", "wp_std", "archetype", "cluster"]
        print("\n[PREVIEW] Team Archetypes:")
        print(teams[preview_cols].sort_values(["cluster", "team"]).to_string(index=False))
        
        print("\n[PREVIEW] Cluster Centroids (original scale):")
        print(cluster_summary.to_string(index=False))


if __name__ == "__main__":
    main()
