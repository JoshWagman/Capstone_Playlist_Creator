"""
Generate visualisations from virtual listening test results.

Produces 5 figures saved as PNGs in the results/ directory:
  1. Cohesion breakdown bar chart
  2. Per-song fit score box plots
  3. Genre preference stacked bar chart
  4. Song fit score heatmap
  5. Radar charts: ideal vs actual audio features

Usage:
    python Backend/plot_listening_tests.py
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_RESULTS = os.path.join(_ROOT, "results")

SONG_CSV = os.path.join(_RESULTS, "song_scores.csv")
PLAYLIST_CSV = os.path.join(_RESULTS, "playlist_scores.csv")

song_df = pd.read_csv(SONG_CSV)
pl_df = pd.read_csv(PLAYLIST_CSV)

NAMES = pl_df["listener_name"].tolist()
SHORT = [n.split("/")[0].split(" ")[0] if "/" in n else n.split(" ")[0] for n in NAMES]

LISTENER_IDEALS = [
    {"danceability": 0.45, "energy": 0.35, "valence": 0.50, "acousticness": 0.75, "tempo": 110},
    {"danceability": 0.45, "energy": 0.80, "valence": 0.55, "acousticness": 0.15, "tempo": 128},
    {"danceability": 0.70, "energy": 0.65, "valence": 0.65, "acousticness": 0.15, "tempo": 118},
    {"danceability": 0.75, "energy": 0.70, "valence": 0.45, "acousticness": 0.10, "tempo": 130},
    {"danceability": 0.72, "energy": 0.85, "valence": 0.45, "acousticness": 0.05, "tempo": 128},
    {"danceability": 0.55, "energy": 0.50, "valence": 0.55, "acousticness": 0.40, "tempo": 115},
    {"danceability": 0.25, "energy": 0.15, "valence": 0.30, "acousticness": 0.90, "tempo": 95},
    {"danceability": 0.78, "energy": 0.75, "valence": 0.70, "acousticness": 0.12, "tempo": 100},
    {"danceability": 0.55, "energy": 0.55, "valence": 0.65, "acousticness": 0.50, "tempo": 120},
    {"danceability": 0.72, "energy": 0.78, "valence": 0.60, "acousticness": 0.10, "tempo": 125},
]

RADAR_FEATS = ["danceability", "energy", "valence", "acousticness", "tempo"]
TEMPO_MAX = 180.0


# ── 1. Cohesion Breakdown ──────────────────────────────────────

def plot_cohesion():
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(NAMES))
    w = 0.28

    bars_gc = ax.bar(x - w, pl_df["genre_consistency"], w, label="Genre Consistency")
    bars_fc = ax.bar(x, pl_df["feature_coherence"], w, label="Feature Coherence")
    bars_co = ax.bar(x + w, pl_df["cohesion_score"], w, label="Overall Cohesion")

    ax.set_ylabel("Score (1–10)")
    ax.set_title("Playlist Cohesion Breakdown by Listener")
    ax.set_xticks(x)
    ax.set_xticklabels(NAMES, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, 11)
    ax.legend(loc="upper right")
    ax.yaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(_RESULTS, "cohesion_breakdown.png"), dpi=150)
    plt.close(fig)
    print("  Saved cohesion_breakdown.png")


# ── 2. Fit Score Box Plots ─────────────────────────────────────

def plot_fit_boxplots():
    fig, ax = plt.subplots(figsize=(12, 6))

    data = [
        song_df[song_df["listener_id"] == lid]["fit_score"].values
        for lid in pl_df["listener_id"]
    ]
    bp = ax.boxplot(data, patch_artist=True, labels=NAMES, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white", markersize=5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(NAMES)))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)

    ax.set_ylabel("Fit Score (1–10)")
    ax.set_title("Per-Song Fit Score Distribution by Listener")
    ax.set_xticklabels(NAMES, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, 11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(_RESULTS, "fit_score_boxplots.png"), dpi=150)
    plt.close(fig)
    print("  Saved fit_score_boxplots.png")


# ── 3. Genre Preference Stacked Bar ───────────────────────────

def plot_genre_stacked():
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(NAMES))

    pct_loved = pl_df["pct_loved"].values
    pct_tolerated = pl_df["pct_tolerated"].values
    pct_despised = pl_df["pct_despised"].values
    pct_neutral = 100 - pct_loved - pct_tolerated - pct_despised

    ax.bar(x, pct_loved, label="Loved", color="#2ecc71")
    ax.bar(x, pct_tolerated, bottom=pct_loved, label="Tolerated", color="#f39c12")
    ax.bar(x, pct_neutral, bottom=pct_loved + pct_tolerated, label="Neutral", color="#95a5a6")
    ax.bar(x, pct_despised, bottom=pct_loved + pct_tolerated + pct_neutral,
           label="Despised", color="#e74c3c")

    ax.set_ylabel("% of Playlist")
    ax.set_title("Genre Preference Breakdown by Listener")
    ax.set_xticks(x)
    ax.set_xticklabels(NAMES, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(_RESULTS, "genre_preference_stacked.png"), dpi=150)
    plt.close(fig)
    print("  Saved genre_preference_stacked.png")


# ── 4. Song Fit Score Heatmap ──────────────────────────────────

def plot_heatmap():
    pivot = song_df.pivot(index="listener_name", columns="song_idx", values="fit_score")
    pivot = pivot.reindex(NAMES)

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=1, vmax=10)

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(range(1, pivot.shape[1] + 1), fontsize=7)
    ax.set_yticks(range(len(NAMES)))
    ax.set_yticklabels(NAMES, fontsize=8)
    ax.set_xlabel("Song Position in Playlist")
    ax.set_title("Per-Song Fit Scores (all listeners)")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Fit Score")

    fig.tight_layout()
    fig.savefig(os.path.join(_RESULTS, "fit_score_heatmap.png"), dpi=150)
    plt.close(fig)
    print("  Saved fit_score_heatmap.png")


# ── 5. Radar Charts: Ideal vs Actual ──────────────────────────

def _norm_feat(val, feat):
    if feat == "tempo":
        return val / TEMPO_MAX
    return val


def plot_radars():
    n_listeners = len(NAMES)
    cols = 5
    rows = (n_listeners + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 4),
                             subplot_kw=dict(polar=True))
    axes = axes.flatten()

    angles = np.linspace(0, 2 * np.pi, len(RADAR_FEATS), endpoint=False).tolist()
    angles += angles[:1]

    for i, (name, ideal) in enumerate(zip(NAMES, LISTENER_IDEALS)):
        ax = axes[i]
        listener_songs = song_df[song_df["listener_name"] == name]

        ideal_vals = [_norm_feat(ideal[f], f) for f in RADAR_FEATS]
        ideal_vals += ideal_vals[:1]

        from functions import _get_songs_df
        full_df = _get_songs_df()

        actual_means = {}
        matched = full_df[full_df["track_name"].isin(listener_songs["track_name"])]
        for feat in RADAR_FEATS:
            if feat in matched.columns:
                actual_means[feat] = matched[feat].mean()
            else:
                actual_means[feat] = 0
        actual_vals = [_norm_feat(actual_means.get(f, 0), f) for f in RADAR_FEATS]
        actual_vals += actual_vals[:1]

        ax.plot(angles, ideal_vals, "o-", linewidth=2, label="Ideal", color="#3498db")
        ax.fill(angles, ideal_vals, alpha=0.15, color="#3498db")
        ax.plot(angles, actual_vals, "s-", linewidth=2, label="Actual", color="#e74c3c")
        ax.fill(angles, actual_vals, alpha=0.15, color="#e74c3c")

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(RADAR_FEATS, fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_title(name, fontsize=9, pad=15)
        if i == 0:
            ax.legend(loc="upper right", fontsize=7, bbox_to_anchor=(1.3, 1.15))

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Ideal vs Actual Audio Features per Listener", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(_RESULTS, "radar_ideal_vs_actual.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  Saved radar_ideal_vs_actual.png")


if __name__ == "__main__":
    print("Generating listening test visualisations...")
    plot_cohesion()
    plot_fit_boxplots()
    plot_genre_stacked()
    plot_heatmap()
    plot_radars()
    print("Done — all plots saved to results/")
