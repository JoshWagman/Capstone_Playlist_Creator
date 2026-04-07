"""Visualise human survey responses for the Capstone Playlist Creator."""

from __future__ import annotations

import pathlib, textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

CSV_PATH = pathlib.Path("/Users/joshwagman/Downloads")  \
           / "Capstone Playlist Rankings (Responses) - Form Responses 1.csv"
OUT_DIR  = pathlib.Path(__file__).resolve().parent
OUT_DIR.mkdir(exist_ok=True)

SONG_COLS = [
    "Rank song #1 based on how well it represents the base songs",
    "Rank song #2 based on how well it represents the base songs",
    "Rank song #3 based on how well it represents the base songs",
    "Rank song #4 based on how well it represents the base songs",
    "Rank song #5 based on how well it represents the base songs",
    "Rank song #6 based on how well it represents the base songs",
]
COHESION_COL = "Rank the entire playlist based on its cohesion"
SEED_COL     = 'How many "base" songs did you choose?'
GENRE_COL    = "What genres did you base the playlist off?"

PALETTE = ["#6366f1", "#8b5cf6", "#a78bfa", "#c4b5fd", "#e0e7ff", "#818cf8"]

df = pd.read_csv(CSV_PATH)
song_ratings = df[SONG_COLS].values  # (n_responses, 6)


# ── 1. Distribution of all individual song-fit ratings ────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
all_ratings = song_ratings.flatten()
all_ratings = all_ratings[~np.isnan(all_ratings)]
bins = np.arange(0.5, 11.5, 1)
ax.hist(all_ratings, bins=bins, color=PALETTE[0], edgecolor="white", linewidth=0.8)
ax.set_xlabel("Song Fit Rating (1–10)", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("Distribution of Individual Song Fit Ratings", fontsize=13, weight="bold")
ax.set_xticks(range(1, 11))
ax.xaxis.set_minor_locator(mticker.NullLocator())
mean_val = np.nanmean(all_ratings)
ax.axvline(mean_val, color="#ef4444", ls="--", lw=1.5, label=f"Mean = {mean_val:.1f}")
ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig(OUT_DIR / "survey_song_fit_distribution.png", dpi=200)
plt.close(fig)
print("✓ survey_song_fit_distribution.png")


# ── 2. Average fit rating by song position ────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
means = np.nanmean(song_ratings, axis=0)
stds  = np.nanstd(song_ratings, axis=0)
positions = np.arange(1, 7)
ax.bar(positions, means, yerr=stds, color=PALETTE, edgecolor="white",
       linewidth=0.8, capsize=4, error_kw={"lw": 1.2})
ax.set_xlabel("Song Position in Playlist", fontsize=11)
ax.set_ylabel("Average Fit Rating", fontsize=11)
ax.set_title("Average Song Fit by Position", fontsize=13, weight="bold")
ax.set_xticks(positions)
ax.set_ylim(0, 11)
for i, m in enumerate(means):
    ax.text(positions[i], m + stds[i] + 0.3, f"{m:.1f}", ha="center",
            fontsize=9, weight="bold", color="#374151")
fig.tight_layout()
fig.savefig(OUT_DIR / "survey_fit_by_position.png", dpi=200)
plt.close(fig)
print("✓ survey_fit_by_position.png")


# ── 3. Playlist cohesion score distribution ───────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
cohesion = df[COHESION_COL].dropna().values
ax.hist(cohesion, bins=bins, color="#10b981", edgecolor="white", linewidth=0.8)
ax.set_xlabel("Cohesion Rating (1–10)", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("Playlist Cohesion Score Distribution", fontsize=13, weight="bold")
ax.set_xticks(range(1, 11))
ax.xaxis.set_minor_locator(mticker.NullLocator())
c_mean = np.nanmean(cohesion)
ax.axvline(c_mean, color="#ef4444", ls="--", lw=1.5, label=f"Mean = {c_mean:.1f}")
ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig(OUT_DIR / "survey_cohesion_distribution.png", dpi=200)
plt.close(fig)
print("✓ survey_cohesion_distribution.png")


# ── 4. Cohesion vs mean song fit (scatter) ────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
mean_fit = np.nanmean(song_ratings, axis=1)
ax.scatter(mean_fit, df[COHESION_COL], s=70, c=PALETTE[0], edgecolor="white",
           linewidth=0.6, alpha=0.85, zorder=3)
z = np.polyfit(mean_fit, df[COHESION_COL].values, 1)
x_line = np.linspace(mean_fit.min() - 0.3, mean_fit.max() + 0.3, 50)
ax.plot(x_line, np.polyval(z, x_line), color="#ef4444", ls="--", lw=1.3,
        label=f"Trend (r = {np.corrcoef(mean_fit, df[COHESION_COL])[0,1]:.2f})")
ax.set_xlabel("Mean Song Fit (per response)", fontsize=11)
ax.set_ylabel("Playlist Cohesion Rating", fontsize=11)
ax.set_title("Cohesion vs. Average Song Fit", fontsize=13, weight="bold")
ax.legend(fontsize=10)
ax.set_xlim(1, 10.5)
ax.set_ylim(1, 10.5)
fig.tight_layout()
fig.savefig(OUT_DIR / "survey_cohesion_vs_fit.png", dpi=200)
plt.close(fig)
print("✓ survey_cohesion_vs_fit.png")


# ── 5. Average song fit & cohesion grouped by number of seed songs ────────────
fig, ax = plt.subplots(figsize=(6, 4.5))
df["_mean_fit"] = mean_fit
grouped = df.groupby(SEED_COL).agg(
    avg_fit=("_mean_fit", "mean"),
    avg_cohesion=(COHESION_COL, "mean"),
    count=(SEED_COL, "size"),
).reset_index()
x = np.arange(len(grouped))
w = 0.32
bars1 = ax.bar(x - w/2, grouped["avg_fit"], w, label="Avg Song Fit",
               color=PALETTE[0], edgecolor="white")
bars2 = ax.bar(x + w/2, grouped["avg_cohesion"], w, label="Avg Cohesion",
               color="#10b981", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels([f"{int(v)} seed{'s' if v > 1 else ''}\n(n={int(c)})"
                    for v, c in zip(grouped[SEED_COL], grouped["count"])])
ax.set_ylabel("Average Rating (1–10)", fontsize=11)
ax.set_title("Ratings by Number of Seed Songs", fontsize=13, weight="bold")
ax.set_ylim(0, 11)
ax.legend(fontsize=10)
for bar_group in [bars1, bars2]:
    for bar in bar_group:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.25, f"{h:.1f}",
                ha="center", fontsize=9, weight="bold", color="#374151")
fig.tight_layout()
fig.savefig(OUT_DIR / "survey_ratings_by_seeds.png", dpi=200)
plt.close(fig)
print("✓ survey_ratings_by_seeds.png")


# ── 6. Genre frequency bar chart ─────────────────────────────────────────────
genre_series = df[GENRE_COL].dropna().str.split(r",\s*")
all_genres = [g.strip() for gs in genre_series for g in gs]
genre_counts = pd.Series(all_genres).value_counts()

fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.barh(genre_counts.index[::-1], genre_counts.values[::-1],
               color=PALETTE[0], edgecolor="white", linewidth=0.8)
ax.set_xlabel("Number of Responses", fontsize=11)
ax.set_title("Genre Frequency Across Responses", fontsize=13, weight="bold")
for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.15, bar.get_y() + bar.get_height()/2, str(int(w)),
            va="center", fontsize=9, weight="bold", color="#374151")
fig.tight_layout()
fig.savefig(OUT_DIR / "survey_genre_frequency.png", dpi=200)
plt.close(fig)
print("✓ survey_genre_frequency.png")


# ── 7. Heatmap of all ratings (respondent × song position + cohesion) ─────────
fig, ax = plt.subplots(figsize=(8, 7))
heat_data = df[SONG_COLS + [COHESION_COL]].copy()
heat_data.columns = [f"Song {i}" for i in range(1, 7)] + ["Cohesion"]
heat_data.index = [f"R{i+1}" for i in range(len(heat_data))]
cmap = matplotlib.colormaps.get_cmap("RdYlGn")
im = ax.imshow(heat_data.values, aspect="auto", cmap=cmap, vmin=1, vmax=10)
ax.set_xticks(range(heat_data.shape[1]))
ax.set_xticklabels(heat_data.columns, fontsize=9)
ax.set_yticks(range(heat_data.shape[0]))
ax.set_yticklabels(heat_data.index, fontsize=8)
for i in range(heat_data.shape[0]):
    for j in range(heat_data.shape[1]):
        v = heat_data.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=8,
                    color="white" if v < 4 else "black")
ax.set_title("Rating Heatmap (all respondents)", fontsize=13, weight="bold")
fig.colorbar(im, ax=ax, shrink=0.6, label="Rating")
fig.tight_layout()
fig.savefig(OUT_DIR / "survey_rating_heatmap.png", dpi=200)
plt.close(fig)
print("✓ survey_rating_heatmap.png")


# ── 8. Cohesion by genre (box plot) ──────────────────────────────────────────
genre_cohesion = []
for _, row in df.iterrows():
    if pd.isna(row[GENRE_COL]):
        continue
    genres = [g.strip() for g in row[GENRE_COL].split(",")]
    for g in genres:
        genre_cohesion.append({"genre": g, "cohesion": row[COHESION_COL]})
gc_df = pd.DataFrame(genre_cohesion)
genre_order = gc_df.groupby("genre")["cohesion"].median().sort_values(ascending=False).index

fig, ax = plt.subplots(figsize=(8, 4.5))
bp_data = [gc_df[gc_df["genre"] == g]["cohesion"].values for g in genre_order]
bp = ax.boxplot(bp_data, patch_artist=True, labels=genre_order, widths=0.5)
for i, patch in enumerate(bp["boxes"]):
    patch.set_facecolor(PALETTE[i % len(PALETTE)])
    patch.set_edgecolor("white")
ax.set_ylabel("Cohesion Rating", fontsize=11)
ax.set_title("Playlist Cohesion by Genre", fontsize=13, weight="bold")
ax.tick_params(axis="x", rotation=35, labelsize=9)
fig.tight_layout()
fig.savefig(OUT_DIR / "survey_cohesion_by_genre.png", dpi=200)
plt.close(fig)
print("✓ survey_cohesion_by_genre.png")


# ── 9. Average cohesion per genre (bar chart) ────────────────────────────────
avg_coh = gc_df.groupby("genre")["cohesion"].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(range(len(avg_coh)), avg_coh.values, color=PALETTE[0],
              edgecolor="white", linewidth=0.8)
ax.set_xticks(range(len(avg_coh)))
ax.set_xticklabels(avg_coh.index, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Average Cohesion Rating", fontsize=11)
ax.set_title("Average Playlist Cohesion by Genre", fontsize=13, weight="bold")
ax.set_ylim(0, 10.5)
for i, (bar, val) in enumerate(zip(bars, avg_coh.values)):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.2, f"{val:.1f}",
            ha="center", fontsize=9, weight="bold", color="#374151")
fig.tight_layout()
fig.savefig(OUT_DIR / "survey_avg_cohesion_by_genre.png", dpi=200)
plt.close(fig)
print("✓ survey_avg_cohesion_by_genre.png")


# ── 10. Summary stats table ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 2.5))
ax.axis("off")
summary = [
    ["Metric", "Value"],
    ["Responses", f"{len(df)}"],
    ["Mean Song Fit", f"{np.nanmean(all_ratings):.2f}"],
    ["Median Song Fit", f"{np.nanmedian(all_ratings):.1f}"],
    ["Mean Cohesion", f"{np.nanmean(cohesion):.2f}"],
    ["Median Cohesion", f"{np.nanmedian(cohesion):.1f}"],
    ["Songs Rated ≥ 7", f"{(all_ratings >= 7).sum()} / {len(all_ratings)}  "
                        f"({100*(all_ratings >= 7).mean():.0f}%)"],
    ["Cohesion ≥ 7", f"{(cohesion >= 7).sum()} / {len(cohesion)}  "
                     f"({100*(cohesion >= 7).mean():.0f}%)"],
]
table = ax.table(cellText=summary[1:], colLabels=summary[0],
                 loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)
for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor("#d1d5db")
    if r == 0:
        cell.set_facecolor(PALETTE[0])
        cell.set_text_props(color="white", weight="bold")
    else:
        cell.set_facecolor("#f9fafb" if r % 2 == 0 else "white")
ax.set_title("Survey Summary", fontsize=13, weight="bold", pad=15)
fig.tight_layout()
fig.savefig(OUT_DIR / "survey_summary_table.png", dpi=200)
plt.close(fig)
print("✓ survey_summary_table.png")

print(f"\nDone — 9 figures saved to {OUT_DIR}")
