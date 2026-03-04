"""
build_top500k_catalog.py
-------------------------
Combines all 9 Spotify datasets, deduplicates by track ID, fills in metadata
from the best available source, then outputs the 500,000 most popular tracks.

Output:
  data/processed/top500k_catalog.csv

Usage:
  python Backend/build_top500k_catalog.py
"""

import pandas as pd
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path("/Users/williamgorman/Documents/GitHub/Capstone_Playlist_Creator/Untitled")
RAW_DIR      = PROJECT_ROOT / "data/raw"
OUT_CSV      = PROJECT_ROOT / "data/processed/top500k_catalog.csv"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

TOP_N = 500_000

# ---------------------------------------------------------------------------
# Target schema
# ---------------------------------------------------------------------------

AUDIO_FEATURE_COLS = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo", "time_signature", "duration_ms",
]

METADATA_COLS = ["track_name", "artist_name", "album_name", "genre",
                 "release_date", "popularity", "explicit"]

ALL_COLS = ["track_id"] + METADATA_COLS + AUDIO_FEATURE_COLS

# ---------------------------------------------------------------------------
# Per-dataset configs
# Datasets are ordered by data quality — earlier = higher priority when
# a track appears in multiple datasets.
# ---------------------------------------------------------------------------

DATASET_CONFIGS = [
    {
        "file": "tracks_features.csv",
        "encoding": "utf-8",
        "id_col": "id",
        "rename": {
            "name":        "track_name",
            "artists":     "artist_name",
            "album":       "album_name",
            "release_date":"release_date",
        },
    },
    {
        "file": "Dataset_with_Genres.csv",
        "encoding": "utf-8",
        "id_col": "id",
        "rename": {
            "name":                     "track_name",
            "artists":                  "artist_name",
            "release_date":             "release_date",
            "consolidates_genre_lists": "genre",
        },
    },
    {
        "file": "dataset.csv",
        "encoding": "utf-8",
        "id_col": "track_id",
        "rename": {
            "track_name":  "track_name",
            "artists":     "artist_name",
            "album_name":  "album_name",
            "track_genre": "genre",
        },
    },
    {
        "file": "spotify dataset.csv",
        "encoding": "utf-8",
        "id_col": "track_id",
        "rename": {
            "track_name":               "track_name",
            "track_artist":             "artist_name",
            "track_album_name":         "album_name",
            "track_album_release_date": "release_date",
            "track_popularity":         "popularity",
            "playlist_genre":           "genre",
        },
    },
    {
        "file": "SpotifyAudioFeaturesApril2019.csv",
        "encoding": "utf-8",
        "id_col": "track_id",
        "rename": {
            "track_name":  "track_name",
            "artist_name": "artist_name",
        },
    },
    {
        "file": "spotify_data.csv",
        "encoding": "latin-1",
        "id_col": "track_id",
        "rename": {
            "track_name":  "track_name",
            "artist_name": "artist_name",
            "genre":       "genre",
            "year":        "release_date",
        },
    },
    {
        "file": "song_track.csv",
        "encoding": "latin-1",
        "id_col": "track_id",
        "rename": {
            "track_name":  "track_name",
            "artist_name": "artist_name",
            "genre":       "genre",
        },
    },
    {
        "file": "Music Info.csv",
        "encoding": "latin-1",
        "id_col": "spotify_id",
        "rename": {
            "name":   "track_name",
            "artist": "artist_name",
            "genre":  "genre",
            "year":   "release_date",
        },
    },
    {
        # No track ID — cannot deduplicate, skip
        "file": "Popular_Spotify_Songs.csv",
        "skip": True,
        "reason": "No Spotify track ID column — cannot deduplicate reliably.",
    },
]

PERCENT_COLS = ["danceability", "energy", "valence", "acousticness",
                "instrumentalness", "liveness", "speechiness"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace("spotify:track:", "", regex=False).str.strip()


def normalize_percent_cols(df: pd.DataFrame) -> pd.DataFrame:
    for col in PERCENT_COLS:
        if col in df.columns:
            if df[col].dropna().max() > 1.0:
                df[col] = df[col] / 100.0
    return df


def load_dataset(config: dict) -> Optional[pd.DataFrame]:
    if config.get("skip"):
        print(f"  Skipping {config['file']}: {config['reason']}")
        return None

    filepath = RAW_DIR / config["file"]
    if not filepath.exists():
        print(f"  ⚠️  Not found, skipping: {config['file']}")
        return None

    try:
        df = pd.read_csv(filepath, low_memory=False, encoding=config.get("encoding", "utf-8"))
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, low_memory=False, encoding="latin-1")

    id_col = config["id_col"]
    if id_col not in df.columns:
        print(f"  ⚠️  ID column '{id_col}' not found in {config['file']}, skipping.")
        return None

    df["track_id"] = normalize_id(df[id_col])
    df = df.rename(columns=config.get("rename", {}))
    df = normalize_percent_cols(df)

    # Keep only schema columns that exist in this file
    cols_present = [c for c in ALL_COLS if c in df.columns]
    df = df[cols_present].copy()

    df = df.dropna(subset=["track_id"])
    df = df[df["track_id"].str.len() > 0]

    print(f"  Loaded {len(df):,} rows  |  {config['file']}")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 55)
    print("STEP 1: Loading all datasets")
    print("=" * 55)

    frames = []
    for config in DATASET_CONFIGS:
        df = load_dataset(config)
        if df is not None:
            frames.append(df)

    if not frames:
        raise RuntimeError("No datasets loaded — check your data/raw/ folder.")

    # Stack all rows
    combined = pd.concat(frames, ignore_index=True)
    print(f"\nTotal rows across all datasets: {len(combined):,}")

    # ---------------------------------------------------------------------------
    # Deduplication strategy:
    # For each track_id, keep the row with the highest popularity score.
    # If popularity is missing, fill with 0 so it loses ties gracefully.
    # This also means higher-quality rows naturally win since better datasets
    # tend to have more complete popularity scores.
    # ---------------------------------------------------------------------------

    print("\n" + "=" * 55)
    print("STEP 2: Deduplicating by track ID")
    print("=" * 55)

    combined["popularity"] = pd.to_numeric(combined.get("popularity", 0), errors="coerce").fillna(0)
    combined = combined.sort_values("popularity", ascending=False)
    combined = combined.drop_duplicates(subset="track_id", keep="first")

    print(f"Unique tracks after dedup: {len(combined):,}")

    # ---------------------------------------------------------------------------
    # Select top 500k by popularity
    # ---------------------------------------------------------------------------

    print("\n" + "=" * 55)
    print(f"STEP 3: Selecting top {TOP_N:,} by popularity")
    print("=" * 55)

    top500k = combined.nlargest(TOP_N, "popularity").reset_index(drop=True)

    pop_min = top500k["popularity"].min()
    pop_max = top500k["popularity"].max()
    pop_mean = top500k["popularity"].mean()
    print(f"Popularity range: {pop_min:.0f} – {pop_max:.0f}  (mean: {pop_mean:.1f})")

    # ---------------------------------------------------------------------------
    # Coverage report
    # ---------------------------------------------------------------------------

    print("\n" + "=" * 55)
    print("COVERAGE REPORT")
    print("=" * 55)

    total = len(top500k)
    all_report_cols = METADATA_COLS + AUDIO_FEATURE_COLS
    for col in all_report_cols:
        if col in top500k.columns:
            filled = top500k[col].notna().sum()
            pct = 100 * filled / total
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"  {col:<22} {bar}  {pct:.0f}%")

    # ---------------------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------------------

    # Reorder columns cleanly
    final_cols = [c for c in ALL_COLS if c in top500k.columns]
    extra_cols = [c for c in top500k.columns if c not in final_cols]
    top500k = top500k[final_cols + extra_cols]

    top500k.to_csv(OUT_CSV, index=False)

    print(f"\n✅ Saved top {TOP_N:,} tracks to: {OUT_CSV}")
    print(f"   Rows: {len(top500k):,} | Columns: {len(top500k.columns)}")
    print(f"\nColumns: {list(top500k.columns)}")


if __name__ == "__main__":
    main()