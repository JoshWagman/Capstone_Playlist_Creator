"""
build_top50k_catalog.py
-------------------------
Combines all 9 Spotify datasets, deduplicates by track ID, fills in metadata
from the best available source, then outputs a training set of exactly 50,000
songs: the 1,000 most popular artists × their 5 most popular tracks each.

Additional cleaning steps:
  - Remastered/re-recorded/live/deluxe versions of tracks are removed so only
    the canonical studio recording survives per song title per artist.
  - Raw genre tags from all datasets are mapped onto 15 canonical genres.
    Tracks whose genre cannot be mapped are labelled "Unknown".

Output:
  data/processed/top50k_catalog.csv

Usage:
  python Backend/build_top50k_catalog.py
"""

import re

import pandas as pd
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path("/Users/williamgorman/Documents/GitHub/Capstone_Playlist_Creator/Untitled")
RAW_DIR      = PROJECT_ROOT / "data/raw"
OUT_CSV      = PROJECT_ROOT / "data/processed/top50k_catalog.csv"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

TOP_ARTISTS      = 5_000
SONGS_PER_ARTIST = 10

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
# Remaster / variant filter
# Tracks whose title contains any of these patterns (case-insensitive) are
# considered non-canonical and will be removed.  The list covers the most
# common Spotify naming conventions.
# ---------------------------------------------------------------------------

REMASTER_PATTERN = re.compile(
    r"""
    \b(
        remaster(ed)?           |   # "Remastered", "Remaster"
        re[-\s]?record(ed)?     |   # "Re-Recorded", "ReRecorded"
        re[-\s]?mix(ed)?        |   # "Remix", "Remixed"
        live(\s+version)?       |   # "Live", "Live Version"
        live\s+at\b             |   # "Live at ..."
        acoustic(\s+version)?   |   # "Acoustic Version"
        demo(\s+version)?       |   # "Demo", "Demo Version"
        radio\s+edit            |   # "Radio Edit"
        single\s+version        |   # "Single Version"
        deluxe(\s+edition)?     |   # "Deluxe", "Deluxe Edition"
        anniversary\s+edition   |   # "Anniversary Edition"
        expanded\s+edition      |   # "Expanded Edition"
        bonus\s+track           |   # "Bonus Track"
        orchestral\s+version    |   # "Orchestral Version"
        instrumental            |   # "Instrumental"
        karaoke                     # "Karaoke"
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# ---------------------------------------------------------------------------
# Genre mapping → 15 canonical genres
# Keys are lowercase substrings / keywords found in raw genre tags.
# The list is checked in order; the first match wins.
# ---------------------------------------------------------------------------

CANONICAL_GENRES = [
    "Hip-Hop/Rap",
    "Pop",
    "Rock/Classic Rock",
    "Country",
    "R&B",
    "Latin Music",
    "EDM",
    "Alternative/Indie",
    "Heavy Metal/Hard Rock",
    "K-Pop",
    "Jazz",
    "Classical",
    "Gospel/Contemporary Christian",
    "Reggae",
    "Folk/Americana",
]

# Each entry: (substring_to_match, canonical_genre)
# Order matters — more specific patterns should come before broad ones.
GENRE_MAP = [
    # Hip-Hop / Rap
    ("hip hop",         "Hip-Hop/Rap"),
    ("hip-hop",         "Hip-Hop/Rap"),
    ("hiphop",          "Hip-Hop/Rap"),
    ("rap",             "Hip-Hop/Rap"),
    ("trap",            "Hip-Hop/Rap"),
    ("drill",           "Hip-Hop/Rap"),
    ("grime",           "Hip-Hop/Rap"),

    # K-Pop  (before generic "pop" so it doesn't fall through)
    ("k-pop",           "K-Pop"),
    ("kpop",            "K-Pop"),
    ("k pop",           "K-Pop"),
    ("korean pop",      "K-Pop"),

    # Heavy Metal / Hard Rock  (before generic "rock")
    ("metal",           "Heavy Metal/Hard Rock"),
    ("hard rock",       "Heavy Metal/Hard Rock"),
    ("hardcore",        "Heavy Metal/Hard Rock"),
    ("death",           "Heavy Metal/Hard Rock"),   # death metal
    ("thrash",          "Heavy Metal/Hard Rock"),
    ("black metal",     "Heavy Metal/Hard Rock"),

    # Rock / Classic Rock
    ("classic rock",    "Rock/Classic Rock"),
    ("rock",            "Rock/Classic Rock"),
    ("punk",            "Rock/Classic Rock"),
    ("grunge",          "Rock/Classic Rock"),
    ("garage",          "Rock/Classic Rock"),
    ("emo",             "Rock/Classic Rock"),

    # Alternative / Indie
    ("alternative",     "Alternative/Indie"),
    ("indie",           "Alternative/Indie"),
    ("alt-",            "Alternative/Indie"),
    ("lo-fi",           "Alternative/Indie"),
    ("shoegaze",        "Alternative/Indie"),
    ("dream pop",       "Alternative/Indie"),

    # EDM
    ("edm",             "EDM"),
    ("electronic",      "EDM"),
    ("electronica",     "EDM"),
    ("dance",           "EDM"),
    ("house",           "EDM"),
    ("techno",          "EDM"),
    ("trance",          "EDM"),
    ("dubstep",         "EDM"),
    ("drum and bass",   "EDM"),
    ("dnb",             "EDM"),
    ("synthpop",        "EDM"),
    ("synth-pop",       "EDM"),

    # R&B
    ("r&b",             "R&B"),
    ("rnb",             "R&B"),
    ("rhythm and blues","R&B"),
    ("soul",            "R&B"),
    ("funk",            "R&B"),
    ("neo soul",        "R&B"),

    # Latin Music
    ("latin",           "Latin Music"),
    ("reggaeton",       "Latin Music"),
    ("salsa",           "Latin Music"),
    ("bachata",         "Latin Music"),
    ("cumbia",          "Latin Music"),
    ("merengue",        "Latin Music"),
    ("bossa nova",      "Latin Music"),
    ("samba",           "Latin Music"),
    ("flamenco",        "Latin Music"),

    # Reggae
    ("reggae",          "Reggae"),
    ("dancehall",       "Reggae"),
    ("ska",             "Reggae"),
    ("dub",             "Reggae"),

    # Country
    ("country",         "Country"),
    ("bluegrass",       "Country"),
    ("honky tonk",      "Country"),
    ("outlaw country",  "Country"),

    # Folk / Americana
    ("folk",            "Folk/Americana"),
    ("americana",       "Folk/Americana"),
    ("singer-songwriter","Folk/Americana"),
    ("singer songwriter","Folk/Americana"),
    ("acoustic",        "Folk/Americana"),

    # Jazz
    ("jazz",            "Jazz"),
    ("blues",           "Jazz"),
    ("swing",           "Jazz"),
    ("bebop",           "Jazz"),

    # Classical
    ("classical",       "Classical"),
    ("orchestra",       "Classical"),
    ("symphony",        "Classical"),
    ("opera",           "Classical"),
    ("chamber",         "Classical"),
    ("baroque",         "Classical"),
    ("piano",           "Classical"),

    # Gospel / Contemporary Christian
    ("gospel",          "Gospel/Contemporary Christian"),
    ("christian",       "Gospel/Contemporary Christian"),
    ("worship",         "Gospel/Contemporary Christian"),
    ("ccm",             "Gospel/Contemporary Christian"),
    ("contemporary christian", "Gospel/Contemporary Christian"),
    ("spiritual",       "Gospel/Contemporary Christian"),

    # Pop  (broad catch-all — intentionally last among named genres)
    ("pop",             "Pop"),
]


def map_genre(raw_genre) -> str:
    """Map a raw genre string to one of the 15 canonical genres.

    Returns "Unknown" if no keyword matches.
    """
    if pd.isna(raw_genre) or str(raw_genre).strip() == "":
        return "Unknown"
    lowered = str(raw_genre).lower()
    for keyword, canonical in GENRE_MAP:
        if keyword in lowered:
            return canonical
    return "Unknown"

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
    # ---------------------------------------------------------------------------

    print("\n" + "=" * 55)
    print("STEP 2: Deduplicating by track ID")
    print("=" * 55)

    combined["popularity"] = pd.to_numeric(combined.get("popularity", 0), errors="coerce").fillna(0)
    combined = combined.sort_values("popularity", ascending=False)
    combined = combined.drop_duplicates(subset="track_id", keep="first")

    print(f"Unique tracks after track ID dedup: {len(combined):,}")

    # ---------------------------------------------------------------------------
    # Drop rows with no artist name — can't group without one
    # ---------------------------------------------------------------------------

    combined = combined.dropna(subset=["artist_name"])
    combined = combined[combined["artist_name"].str.strip() != ""]
    print(f"Tracks with a valid artist name: {len(combined):,}")

    # ---------------------------------------------------------------------------
    # Normalise artist_name: some datasets store multiple artists as a
    # stringified Python list e.g. "['Drake', 'Future']".  Extract just the
    # first listed artist so every row uses a plain string, then strip quotes
    # and whitespace.  This must happen before any dedup so that
    # "['Drake']" and "Drake" are treated as the same artist.
    # ---------------------------------------------------------------------------

    def extract_primary_artist(val: str) -> str:
        val = str(val).strip()
        if val.startswith("["):
            # Parse the list-like string and take the first element
            import ast
            try:
                parsed = ast.literal_eval(val)
                if isinstance(parsed, list) and parsed:
                    return str(parsed[0]).strip()
            except (ValueError, SyntaxError):
                # Fall back to a simple regex strip if ast fails
                val = val.strip("[]").split(",")[0]
        # Remove any surrounding quotes left over
        return val.strip("'" '"')

    combined["artist_name"] = combined["artist_name"].apply(extract_primary_artist)
    # Drop any rows that became empty after normalisation
    combined = combined[combined["artist_name"] != ""]
    print(f"Tracks after artist name normalisation: {len(combined):,}")

    # ---------------------------------------------------------------------------
    # Second dedup pass: same song title + artist, different track IDs
    # (e.g. single vs album vs region variant).  Normalise both fields to
    # lowercase + stripped whitespace so casing differences don't create false
    # duplicates.  The sort above already put the highest-popularity row first,
    # so keep="first" retains the most popular version.
    # ---------------------------------------------------------------------------

    before = len(combined)
    combined["_artist_norm"] = combined["artist_name"].str.lower().str.strip()
    combined["_title_norm"]  = combined["track_name"].str.lower().str.strip() if "track_name" in combined.columns else ""
    combined = combined.drop_duplicates(subset=["_artist_norm", "_title_norm"], keep="first")
    combined = combined.drop(columns=["_artist_norm", "_title_norm"])
    print(f"Unique tracks after title+artist dedup: {len(combined):,}  "
          f"(removed {before - len(combined):,} same-song duplicates)")

    # ---------------------------------------------------------------------------
    # STEP 3: Remove remastered / live / variant versions
    # We match against track_name.  For each (artist_name, base_title) pair the
    # highest-popularity canonical version already won dedup, so flagged rows are
    # safe to drop.
    # ---------------------------------------------------------------------------

    print("\n" + "=" * 55)
    print("STEP 3: Removing remastered / live / variant tracks")
    print("=" * 55)

    if "track_name" in combined.columns:
        remaster_mask = combined["track_name"].astype(str).str.contains(
            REMASTER_PATTERN, regex=True, na=False
        )
        n_removed = remaster_mask.sum()
        combined = combined[~remaster_mask].reset_index(drop=True)
        print(f"Removed {n_removed:,} remastered / variant tracks")
        print(f"Tracks remaining: {len(combined):,}")
    else:
        print("  ⚠️  'track_name' column not present — skipping remaster filter.")

    # ---------------------------------------------------------------------------
    # STEP 4: Normalise genres → 15 canonical genres
    # ---------------------------------------------------------------------------

    print("\n" + "=" * 55)
    print("STEP 4: Normalising genres to 15 canonical labels")
    print("=" * 55)

    if "genre" in combined.columns:
        combined["genre"] = combined["genre"].apply(map_genre)
        genre_counts = combined["genre"].value_counts()
        for genre, count in genre_counts.items():
            pct = 100 * count / len(combined)
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            print(f"  {genre:<35} {bar}  {count:,} ({pct:.1f}%)")
    else:
        print("  ⚠️  'genre' column not present — skipping genre normalisation.")

    # ---------------------------------------------------------------------------
    # STEP 5: Select the top 1,000 artists by their max track popularity,
    # then take each artist's top 5 songs → 50,000 rows exactly (or fewer if
    # some artists have fewer than 5 tracks in the catalog).
    # ---------------------------------------------------------------------------

    TARGET_TRACKS = TOP_ARTISTS * SONGS_PER_ARTIST   # 50,000

    print("\n" + "=" * 55)
    print(f"STEP 5: Building {TARGET_TRACKS:,}-track dataset")
    print(f"        (target: {TOP_ARTISTS:,} artists × {SONGS_PER_ARTIST} songs, "
          "expanding artist pool if needed)")
    print("=" * 55)

    # ---------------------------------------------------------------------------
    # Pre-compute a global pool of known-genre tracks only, sorted by popularity.
    # We'll draw from this pool repeatedly if the first artist batch falls short.
    # ---------------------------------------------------------------------------

    if "genre" in combined.columns:
        known_pool = combined[combined["genre"] != "Unknown"].copy()
        n_unknown_dropped = len(combined) - len(known_pool)
        print(f"  Dropped {n_unknown_dropped:,} Unknown-genre tracks from selection pool")
    else:
        known_pool = combined.copy()

    known_pool = known_pool.sort_values("popularity", ascending=False)

    # Rank ALL artists by peak popularity so we can expand the list incrementally.
    artist_peak = (
        known_pool.groupby("artist_name")["popularity"]
        .max()
        .reset_index()
        .rename(columns={"popularity": "artist_peak_popularity"})
        .sort_values("artist_peak_popularity", ascending=False)
        .reset_index(drop=True)
    )

    all_artists_ranked = artist_peak["artist_name"].tolist()
    total_available_artists = len(all_artists_ranked)

    # ---------------------------------------------------------------------------
    # Iterative top-up: start with TOP_ARTISTS, keep adding artists in popularity
    # order until we hit TARGET_TRACKS or run out of artists entirely.
    # ---------------------------------------------------------------------------

    batch_size   = TOP_ARTISTS          # grow by this many artists each iteration
    artists_used = 0
    top50k       = pd.DataFrame()

    while len(top50k) < TARGET_TRACKS and artists_used < total_available_artists:
        artists_used = min(artists_used + batch_size, total_available_artists)
        current_artists = all_artists_ranked[:artists_used]

        top50k = (
            known_pool[known_pool["artist_name"].isin(current_artists)]
            .sort_values("popularity", ascending=False)
            .groupby("artist_name", sort=False)
            .head(SONGS_PER_ARTIST)
            .reset_index(drop=True)
        )

        print(f"  Artists in pool: {artists_used:,}  →  tracks collected: {len(top50k):,}")

        if len(top50k) >= TARGET_TRACKS:
            break

        if artists_used >= total_available_artists:
            print(f"  ⚠️  Exhausted all {total_available_artists:,} artists in the catalog.")
            break

    # Trim to exactly TARGET_TRACKS if we overshot (unlikely but possible if
    # the last artist batch pushed us past the target).
    if len(top50k) > TARGET_TRACKS:
        top50k = top50k.head(TARGET_TRACKS).reset_index(drop=True)

    actual_rows    = len(top50k)
    actual_artists = top50k["artist_name"].nunique()
    print(f"\nFinal dataset: {actual_rows:,} tracks across {actual_artists:,} artists")

    if actual_rows < TARGET_TRACKS:
        print(
            f"  ℹ️  Could only collect {actual_rows:,} qualifying tracks from the full catalog "
            f"({TARGET_TRACKS - actual_rows:,} short of the {TARGET_TRACKS:,} target)."
        )

    # ---------------------------------------------------------------------------
    # Popularity summary
    # ---------------------------------------------------------------------------

    pop_min  = top50k["popularity"].min()
    pop_max  = top50k["popularity"].max()
    pop_mean = top50k["popularity"].mean()
    print(f"Popularity range: {pop_min:.0f} – {pop_max:.0f}  (mean: {pop_mean:.1f})")

    # ---------------------------------------------------------------------------
    # Coverage report
    # ---------------------------------------------------------------------------

    print("\n" + "=" * 55)
    print("COVERAGE REPORT")
    print("=" * 55)

    total = len(top50k)
    all_report_cols = METADATA_COLS + AUDIO_FEATURE_COLS
    for col in all_report_cols:
        if col in top50k.columns:
            filled = top50k[col].notna().sum()
            pct = 100 * filled / total
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"  {col:<22} {bar}  {pct:.0f}%")

    if "genre" in top50k.columns:
        print("\nGenre distribution in final dataset:")
        for genre, count in top50k["genre"].value_counts().items():
            pct = 100 * count / total
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            print(f"  {genre:<35} {bar}  {count:,} ({pct:.1f}%)")

    # ---------------------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------------------

    # Reorder columns cleanly
    final_cols = [c for c in ALL_COLS if c in top50k.columns]
    extra_cols = [c for c in top50k.columns if c not in final_cols]
    top50k = top50k[final_cols + extra_cols]

    top50k.to_csv(OUT_CSV, index=False)

    print(f"\n✅ Saved {actual_rows:,} tracks to: {OUT_CSV}")
    print(f"   Rows: {len(top50k):,} | Columns: {len(top50k.columns)}")
    print(f"\nColumns: {list(top50k.columns)}")


if __name__ == "__main__":
    main()
