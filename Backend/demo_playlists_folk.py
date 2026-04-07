"""Generate 30 two-seed folk/country playlist candidates for demo selection."""
from __future__ import annotations
import sys, os, warnings, itertools
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from functions import _load_checkpoint, _get_songs_df, build_playlist_from_seeds

FEATURE_COLS = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo", "popularity",
]

_load_checkpoint()
df = _get_songs_df()


def find_seed(track, artist_sub):
    m = df[
        df["track_name"].str.contains(track, na=False, regex=False)
        & df["artist_name"].str.contains(artist_sub, na=False, regex=False)
    ].sort_values("popularity", ascending=False).head(1)
    return m if not m.empty else None


def clean(s):
    return str(s).replace("['", "").replace("']", "").replace("'", "")


SEEDS = [
    ("Atlantis", "Seafret"),
    ("ceilings", "Lizzy McAlpine"),
    ("The Night We Met", "Lord Huron"),
    ("Riptide", "Vance Joy"),
    ("Ho Hey", "Lumineers"),
    ("Little Talks", "Of Monsters and Men"),
    ("Green Green Grass", "George Ezra"),
    ("Something in the Orange", "Zach Bryan"),
    ("Chicken Fried", "Zac Brown"),
    ("Take Me Home, Country Roads", "John Denver"),
    ("Wagon Wheel", "Darius Rucker"),
    ("Fast Car", "Tracy Chapman"),
    ("Jolene", "Dolly Parton"),
    ("Need You Now", "Lady A"),
    ("Budapest", "George Ezra"),
    ("Stubborn Love", "Lumineers"),
    ("exile", "Taylor Swift"),
    ("Someone You Loved", "Lewis Capaldi"),
]

seed_cache = {}
for track, artist_sub in SEEDS:
    s = find_seed(track, artist_sub)
    if s is not None:
        key = f"{clean(s.iloc[0]['artist_name'])} - {s.iloc[0]['track_name']}"
        seed_cache[(track, artist_sub)] = (key, s)

print(f"Found {len(seed_cache)}/{len(SEEDS)} seeds\n")

valid_keys = list(seed_cache.keys())
all_pairs = list(itertools.combinations(valid_keys, 2))

GENRE_HITS = ["folk", "country", "indie_folk", "contemporary_country", "singer-songwriter"]

results = []
for k1, k2 in all_pairs:
    label1, s1 = seed_cache[k1]
    label2, s2 = seed_cache[k2]
    seeds = pd.concat([s1, s2])
    seed_specs = [row[FEATURE_COLS].to_dict() for _, row in seeds.iterrows()]
    seed_genres = seeds["genre"].dropna().unique().tolist()
    songs = build_playlist_from_seeds(seed_specs, k=10, seed_genres=seed_genres, seed_language="en")

    genres = [str(s.get("genre", "")).lower() for s in songs]
    match_ct = sum(1 for g in genres if any(x in g for x in GENRE_HITS))
    pops = [s.get("popularity", 0) for s in songs]
    avg_pop = sum(pops) / len(pops)
    min_pop = min(pops)

    results.append({
        "label1": label1, "label2": label2,
        "match_pct": match_ct * 10, "avg_pop": avg_pop, "min_pop": min_pop,
        "songs": songs,
    })

results.sort(key=lambda r: (r["match_pct"], r["avg_pop"]), reverse=True)

for idx, r in enumerate(results[:30], 1):
    print("=" * 95)
    print(f"#{idx:>2}  Seeds: {r['label1']}  +  {r['label2']}")
    print(f"     Folk/Country: {r['match_pct']}%  |  Avg Pop: {r['avg_pop']:.1f}  |  Min Pop: {r['min_pop']:.0f}")
    print("-" * 95)
    for i, s in enumerate(r["songs"], 1):
        a = clean(s.get("artist_name", "?"))[:30]
        t = str(s.get("track_name", "?"))[:38]
        g = str(s.get("genre", "?"))[:18]
        p = s.get("popularity", 0)
        print(f"  {i:>2}. {a:<30} {t:<38} {g:<18} {p:>3.0f}")
    print()
