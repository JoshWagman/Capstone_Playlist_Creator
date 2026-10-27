from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import requests
from datetime import datetime, timezone
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("music_matcher")

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Model'))
from feature_io import save_vector, load_vector_with_meta, VECTOR_PATH

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Backend'))
from functions import build_playlist, build_playlist_from_seeds, detect_seed_language

# -----------------------
# CONFIG
# -----------------------
st.set_page_config(page_title="Music Matcher", page_icon="🎧", layout="wide")

SPOTIFY_CLIENT_ID     = st.secrets["spotify"]["client_id"]
SPOTIFY_CLIENT_SECRET = st.secrets["spotify"]["client_secret"]
SPOTIFY_REDIRECT_URI  = st.secrets["spotify"]["redirect_uri"]

SCOPE = "user-read-private playlist-modify-public playlist-modify-private"
REQUIRED_SCOPES = {"user-read-private", "playlist-modify-public", "playlist-modify-private"}

SPOTIFY_AUTH_URL  = "https://accounts.spotify.com/authorize"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API_BASE  = "https://api.spotify.com/v1"

LOCAL_DB_PATH = "top100k_catalog.csv"

GENRE_MAP = {
    "Pop": "pop", "Hip-Hop": "hip-hop", "Rock": "rock",
    "Electro": "electro", "Jazz": "jazz", "Indie": "indie",
    "R&B": "r&b", "Rap": "rap", "Classical": "classical",
    "Country": "country", "Metal": "metal", "K-Pop": "k-pop",
    "Folk": "folk", "Soul / Funk": "soul", "Dance": "dance",
}

GENRE_ALIASES = {
    "rock": ["rock", "alt-rock", "alternative", "modern_rock", "classic_rock",
             "hard_rock", "punk", "grunge", "garage_rock"],
    "hip-hop": ["hip-hop", "hip_hop", "trap", "melodic_rap", "brooklyn_drill",
                "underground_hip_hop"],
    "rap": ["rap", "melodic_rap", "emo_rap", "trap"],
    "electro": ["electro", "edm", "electro_house", "electronic", "house",
                "trance", "dubstep"],
    "jazz": ["jazz", "blues", "smooth_jazz"],
    "r&b": ["r&b", "r-b", "neo_soul", "indie_soul"],
    "soul": ["soul", "funk", "neo_soul", "indie_soul"],
    "dance": ["dance", "disco", "house", "edm"],
    "folk": ["folk", "indie_folk", "singer-songwriter"],
    "country": ["country", "contemporary_country"],
    "metal": ["metal", "heavy_metal", "death_metal", "metalcore"],
    "classical": ["classical", "ambient", "piano"],
    "k-pop": ["k-pop", "j-pop"],
    "indie": ["indie", "indie-pop", "indie_folk", "indie_soul"],
}

KEY_MAP = {
    0: "C", 1: "C♯/D♭", 2: "D", 3: "D♯/E♭", 4: "E", 5: "F",
    6: "F♯/G♭", 7: "G", 8: "G♯/A♭", 9: "A", 10: "A♯/B♭", 11: "B",
}


# -----------------------
# SESSION STATE INIT
# -----------------------
_defaults = {
    "sp_token_info":     None,
    "sp_granted_scopes": set(),
    "sp_display_name":   None,
    "playlist_data":     None,
    "swiper_songs":      [],
    "current_index":     0,
    "liked_songs":       [],
    "upload_url":        None,
    "swiper_upload_url": None,
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ============================================================
# RAW SPOTIFY AUTH
# ============================================================

def _build_auth_url() -> str:
    return requests.Request(
        "GET", SPOTIFY_AUTH_URL,
        params={
            "client_id":     SPOTIFY_CLIENT_ID,
            "response_type": "code",
            "redirect_uri":  SPOTIFY_REDIRECT_URI,
            "scope":         SCOPE,
            "show_dialog":   "true",
        },
    ).prepare().url


def _exchange_code(code: str) -> dict | None:
    r = requests.post(
        SPOTIFY_TOKEN_URL,
        data={
            "grant_type":   "authorization_code",
            "code":         code,
            "redirect_uri": SPOTIFY_REDIRECT_URI,
        },
        auth=(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET),
    )
    if r.status_code != 200:
        st.error(f"Token exchange failed {r.status_code}: {r.text}")
        return None
    data = r.json()
    data["expires_at"] = (
        datetime.now(timezone.utc).timestamp() + data.get("expires_in", 3600) - 60
    )
    return data


def _refresh_token(refresh_token: str) -> dict | None:
    r = requests.post(
        SPOTIFY_TOKEN_URL,
        data={"grant_type": "refresh_token", "refresh_token": refresh_token},
        auth=(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET),
    )
    if r.status_code != 200:
        return None
    data = r.json()
    data["expires_at"] = (
        datetime.now(timezone.utc).timestamp() + data.get("expires_in", 3600) - 60
    )
    if "refresh_token" not in data:
        data["refresh_token"] = refresh_token
    return data


def _is_expired(token_info: dict) -> bool:
    return datetime.now(timezone.utc).timestamp() >= token_info.get("expires_at", 0)


def _get_access_token() -> str | None:
    info = st.session_state.sp_token_info
    if info is None:
        return None
    if _is_expired(info):
        refreshed = _refresh_token(info["refresh_token"])
        if refreshed is None:
            for k in _defaults:
                st.session_state[k] = _defaults[k]
            st.warning("Your session expired. Please log in again.")
            st.rerun()
        st.session_state.sp_token_info     = refreshed
        st.session_state.sp_granted_scopes = set(refreshed.get("scope", "").split())
        return refreshed["access_token"]
    return info["access_token"]


def _auth_headers() -> dict:
    return {"Authorization": f"Bearer {_get_access_token()}"}


# ============================================================
# RAW SPOTIFY API HELPERS
# ============================================================

def _handle_api_error(r: requests.Response, context: str = "") -> None:
    status = r.status_code
    try:
        body      = r.json()
        sp_msg    = body.get("error", {}).get("message", "No message")
        sp_reason = body.get("error", {}).get("reason", "")
    except Exception:
        sp_msg, sp_reason = r.text, ""

    label = f"[{context}] " if context else ""
    if status == 403:
        st.error(
            f"**{label}403 Forbidden** — `{sp_msg}`"
            + (f" · reason: `{sp_reason}`" if sp_reason else "")
        )
    elif status == 400:
        st.error(f"**{label}400 Bad Request** — `{sp_msg}`")
    elif status == 401:
        st.error(f"**{label}401 Unauthorized** — Please log out and log back in.")
    else:
        st.error(f"**{label}Spotify error {status}** — `{sp_msg}`")


def spotify_get(endpoint: str, params: dict | None = None) -> dict | None:
    token = _get_access_token()
    if not token:
        return None
    r = requests.get(
        f"{SPOTIFY_API_BASE}/{endpoint}",
        headers=_auth_headers(),
        params=params or {},
    )
    if r.status_code == 401:
        for k in _defaults:
            st.session_state[k] = _defaults[k]
        st.warning("Authentication expired. Please log in again.")
        st.rerun()
    if not r.ok:
        return None
    return r.json()


def spotify_post(endpoint: str, payload: dict, context: str = "") -> dict | None:
    token = _get_access_token()
    if not token:
        return None
    r = requests.post(
        f"{SPOTIFY_API_BASE}/{endpoint}",
        headers={**_auth_headers(), "Content-Type": "application/json"},
        json=payload,
    )
    if r.status_code == 401:
        for k in _defaults:
            st.session_state[k] = _defaults[k]
        st.warning("Authentication expired. Please log in again.")
        st.rerun()
    if not r.ok:
        _handle_api_error(r, context=context)
        return None
    return r.json() if r.content else {}


# ============================================================
# TRACK ID RESOLUTION
# ============================================================

def search_track_id(artist: str, title: str) -> str | None:
    r = requests.get(
        f"{SPOTIFY_API_BASE}/search",
        headers=_auth_headers(),
        params={"q": f"track:{title} artist:{artist}", "type": "track", "limit": 1},
    )
    if not r.ok:
        return None
    items = r.json().get("tracks", {}).get("items", [])
    return items[0]["id"] if items else None


def validate_or_resolve_ids(rows: list[dict]) -> tuple[list[str], list[dict]]:
    """
    Validates each stored ID via GET /tracks/{id}.
    Falls back to search if the stored ID is invalid or missing.
    """
    resolved_ids = []
    failed_rows  = []

    for row in rows:
        stored_id = str(row.get("ID", "")).strip()
        found_id  = None

        if stored_id:
            r = requests.get(
                f"{SPOTIFY_API_BASE}/tracks/{stored_id}",
                headers=_auth_headers(),
            )
            if r.ok:
                found_id = stored_id

        if not found_id:
            found_id = search_track_id(
                row.get("Artist", ""),
                row.get("Song Title", ""),
            )

        if found_id:
            resolved_ids.append(found_id)
        else:
            failed_rows.append(row)

    return resolved_ids, failed_rows


# ============================================================
# LOGIN GATE
# ============================================================

def _show_login_wall(login_url: str) -> None:
    _, col_c, _ = st.columns([1, 2, 1])
    with col_c:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.title("🎧 Music Matcher")
        st.markdown(
            "Generate AI-curated playlists from your local music database "
            "and upload them directly to your Spotify account."
        )
        st.divider()
        st.markdown("#### Connect your Spotify account to get started")
        st.link_button(
            "🔗 Log in with Spotify", login_url,
            use_container_width=True, type="primary",
        )
        st.caption("Permissions: view account info, create and modify playlists.")


def run_login_gate() -> None:
    if st.session_state.sp_token_info is not None:
        missing = REQUIRED_SCOPES - st.session_state.sp_granted_scopes
        if missing:
            for k in _defaults:
                st.session_state[k] = _defaults[k]
            st.warning(f"Missing permissions: **{', '.join(missing)}**. Please log in again.")

    code = st.query_params.get("code", None)

    if st.session_state.sp_token_info is None:
        if code:
            token_info = _exchange_code(code)
            if token_info:
                st.session_state.sp_token_info     = token_info
                st.session_state.sp_granted_scopes = set(token_info.get("scope", "").split())
                st.query_params.clear()
                me = spotify_get("me")
                if me:
                    st.session_state.sp_display_name = (
                        me.get("display_name") or me.get("id", "User")
                    )
                st.rerun()
        else:
            _show_login_wall(_build_auth_url())
            st.stop()


run_login_gate()


# -----------------------
# LOCAL DATABASE
# -----------------------
@st.cache_data
def _load_song_catalog() -> pd.DataFrame:
    """Load top500k_catalog.csv and normalise genre / key / mode for the picker."""
    import ast as _ast

    path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'top500k_catalog.csv')
    df = pd.read_csv(path)

    _NOTE_TO_INT = {
        'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
        'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11,
    }
    _MODE_MAP = {'Major': 1, 'Minor': 0}

    def _parse_key(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip()
        if s in _NOTE_TO_INT:
            return _NOTE_TO_INT[s]
        try:
            return int(float(s))
        except (ValueError, TypeError):
            return np.nan

    def _parse_mode(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip()
        if s in _MODE_MAP:
            return _MODE_MAP[s]
        try:
            return int(float(s))
        except (ValueError, TypeError):
            return np.nan

    def _parse_genre(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip()
        if s.startswith('['):
            try:
                lst = _ast.literal_eval(s)
                return str(lst[0]) if lst else np.nan
            except Exception:
                return np.nan
        return s

    df['key'] = df['key'].apply(_parse_key)
    df['mode'] = df['mode'].apply(_parse_mode)
    df['genre'] = df['genre'].apply(_parse_genre)
    df = df.dropna(subset=['genre', 'key', 'mode'])
    df['key'] = df['key'].astype(int)
    df['mode'] = df['mode'].astype(int)
    return df


@st.cache_data
def load_local_database(path: str = LOCAL_DB_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"Database file **'{path}'** not found.")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.strip()

    aliases = {
        "id":                ["track_id", "spotify_id"],
        "name":              ["track_name", "title", "song_title"],
        "artist":            ["artists", "artist_name"],
        "genre":             ["genre_group", "track_genre"],
        "tempo":             ["bpm"],
        "track_popularity":  ["popularity", "track_pop"],
        "artist_popularity": ["artist_pop"],
        "cover_url":         ["album_art", "cover", "image_url"],
        "preview_url":       ["preview"],
        "spotify_url":       ["external_url", "link"],
    }
    for standard, alts in aliases.items():
        if standard not in df.columns:
            for alt in alts:
                if alt in df.columns:
                    df = df.rename(columns={alt: standard})
                    break
    return df


def _similarity_score(row: pd.Series, targets: dict) -> float:
    weights     = {"danceability": 2.0, "energy": 2.0, "valence": 2.0,
                   "tempo": 1.5, "loudness": 1.0, "acousticness": 1.0,
                   "instrumentalness": 0.5, "liveness": 0.5, "speechiness": 0.5}
    normalizers = {"tempo": 150.0, "loudness": 60.0}
    score, total = 0.0, 0.0
    for feat, w in weights.items():
        val = row.get(feat)
        if pd.isna(val) or feat not in targets:
            continue
        diff = abs(float(val) - float(targets[feat])) / normalizers.get(feat, 1.0)
        score += w * (1.0 - min(diff, 1.0))
        total += w
    return score / total if total else 0.0


def get_filtered_tracks_local(
    genre_group: str,
    target_danceability: float, target_energy: float, target_key: int,
    target_loudness: float, target_mode: int, target_speechiness: float,
    target_acousticness: float, target_instrumentalness: float,
    target_liveness: float, target_valence: float, target_tempo: float,
    min_artist_pop: int, min_track_pop: int,
    limit: int = 12,
) -> list[dict]:
    df = load_local_database()
    if df.empty:
        return []

    seed = GENRE_MAP.get(genre_group, "pop").lower()
    if "genre" in df.columns:
        mask = (
            df["genre"].str.lower().str.contains(seed, na=False)
            | df["genre"].str.lower().str.contains(genre_group.lower(), na=False)
        )
        filtered = df[mask].copy() if mask.any() else df.copy()
    else:
        filtered = df.copy()

    if "track_popularity" in filtered.columns:
        filtered = filtered[filtered["track_popularity"].fillna(0) >= min_track_pop]
    if "artist_popularity" in filtered.columns:
        filtered = filtered[filtered["artist_popularity"].fillna(0) >= min_artist_pop]

    if filtered.empty:
        return []

    targets = dict(
        danceability=target_danceability, energy=target_energy,
        valence=target_valence, loudness=target_loudness,
        acousticness=target_acousticness, instrumentalness=target_instrumentalness,
        liveness=target_liveness, speechiness=target_speechiness, tempo=target_tempo,
    )
    filtered["_score"] = filtered.apply(_similarity_score, targets=targets, axis=1)
    top = filtered.nlargest(limit, "_score")

    result = []
    for _, row in top.iterrows():
        key_val  = int(row.get("key",  0)) if pd.notna(row.get("key"))  else 0
        mode_val = int(row.get("mode", 1)) if pd.notna(row.get("mode")) else 1
        bpm_raw  = row.get("tempo", 0)
        result.append({
            "ID":         str(row.get("id", "")),
            "Artist":     str(row.get("artist", "Unknown")),
            "Song Title": str(row.get("name",   "Unknown")),
            "BPM":        int(round(float(bpm_raw))) if pd.notna(bpm_raw) else 0,
            "Key":        f"{KEY_MAP.get(key_val, key_val)} {'Major' if mode_val else 'Minor'}",
            "Pop (T)":    int(row["track_popularity"])  if pd.notna(row.get("track_popularity"))  else 0,
            "Pop (A)":    int(row["artist_popularity"]) if pd.notna(row.get("artist_popularity")) else 0,
            "Preview":    row.get("preview_url") if pd.notna(row.get("preview_url")) else None,
            "Cover":      row.get("cover_url")   if pd.notna(row.get("cover_url"))   else None,
            "Link":       row.get("spotify_url") if pd.notna(row.get("spotify_url")) else None,
            "Similarity": round(row["_score"], 3),
        })
    return result


# ============================================================
# SPOTIFY PLAYLIST UPLOAD
# ============================================================

def upload_playlist_to_spotify(
    rows: list[dict],
    playlist_name: str | None = None,
    playlist_description: str = "",
    public: bool = False,
) -> tuple[str | None, int, int]:
    """Returns (playlist_url, n_added, n_failed)."""
    if not rows:
        return None, 0, 0

    if playlist_name is None:
        playlist_name = f"Music Matcher – {datetime.now().strftime('%b %d, %Y %H:%M')}"

    # Step 1 — Resolve IDs
    with st.status("🔍 Validating track IDs via Spotify…", expanded=True) as status:
        st.write(f"Checking {len(rows)} tracks individually…")
        resolved_ids, failed_rows = validate_or_resolve_ids(rows)

        if failed_rows:
            st.write(f"⚠️ Could not resolve **{len(failed_rows)}** track(s):")
            for row in failed_rows:
                st.write(f"  - {row.get('Artist')} — {row.get('Song Title')}")

        st.write(f"✅ Resolved **{len(resolved_ids)}** valid track IDs.")
        status.update(label="Track IDs resolved.", state="complete")

    if not resolved_ids:
        st.error("No tracks could be resolved. Check your CSV artist and song names.")
        return None, 0, len(rows)

    # Step 2 — Create playlist via POST /me/playlists
    playlist = spotify_post(
        "me/playlists",
        payload={
            "name":        playlist_name,
            "public":      public,
            "description": playlist_description,
        },
        context="Create Playlist",
    )
    if playlist is None:
        return None, 0, len(rows)

    playlist_id  = playlist.get("id")
    playlist_url = playlist.get("external_urls", {}).get("spotify")

    if not playlist_id:
        st.error("Spotify returned a playlist with no ID.")
        return None, 0, len(rows)

    uris     = [f"spotify:track:{tid}" for tid in resolved_ids]
    n_added  = 0
    n_failed = len(failed_rows)

    for i in range(0, len(uris), 100):
        batch  = uris[i : i + 100]
        result = spotify_post(
            f"playlists/{playlist_id}/items",  # ← /tracks → /items
            payload={"uris": batch},
            context=f"Add Items {i + 1}–{i + len(batch)}",
        )
        if result is not None:
            n_added += len(batch)
        else:
            n_failed += len(batch)

    return playlist_url, n_added, n_failed


# ============================================================
# DEBUG EXPANDER
# ============================================================

def _render_debug_expander(rows: list[dict] | None = None) -> None:
    with st.expander("🔍 Debug — Token & Track IDs", expanded=False):
        info = st.session_state.sp_token_info or {}

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Granted scopes:**")
            for s in sorted(st.session_state.sp_granted_scopes):
                st.markdown(f"{'✅' if s in REQUIRED_SCOPES else '⚠️'} `{s}`")
        with col2:
            expires_ts = info.get("expires_at", 0)
            st.write("**Token expires at:**",
                     datetime.fromtimestamp(expires_ts) if expires_ts else "N/A")
            st.write("**Has refresh token:**", bool(info.get("refresh_token")))

        missing = REQUIRED_SCOPES - st.session_state.sp_granted_scopes
        if missing:
            st.error(f"Missing: `{', '.join(missing)}`. Log out and back in.")

        if rows:
            st.write("**Sample track IDs (first 5):**", [r.get("ID", "") for r in rows[:5]])

            if st.button("🧪 Validate first ID individually", key="debug_validate"):
                row   = rows[0]
                tid   = str(row.get("ID", "")).strip()
                token = _get_access_token()
                r     = requests.get(
                    f"{SPOTIFY_API_BASE}/tracks/{tid}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                st.write(f"**Status:** `{r.status_code}`")
                if r.ok:
                    d = r.json()
                    st.write(f"✅ Found: **{d['name']}** — {d['artists'][0]['name']}")
                else:
                    st.write("❌ Invalid — trying search fallback…")
                    found = search_track_id(row.get("Artist", ""), row.get("Song Title", ""))
                    st.success(f"Search found: `{found}`") if found else st.error("Not found.")

            if st.button("🧪 Full dry-run upload (1 track)", key="debug_dry_run"):
                token = _get_access_token()
                pl_r  = requests.post(
                    f"{SPOTIFY_API_BASE}/me/playlists",
                    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                    json={"name": "Debug Test – delete me", "public": False, "description": ""},
                )
                st.write(f"**Create playlist:** `{pl_r.status_code}`")
                if pl_r.ok:
                    pl_id = pl_r.json().get("id")
                    row   = rows[0]
                    tid   = str(row.get("ID", "")).strip() or search_track_id(
                        row.get("Artist", ""), row.get("Song Title", "")
                    )
                    tr_r = requests.post(
                        f"{SPOTIFY_API_BASE}/playlists/{pl_id}/items",  # ← /tracks → /items
                        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                        json={"uris": [f"spotify:track:{tid}"]},
                    )
                    st.write(f"**Add item (`/items`):** `{tr_r.status_code}`")
                    if tr_r.ok:
                        st.success("✅ Full upload flow works! Delete the test playlist from Spotify.")
                    else:
                        st.json(tr_r.json())
                else:
                    st.json(pl_r.json())


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("Navigation")

if st.session_state.sp_display_name:
    st.sidebar.markdown(f"👤 **{st.session_state.sp_display_name}**")

if st.sidebar.button("🔓 Log out"):
    for k in _defaults:
        st.session_state[k] = _defaults[k]
    st.rerun()

st.sidebar.divider()
page = st.sidebar.radio("Go to", ["🎛️ Playlist Generator", "🔥 Song Swiper"])


# ==========================================
# PAGE 1: PLAYLIST GENERATOR
# ==========================================
if page == "🎛️ Playlist Generator":
    catalog = _load_song_catalog()
    log.info("Song catalog loaded: %d rows", len(catalog))

    FEATURE_COLS = [
        "danceability", "energy", "key", "loudness", "mode",
        "speechiness", "acousticness", "instrumentalness",
        "liveness", "valence", "tempo", "popularity",
    ]

    # -- Sidebar Controls --
    st.sidebar.header("Settings")
    genre_groups = st.sidebar.multiselect(
        "Genre Groups",
        options=list(GENRE_MAP.keys()),
        default=["Pop"],
        placeholder="Pick one or more genres…",
    )

    with st.sidebar.expander("🎹 Adjustments", expanded=True):
        target_bpm = st.slider("Tempo (BPM)", 50, 200, 120)
        playlist_size = st.slider("Playlist Size", 5, 50, 20)

    st.sidebar.divider()
    st.sidebar.markdown("### 💾 Feature Vector")
    meta = load_vector_with_meta()
    if meta:
        st.sidebar.caption(f"Last saved: `{meta.get('saved_at', '—')}`")
        with st.sidebar.expander("📄 View saved vector"):
            for k, v in meta["vector"].items():
                st.sidebar.write(f"**{k}**: `{v}`")
    else:
        st.sidebar.caption("No vector saved yet. Generate a playlist to save one.")

    # -- Main Content --
    st.title("🎵 AI Playlist Generator")
    st.markdown(
        "Pick **1–5 songs** you enjoy and we'll build a playlist that "
        "matches your taste using our latent-space model."
    )

    # Filter catalog by selected genres for the song picker
    if genre_groups and "genre" in catalog.columns:
        alias_keys = set()
        for g in genre_groups:
            base = GENRE_MAP.get(g, "").lower()
            alias_keys.update(GENRE_ALIASES.get(base, [base]))
        mask = catalog["genre"].str.lower().apply(
            lambda v: any(k in v for k in alias_keys) if pd.notna(v) else False
        )
        genre_catalog = catalog[mask].copy() if mask.any() else catalog.copy()
    else:
        genre_catalog = catalog.copy()

    genre_label = ", ".join(genre_groups) if genre_groups else "All"
    log.info("Genre filter %s -> %d songs in picker", genre_label, len(genre_catalog))

    genre_catalog = genre_catalog.sort_values("popularity", ascending=False, na_position="last")

    genre_catalog["_label"] = (
        genre_catalog["artist_name"].fillna("Unknown")
        + " — "
        + genre_catalog["track_name"].fillna("Unknown")
    )
    label_to_idx = dict(zip(genre_catalog["_label"], genre_catalog.index))

    selected_labels = st.multiselect(
        f"🔍 Search songs  ·  {len(genre_catalog):,} {genre_label} tracks available",
        options=genre_catalog["_label"].tolist(),
        max_selections=5,
        placeholder="Type an artist or song name…",
    )

    if selected_labels:
        seed_indices = [label_to_idx[l] for l in selected_labels if l in label_to_idx]
        seed_df = catalog.loc[seed_indices]
        with st.expander(f"Selected {len(seed_df)} seed song(s)", expanded=False):
            st.dataframe(
                seed_df[["artist_name", "track_name"]].rename(
                    columns={"artist_name": "Artist", "track_name": "Song"}
                ),
                use_container_width=True,
                hide_index=True,
            )

    if st.button("✨ Generate Playlist", type="primary"):
        if not selected_labels:
            st.error("Please select at least 1 seed song.")
        else:
            with st.spinner("Mapping preferences to latent space…"):
                seed_indices = [label_to_idx[l] for l in selected_labels if l in label_to_idx]
                seed_df = catalog.loc[seed_indices]
                log.info("Seed songs selected: %s", selected_labels)

                seed_specs_list = []
                for _, row in seed_df.iterrows():
                    specs = row[FEATURE_COLS].to_dict()
                    specs["tempo"] = float(target_bpm)
                    seed_specs_list.append(specs)

                try:
                    centroid = seed_df[FEATURE_COLS].mean().to_dict()
                    centroid["tempo"] = float(target_bpm)
                    save_vector(centroid)
                    log.info("Feature vector (centroid) saved to disk")
                except Exception as e:
                    log.warning("Could not save feature vector: %s", e)
                    st.warning(f"Could not save feature vector: {e}")

                seed_genres = seed_df["genre"].dropna().unique().tolist() if "genre" in seed_df.columns else None

                seed_language = detect_seed_language(seed_df["track_name"].dropna())
                log.info("Detected seed language: %s", seed_language)

                log.info("Calling build_playlist_from_seeds(%d seeds, k=%d, seed_genres=%s, lang=%s)…", len(seed_specs_list), playlist_size, seed_genres, seed_language)
                raw_results = build_playlist_from_seeds(seed_specs_list, k=playlist_size, seed_genres=seed_genres, seed_language=seed_language)
                log.info("build_playlist_from_seeds returned %d raw results", len(raw_results))

                display_data = []
                for rec in raw_results:
                    display_data.append({
                        "ID": str(rec.get("track_id", "")),
                        "Artist": str(rec.get("artist_name", "Unknown")),
                        "Song Title": str(rec.get("track_name", "Unknown")),
                        "Genre": str(rec.get("genre", "")),
                        "BPM": int(round(float(rec.get("tempo", 0)))),
                    })

                st.session_state.playlist_data = pd.DataFrame(display_data)
                st.session_state.upload_url = None

                if not display_data:
                    log.error("No matching songs found")
                    st.error("No matching songs found. Try different seed songs.")
                else:
                    log.info("Playlist built: %d tracks", len(display_data))
                    for i, d in enumerate(display_data[:5], 1):
                        log.info("  #%d  %s — %s", i, d["Artist"], d["Song Title"])
                    if len(display_data) > 5:
                        log.info("  … and %d more", len(display_data) - 5)
                    st.success(f"Built a playlist with **{len(display_data)}** tracks!")

    if st.session_state.playlist_data is not None and not st.session_state.playlist_data.empty:
        st.divider()
        st.subheader("Your Curated Mix")

        st.dataframe(
            st.session_state.playlist_data,
            use_container_width=True,
            hide_index=True,
            column_config={
                "BPM": st.column_config.NumberColumn("BPM", format="%d"),
            },
        )

        st.divider()
        st.subheader("📤 Upload to Spotify")

        col_nm, col_vis = st.columns([3, 1])
        with col_nm:
            pl_name = st.text_input(
                "Playlist Name",
                value=f"Music Matcher – {datetime.now().strftime('%b %d, %Y')}",
            )
        with col_vis:
            pl_public = st.toggle("Public", value=False)

        pl_desc = st.text_input(
            "Description (optional)",
            value=f"Generated by Music Matcher | Genre: {genre_label}",
        )

        rows = st.session_state.playlist_data.to_dict("records")
        _render_debug_expander(rows=rows)

        if st.button("🚀 Upload Playlist to Spotify", type="primary"):
            url, n_added, n_failed = upload_playlist_to_spotify(
                rows, pl_name, pl_desc, pl_public
            )
            if url:
                st.session_state.upload_url = url
                if n_failed == 0:
                    st.success(f"✅ Playlist created with all **{n_added}** tracks!")
                else:
                    st.warning(
                        f"Playlist created: **{n_added}** tracks added, "
                        f"**{n_failed}** could not be resolved or added."
                    )

        if st.session_state.upload_url:
            st.link_button("🎧 Open Playlist on Spotify", st.session_state.upload_url)


# ==========================================
# PAGE 2: SONG SWIPER
# ==========================================
elif page == "🔥 Song Swiper":
    st.title("🔥 Song Swiper")
    st.caption("Swipe through local tracks. Like them, then upload to Spotify.")

    with st.expander("Swiper Settings"):
        swiper_genre = st.selectbox("Genre", list(GENRE_MAP.keys()), index=0)
        swiper_bpm   = st.slider("Target BPM", 60, 180, 120)

        if st.button("🔄 Reload Stack"):
            st.session_state.swiper_songs = get_filtered_tracks_local(
                genre_group=swiper_genre,
                target_danceability=0.6, target_energy=0.6, target_key=5,
                target_loudness=-8.0, target_mode=1, target_speechiness=0.1,
                target_acousticness=0.2, target_instrumentalness=0.0,
                target_liveness=0.2, target_valence=0.5,
                target_tempo=swiper_bpm, min_artist_pop=30, min_track_pop=30, limit=20,
            )
            st.session_state.current_index = 0
            st.rerun()

    if not st.session_state.swiper_songs:
        st.session_state.swiper_songs = get_filtered_tracks_local(
            "Pop", 0.6, 0.6, 0, -10.0, 1, 0.05, 0.1, 0.0, 0.1, 0.5, 120, 40, 40, 10,
        )

    songs = st.session_state.swiper_songs
    idx   = st.session_state.current_index

    if songs and idx < len(songs):
        song = songs[idx]
        with st.container(border=True):
            _, c2, _ = st.columns([1, 2, 1])
            with c2:
                if song.get("Cover"):
                    st.image(song["Cover"], width=300)
                st.subheader(song["Song Title"])
                st.write(f"**{song['Artist']}**")
                st.caption(f"Key: {song['Key']} • BPM: {song['BPM']}")
                if song.get("Preview"):
                    st.audio(song["Preview"], format="audio/mp3")
                else:
                    st.warning("No preview available")

        col_pass, _, col_like = st.columns([1, 2, 1])
        with col_pass:
            if st.button("❌ Pass", use_container_width=True):
                st.session_state.current_index += 1
                st.rerun()
        with col_like:
            if st.button("💚 Like", type="primary", use_container_width=True):
                st.session_state.liked_songs.append(song)
                st.session_state.current_index += 1
                st.rerun()
    else:
        st.success("No more songs!")
        if st.button("Restart"):
            st.session_state.current_index = 0
            st.rerun()

    if st.session_state.liked_songs:
        st.divider()
        st.subheader("💚 Liked Songs")
        st.dataframe(
            pd.DataFrame(st.session_state.liked_songs).drop(
                columns=["Similarity"], errors="ignore"
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.divider()
        st.subheader("📤 Upload Liked Songs to Spotify")

        col_nm2, col_vis2 = st.columns([3, 1])
        with col_nm2:
            swiper_pl_name = st.text_input(
                "Playlist Name",
                value=f"Swiper Likes – {datetime.now().strftime('%b %d, %Y')}",
                key="swiper_pl_name",
            )
        with col_vis2:
            swiper_public = st.toggle("Public", value=False, key="swiper_public")

        liked_rows = st.session_state.liked_songs
        _render_debug_expander(rows=liked_rows)

        col_upload, col_clear = st.columns(2)
        with col_upload:
            if st.button("🚀 Upload to Spotify", type="primary", key="swiper_upload"):
                url, n_added, n_failed = upload_playlist_to_spotify(
                    liked_rows, swiper_pl_name, public=swiper_public
                )
                if url:
                    st.session_state.swiper_upload_url = url
                    if n_failed == 0:
                        st.success(f"✅ Playlist created with all **{n_added}** tracks!")
                    else:
                        st.warning(
                            f"Playlist created: **{n_added}** added, "
                            f"**{n_failed}** could not be resolved."
                        )

        with col_clear:
            if st.button("🗑️ Clear Liked Songs", key="swiper_clear"):
                st.session_state.liked_songs       = []
                st.session_state.swiper_upload_url = None
                st.rerun()

        if st.session_state.swiper_upload_url:
            st.link_button("🎧 Open Playlist on Spotify", st.session_state.swiper_upload_url)
