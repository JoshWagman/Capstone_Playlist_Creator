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


# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Music Matcher",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── GLOBAL CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Chrome ───────────────────────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.8rem; padding-bottom: 2rem; max-width: 1100px; }
html, body, [class*="css"] {
    font-family: "Inter", -apple-system, BlinkMacSystemFont, sans-serif;
}
/* Slightly darker bg so white cards have clear separation */
.stApp { background: #E8EAF2; }

/* ── Sidebar ──────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0d1520;
    border-right: 1px solid #1e2d3d;
}
/* Green accent bar pinned to the top of the sidebar */
[data-testid="stSidebar"] > div:first-child {
    border-top: 3px solid #1DB954;
}
/* FIX: was #6b7280 on dark — only ~3.8:1. Now #c9d1dc gives ~8:1 */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div  { color: #c9d1dc !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3  { color: #f0f4f8 !important; }
[data-testid="stSidebar"] hr  { border-color: #263040; }

/* Sidebar nav: pill hover + better padding */
[data-testid="stSidebar"] .stRadio > div { gap: 4px; }
[data-testid="stSidebar"] .stRadio label {
    border-radius: 8px !important;
    padding: 9px 12px !important;
    transition: background 0.15s !important;
    font-weight: 500 !important;
    color: #c9d1dc !important;
}
[data-testid="stSidebar"] .stRadio label:hover { background: #1e2d3d !important; }

[data-testid="stSidebar"] .stButton > button {
    background: #1a2738;
    color: #c9d1dc !important;
    border: 1px solid #2d3f55;
    border-radius: 10px;
    width: 100%;
    transition: background 0.2s;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #1DB954 !important;
    border-color: #1DB954 !important;
    color: white !important;
}

/* ── Primary button ───────────────────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1DB954 0%, #158a3e 100%);
    color: white !important;
    border: none;
    border-radius: 50px;
    font-weight: 600;
    font-size: 0.95rem;
    padding: 0.55rem 1.8rem;
    box-shadow: 0 4px 14px rgba(29,185,84,0.28);
    transition: box-shadow 0.2s, transform 0.15s;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 22px rgba(29,185,84,0.42);
    transform: translateY(-1px);
}

/* ── Secondary / default button ───────────────────────────────────────── */
.stButton > button[kind="secondary"] {
    border-radius: 50px;
    font-weight: 500;
    border: 2px solid #d1d5db;
    background: white;
    color: #374151 !important;
    transition: border-color 0.2s, color 0.2s;
}
.stButton > button[kind="secondary"]:hover {
    border-color: #1DB954;
    color: #15803d !important;
}

/* ── Link button ──────────────────────────────────────────────────────── */
[data-testid="stLinkButton"] a {
    background: #1DB954 !important;
    color: white !important;
    border-radius: 50px !important;
    font-weight: 600 !important;
    border: none !important;
    box-shadow: 0 4px 14px rgba(29,185,84,0.28) !important;
}

/* ── Bordered containers ──────────────────────────────────────────────── */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 16px !important;
    border: 1px solid #dde0ea !important;
    background: white !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06) !important;
}

/* ── Metrics ──────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: white;
    border-radius: 12px;
    padding: 14px 18px;
    border: 1px solid #dde0ea;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05);
}
[data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    font-weight: 700;
    color: #111827 !important;
}
/* FIX: was #6b7280 on white (~4.8:1 ok) but label also renders on #E8EAF2 page bg (~4.2:1 fail).
   #4b5563 gives ~7.2:1 on white and ~6.5:1 on page bg — comfortable pass */
[data-testid="stMetricLabel"] {
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    color: #4b5563 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Dataframe ────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    border: 1px solid #dde0ea;
    overflow: hidden;
    box-shadow: 0 1px 6px rgba(0,0,0,0.04);
}

/* ── Input / select fields ────────────────────────────────────────────── */
[data-baseweb="input"],
div[data-baseweb="select"] > div:first-child { background: white !important; }
[data-baseweb="input"] input { color: #111827 !important; }

/* ── Expander ─────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border-radius: 12px !important;
    border: 1px solid #dde0ea !important;
    background: white !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.04) !important;
    
}
/* ── Sidebar expander content (white bg) ──────────────────────────────── */
/* FIX: sidebar blanket color:#c9d1dc bleeds into white expander bg → ~1.6:1 FAIL.
   Scope dark text only to expander content. */
[data-testid="stSidebar"] [data-testid="stExpander"] p,
[data-testid="stSidebar"] [data-testid="stExpander"] span,
[data-testid="stSidebar"] [data-testid="stExpander"] div,
[data-testid="stSidebar"] [data-testid="stExpander"] label { color: #111827 !important; }

/* Inline code blocks inside the expander (the backtick values) */
[data-testid="stSidebar"] [data-testid="stExpander"] code {
    color: #1e3a5f !important;       /* dark blue on white: ~9:1 */
    background: #f0f4ff !important;  /* soft blue tint */
    border-radius: 4px;
    padding: 1px 5px;
}

/* Bold key labels */
[data-testid="stSidebar"] [data-testid="stExpander"] strong { color: #111827 !important; }

/* Expander summary chevron + label — keep light so it reads on dark sidebar bg */
[data-testid="stSidebar"] [data-testid="stExpander"] summary,
[data-testid="stSidebar"] [data-testid="stExpander"] summary p,
[data-testid="stSidebar"] [data-testid="stExpander"] summary span {
    color: #111827 !important;
}

/* ── Logout button — brighter default text ────────────────────────────── */
/* FIX: #c9d1dc on #1a2738 is ~6:1 (passes), but visually recedes.
   Bump to #f0f4f8 (~12:1) so it reads as a clear action, not a disabled item. */
[data-testid="stSidebar"] .stButton > button {
    background: #1a2738;
    color: #f0f4f8 !important;        /* was #c9d1dc */
    border: 1px solid #2d3f55;
    border-radius: 10px;
    width: 100%;
    font-weight: 500;
    transition: background 0.2s, border-color 0.2s;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #1DB954 !important;
    border-color: #1DB954 !important;
    color: white !important;
}
/* FIX: expander summary label was inheriting a faint color */
[data-testid="stExpander"] details summary p { color: #374151 !important; font-weight: 600 !important; }

/* ── Multiselect tags ─────────────────────────────────────────────────── */
[data-baseweb="tag"] { background: #dcfce7 !important; border-radius: 50px !important; }
[data-baseweb="tag"] span { color: #15803d !important; font-weight: 600; }

/* ── Progress bar ─────────────────────────────────────────────────────── */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #1DB954, #17a44a);
    border-radius: 50px;
}
/* FIX: progress bar caption text */
[data-testid="stProgressBar"] p { color: #374151 !important; font-size: 0.82rem !important; }

/* ── Alerts ───────────────────────────────────────────────────────────── */
[data-testid="stAlert"] { border-radius: 12px !important; }

/* ── Status widget ────────────────────────────────────────────────────── */
[data-testid="stStatusWidget"] { border-radius: 12px !important; }

/* ── Caption ──────────────────────────────────────────────────────────── */
/* FIX: captions on #E8EAF2 bg — #6b7280 is borderline; nudge to #4b5563 */
[data-testid="stCaptionContainer"] p { color: #4b5563 !important; }

/* ── Divider ──────────────────────────────────────────────────────────── */
hr { border-color: #dde0ea; margin: 1.5rem 0; }

/* ── Audio player ─────────────────────────────────────────────────────── */
audio { border-radius: 8px; width: 100%; }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
SPOTIFY_CLIENT_ID     = st.secrets["spotify"]["client_id"]
SPOTIFY_CLIENT_SECRET = st.secrets["spotify"]["client_secret"]
SPOTIFY_REDIRECT_URI  = st.secrets["spotify"]["redirect_uri"]

SCOPE = "user-read-private playlist-modify-public playlist-modify-private"
REQUIRED_SCOPES = {"user-read-private", "playlist-modify-public", "playlist-modify-private"}

SPOTIFY_AUTH_URL  = "https://accounts.spotify.com/authorize"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API_BASE  = "https://api.spotify.com/v1"

LOCAL_DB_PATH = "songs_database.csv"

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

# ─── SESSION STATE ────────────────────────────────────────────────────────────
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


# ─── AUTH ──────────────────────────────────────────────────────────────────────
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


# ─── SPOTIFY API HELPERS ───────────────────────────────────────────────────────
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
        st.error(f"**{label}403 Forbidden** — `{sp_msg}`" + (f" · `{sp_reason}`" if sp_reason else ""))
    elif status == 400:
        st.error(f"**{label}400 Bad Request** — `{sp_msg}`")
    elif status == 401:
        st.error(f"**{label}401 Unauthorized** — Please log out and back in.")
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
    return r.json() if r.ok else None


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


# ─── TRACK ID RESOLUTION ──────────────────────────────────────────────────────
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
    resolved_ids, failed_rows = [], []
    for row in rows:
        stored_id = str(row.get("ID", "")).strip().replace("spotify:track:", "")
        found_id  = None
        if stored_id and stored_id.lower() not in ("nan", "none", ""):
            r = requests.get(f"{SPOTIFY_API_BASE}/tracks/{stored_id}", headers=_auth_headers())
            if r.ok:
                found_id = stored_id
        if not found_id:
            found_id = search_track_id(row.get("Artist", ""), row.get("Song Title", ""))
        if found_id:
            resolved_ids.append(found_id)
        else:
            failed_rows.append(row)
    return resolved_ids, failed_rows


# ─── LOGIN GATE ────────────────────────────────────────────────────────────────
def _show_login_wall(login_url: str) -> None:
    _, col_c, _ = st.columns([1, 2, 1])
    with col_c:
        st.markdown("""
        <div style="background:white;border-radius:24px;padding:52px 44px;
                    box-shadow:0 8px 48px rgba(0,0,0,0.10);text-align:center;margin-top:60px;">
            <div style="font-size:3.5rem;margin-bottom:14px;">🎧</div>
            <h1 style="font-size:2.1rem;font-weight:800;color:#111827;margin:0 0 10px;">
                Music Matcher
            </h1>
            <!-- FIX: #6b7280 on white is 4.48:1 — passes, but #4b5563 is more comfortable -->
            <p style="color:#4b5563;font-size:0.95rem;line-height:1.75;margin:0 0 28px;">
                Generate AI-curated playlists from your personal taste profile<br>
                and publish them directly to Spotify.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='margin-top:12px;'>", unsafe_allow_html=True)
        st.link_button(
            "🔗  Connect with Spotify",
            login_url,
            use_container_width=True,
            type="primary",
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.caption("Permissions requested: view account info · create & modify playlists")


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


# ─── LOCAL DATABASE ────────────────────────────────────────────────────────────
@st.cache_data
def _load_song_catalog() -> pd.DataFrame:
    import ast as _ast
    path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'top500k_catalog.csv')
    df = pd.read_csv(path)

    _NOTE_TO_INT = {'C':0,'C#':1,'D':2,'D#':3,'E':4,'F':5,'F#':6,'G':7,'G#':8,'A':9,'A#':10,'B':11}
    _MODE_MAP    = {'Major': 1, 'Minor': 0}

    def _parse_key(v):
        if pd.isna(v): return np.nan
        s = str(v).strip()
        if s in _NOTE_TO_INT: return _NOTE_TO_INT[s]
        try: return int(float(s))
        except: return np.nan

    def _parse_mode(v):
        if pd.isna(v): return np.nan
        s = str(v).strip()
        if s in _MODE_MAP: return _MODE_MAP[s]
        try: return int(float(s))
        except: return np.nan

    def _parse_genre(v):
        if pd.isna(v): return np.nan
        s = str(v).strip()
        if s.startswith('['):
            try:
                lst = _ast.literal_eval(s)
                return str(lst[0]) if lst else np.nan
            except: return np.nan
        return s

    df['key']   = df['key'].apply(_parse_key)
    df['mode']  = df['mode'].apply(_parse_mode)
    df['genre'] = df['genre'].apply(_parse_genre)
    df = df.dropna(subset=['genre', 'key', 'mode'])
    df['key']  = df['key'].astype(int)
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
    weights     = {"danceability":2.0,"energy":2.0,"valence":2.0,"tempo":1.5,
                   "loudness":1.0,"acousticness":1.0,"instrumentalness":0.5,
                   "liveness":0.5,"speechiness":0.5}
    normalizers = {"tempo":150.0,"loudness":60.0}
    score, total = 0.0, 0.0
    for feat, w in weights.items():
        val = row.get(feat)
        if pd.isna(val) or feat not in targets: continue
        diff = abs(float(val) - float(targets[feat])) / normalizers.get(feat, 1.0)
        score += w * (1.0 - min(diff, 1.0))
        total += w
    return score / total if total else 0.0


def get_filtered_tracks_local(
    genre_group, target_danceability, target_energy, target_key,
    target_loudness, target_mode, target_speechiness, target_acousticness,
    target_instrumentalness, target_liveness, target_valence, target_tempo,
    min_artist_pop, min_track_pop, limit=12,
) -> list[dict]:
    df = load_local_database()
    if df.empty: return []
    seed = GENRE_MAP.get(genre_group, "pop").lower()
    if "genre" in df.columns:
        mask = (
            df["genre"].str.lower().str.contains(seed, na=False)
            | df["genre"].str.lower().str.contains(genre_group.lower(), na=False)
        )
        filtered = df[mask].copy() if mask.any() else df.copy()
    else:
        filtered = df.copy()
    if "track_popularity"  in filtered.columns:
        filtered = filtered[filtered["track_popularity"].fillna(0)  >= min_track_pop]
    if "artist_popularity" in filtered.columns:
        filtered = filtered[filtered["artist_popularity"].fillna(0) >= min_artist_pop]
    if filtered.empty: return []
    targets = dict(
        danceability=target_danceability, energy=target_energy, valence=target_valence,
        loudness=target_loudness, acousticness=target_acousticness,
        instrumentalness=target_instrumentalness, liveness=target_liveness,
        speechiness=target_speechiness, tempo=target_tempo,
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


# ─── SPOTIFY PLAYLIST UPLOAD ──────────────────────────────────────────────────
def upload_playlist_to_spotify(
    rows: list[dict],
    playlist_name: str | None = None,
    playlist_description: str = "",
    public: bool = False,
) -> tuple[str | None, int, int]:
    if not rows:
        return None, 0, 0

    if playlist_name is None:
        playlist_name = f"Music Matcher – {datetime.now().strftime('%b %d, %Y %H:%M')}"

    with st.status("Building your playlist on Spotify…", expanded=True) as status:
        st.write(f"🔍 Validating {len(rows)} tracks…")
        resolved_ids, failed_rows = validate_or_resolve_ids(rows)

        if failed_rows:
            st.write(f"⚠️ Could not resolve **{len(failed_rows)}** track(s).")
        st.write(f"✅ Resolved **{len(resolved_ids)}** valid tracks.")
        status.update(label="Tracks validated.", state="running")

    if not resolved_ids:
        st.error("No tracks could be resolved. Check artist and song names in your database.")
        return None, 0, len(rows)

    playlist = spotify_post(
        "me/playlists",
        payload={"name": playlist_name, "public": public, "description": playlist_description},
        context="Create Playlist",
    )
    if playlist is None:
        return None, 0, len(rows)

    playlist_id  = playlist.get("id")
    playlist_url = playlist.get("external_urls", {}).get("spotify")

    if not playlist_id:
        st.error("Spotify returned a playlist with no ID.")
        return None, 0, len(rows)

    uris = [
        f"spotify:track:{tid}"
        for tid in resolved_ids
        if tid and tid.lower() not in ("nan", "none", "")
    ]

    n_added, n_failed = 0, len(failed_rows)
    for i in range(0, len(uris), 100):
        batch  = uris[i: i + 100]
        result = spotify_post(
            f"playlists/{playlist_id}/items",
            payload={"uris": batch},
            context=f"Add Items {i + 1}–{i + len(batch)}",
        )
        if result is not None:
            n_added += len(batch)
        else:
            n_failed += len(batch)

    return playlist_url, n_added, n_failed

# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding:4px 0 16px;">
        <span style="font-size:1.6rem;">🎧</span>
        <span style="font-size:1.15rem;font-weight:700;color:#f0f4f8;letter-spacing:-0.3px;">
            Music Matcher
        </span>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.sp_display_name:
        st.markdown(
            f"<div style='background:#1a2738;border-radius:10px;padding:10px 14px;"
            f"border:1px solid #263040;margin-bottom:8px;'>"
            f"<span style='font-size:0.82rem;color:#9ba8bb;'>Logged in as</span><br>"
            f"<strong style='color:#f0f4f8;font-size:0.92rem;'>"
            f"👤 {st.session_state.sp_display_name}</strong>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    meta = load_vector_with_meta()
    if meta:
        st.markdown(
            "<p style='color:#9ba8bb;font-size:0.72rem;font-weight:600;"
            "letter-spacing:0.08em;text-transform:uppercase;margin-bottom:6px;'>"
            "💾 Saved Taste Profile</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p style='color:#9ba8bb;font-size:0.75rem;margin:0 0 8px;'>"
            f"Last updated: {meta.get('saved_at', '—')}</p>",
            unsafe_allow_html=True,
        )
        with st.expander("View profile vector"):
            for k, v in meta["vector"].items():
                st.write(f"**{k}**: `{round(float(v), 3) if isinstance(v, (int, float)) else v}`")
    else:
        st.markdown(
            "<p style='color:#9ba8bb;font-size:0.82rem;'>"
            "No taste profile saved yet.</p>",
            unsafe_allow_html=True,
        )

    st.divider()
    if st.button("🔓 Log out", use_container_width=True):
        for k in _defaults:
            st.session_state[k] = _defaults[k]
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — PLAYLIST GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
catalog = _load_song_catalog()
log.info("Song catalog loaded: %d rows", len(catalog))

FEATURE_COLS = [
    "danceability","energy","key","loudness","mode",
    "speechiness","acousticness","instrumentalness",
    "liveness","valence","tempo","popularity",
]

# ── Header ──
st.markdown("""
<div style="margin-bottom:1.5rem;padding-bottom:1.2rem;
            border-bottom:2px solid #dde0ea;">
    <h1 style="font-size:2rem;font-weight:800;color:#111827;margin:0 0 6px;">
        🎵 AI Playlist Generator
    </h1>
    <p style="color:#4b5563;margin:0;font-size:0.95rem;">
        Pick up to 5 seed songs — we'll map your taste into a curated mix.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Controls row ──
col_genre, col_size = st.columns([4, 1.5])
with col_genre:
    genre_groups = st.multiselect(
        "Genre",
        options=list(GENRE_MAP.keys()),
        default=["Pop"],
        placeholder="Pick one or more genres…",
    )
with col_size:
    playlist_size = st.slider("Playlist size", 5, 50, 20)

st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

# ── Song picker ──
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

genre_label   = ", ".join(genre_groups) if genre_groups else "All"
genre_catalog = genre_catalog.sort_values("popularity", ascending=False, na_position="last")
genre_catalog["_label"] = (
    genre_catalog["artist_name"].fillna("Unknown") + " — " +
    genre_catalog["track_name"].fillna("Unknown")
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
    seed_df      = catalog.loc[seed_indices]
    cols = st.columns(len(seed_df))
    for col, (_, row) in zip(cols, seed_df.iterrows()):
        with col:
            st.markdown(
                f"<div style='background:white;border:1px solid #dde0ea;border-radius:12px;"
                f"padding:12px 14px;text-align:center;"
                f"box-shadow:0 1px 6px rgba(0,0,0,0.05);'>"
                f"<div style='font-size:0.82rem;font-weight:600;color:#111827;"
                f"white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>"
                f"🎵 {row.get('track_name','?')}</div>"
                f"<div style='font-size:0.75rem;color:#4b5563;margin-top:3px;"
                f"white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>"
                f"{row.get('artist_name','?')}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

if st.button("✨ Generate Playlist", type="primary", use_container_width=False):
    if not selected_labels:
        st.error("Please select at least 1 seed song.")
    else:
        with st.spinner("Mapping your taste into latent space…"):
            seed_indices    = [label_to_idx[l] for l in selected_labels if l in label_to_idx]
            seed_df         = catalog.loc[seed_indices]
            seed_specs_list = []
            for _, row in seed_df.iterrows():
                specs = row[FEATURE_COLS].to_dict()
                seed_specs_list.append(specs)

            try:
                centroid = seed_df[FEATURE_COLS].mean().to_dict()
                save_vector(centroid)
            except Exception as e:
                log.warning("Could not save feature vector: %s", e)

            seed_genres = (
                seed_df["genre"].dropna().unique().tolist()
                if "genre" in seed_df.columns else None
            )
            seed_language = detect_seed_language(seed_df["track_name"].dropna())

            raw_results = build_playlist_from_seeds(
                seed_specs_list, k=playlist_size,
                seed_genres=seed_genres, seed_language=seed_language,
            )

            display_data = [
                {
                    "ID":         str(r.get("track_id", "")),
                    "Artist":     str(r.get("artist_name", "Unknown")),
                    "Song Title": str(r.get("track_name",  "Unknown")),
                    "Genre":      str(r.get("genre", "")),
                    "BPM":        int(round(float(r.get("tempo", 0)))),
                }
                for r in raw_results
            ]

            st.session_state.playlist_data = pd.DataFrame(display_data)
            st.session_state.upload_url    = None

            if not display_data:
                st.error("No matching songs found. Try different seed songs or genres.")
            else:
                st.success(f"✅ Built a playlist with **{len(display_data)}** tracks!")

# ── Playlist result ──
if st.session_state.playlist_data is not None and not st.session_state.playlist_data.empty:
    st.divider()

    df_display = st.session_state.playlist_data
    c1, c2, c3 = st.columns(3)
    c1.metric("Tracks",  len(df_display))
    c2.metric("Artists", df_display["Artist"].nunique())
    c3.metric("Genres",  df_display["Genre"].nunique())

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        column_config={"BPM": st.column_config.NumberColumn("BPM", format="%d")},
    )

    st.divider()

    st.markdown("""
    <div style="margin-bottom:0.8rem;">
        <h4 style="font-size:1.05rem;font-weight:700;color:#111827;margin:0;">
            📤 Upload to Spotify
        </h4>
    </div>
    """, unsafe_allow_html=True)

    col_nm, col_vis = st.columns([3, 1])
    with col_nm:
        pl_name = st.text_input(
            "Playlist name",
            value=f"Music Matcher – {datetime.now().strftime('%b %d, %Y')}",
        )
    with col_vis:
        pl_public = st.toggle("Public", value=False)

    pl_desc = st.text_input(
        "Description (optional)",
        value=f"Generated by Music Matcher · {genre_label}",
    )

    if st.button("🚀 Upload to Spotify", type="primary"):
        url, n_added, n_failed = upload_playlist_to_spotify(
            df_display.to_dict("records"), pl_name, pl_desc, pl_public
        )
        if url:
            st.session_state.upload_url = url
            if n_failed == 0:
                st.success(f"🎉 Playlist created with all **{n_added}** tracks!")
            else:
                st.warning(f"Playlist created: **{n_added}** added, **{n_failed}** unresolved.")

    if st.session_state.upload_url:
        st.link_button("🎧 Open on Spotify", st.session_state.upload_url, type="primary")
