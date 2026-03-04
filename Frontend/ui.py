import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import requests
from datetime import datetime, timezone

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Model'))
from feature_io import save_vector, load_vector_with_meta, VECTOR_PATH

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

LOCAL_DB_PATH = "songs_database.csv"

GENRE_MAP = {
    "Pop": "pop", "Hip-Hop": "hip-hop", "Rock": "rock",
    "Electro": "electronic", "Jazz": "jazz", "Indie": "indie",
    "R&B": "r-n-b", "Rap": "hip-hop", "Classical": "classical",
    "Country": "country", "Metal": "metal",
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

    # Step 3 — Add tracks via POST /playlists/{id}/items (Feb 2026 renamed endpoint)
    uris     = [f"spotify:track:{tid}" for tid in resolved_ids]
    n_added  = 0
    n_failed = len(failed_rows)

    for i in range(0, len(uris), 100):
        batch  = uris[i : i + 100]
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
                        f"{SPOTIFY_API_BASE}/playlists/{pl_id}/items",
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
    st.sidebar.header("Settings")

    genre_group = st.sidebar.selectbox("Genre Group", list(GENRE_MAP.keys()))

    with st.sidebar.expander("🎹 Musical Features", expanded=True):
        target_bpm          = st.slider("Tempo (BPM)", 50, 200, 120)
        target_key          = st.slider("Key (0=C … 11=B)", 0, 11, 0)
        target_mode         = st.radio(
            "Mode", [0, 1],
            format_func=lambda x: "Minor" if x == 0 else "Major",
            horizontal=True,
        )
        target_danceability = st.slider("Danceability",   0.0, 1.0, 0.5)
        target_energy       = st.slider("Energy",         0.0, 1.0, 0.5)
        target_valence      = st.slider("Valence (Mood)", 0.0, 1.0, 0.5)
        target_loudness     = st.slider("Loudness (dB)", -60.0, 0.0, -10.0)

    with st.sidebar.expander("🎻 Technical Features", expanded=False):
        target_acousticness     = st.slider("Acousticness",     0.0, 1.0, 0.10)
        target_instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.00)
        target_liveness         = st.slider("Liveness",         0.0, 1.0, 0.10)
        target_speechiness      = st.slider("Speechiness",      0.0, 1.0, 0.05)

    with st.sidebar.expander("⭐ Popularity Filters", expanded=True):
        st.caption("Minimum popularity thresholds.")
        min_artist_pop = st.slider("Min Artist Popularity", 0, 100, 50)
        min_track_pop  = st.slider("Min Track Popularity",  0, 100, 40)

    # ── Feature Vector (assembled live from sidebar values) ───────────────────
    current_vector = {
        "genre":                 genre_group,
        "tempo":                 target_bpm,
        "key":                   target_key,
        "mode":                  target_mode,
        "danceability":          target_danceability,
        "energy":                target_energy,
        "valence":               target_valence,
        "loudness":              target_loudness,
        "acousticness":          target_acousticness,
        "instrumentalness":      target_instrumentalness,
        "liveness":              target_liveness,
        "speechiness":           target_speechiness,
        "min_track_popularity":  min_track_pop,
        "min_artist_popularity": min_artist_pop,
    }

    # ── Show last saved vector metadata in sidebar ────────────────────────────
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

    # ── Main page ─────────────────────────────────────────────────────────────
    st.title("🎵 AI Playlist Generator")
    st.markdown(
        f"Selecting **{genre_group}** tracks from your local database "
        "ranked by **audio-feature similarity**."
    )

    if st.button("✨ Generate Playlist", type="primary"):
        with st.spinner("Scanning local database…"):

            # Auto-save feature vector on every generation
            try:
                save_vector(current_vector)
            except Exception as e:
                st.warning(f"Could not save feature vector: {e}")

            data = get_filtered_tracks_local(
                genre_group=genre_group,
                target_danceability=target_danceability,
                target_energy=target_energy,
                target_key=target_key,
                target_loudness=target_loudness,
                target_mode=target_mode,
                target_speechiness=target_speechiness,
                target_acousticness=target_acousticness,
                target_instrumentalness=target_instrumentalness,
                target_liveness=target_liveness,
                target_valence=target_valence,
                target_tempo=target_bpm,
                min_artist_pop=min_artist_pop,
                min_track_pop=min_track_pop,
                limit=15,
            )
            st.session_state.playlist_data = pd.DataFrame(data)
            st.session_state.upload_url    = None
            if not data:
                st.error("No songs found. Try lowering the popularity thresholds.")
            else:
                st.success(f"Found **{len(data)}** tracks matching your criteria!")

    if st.session_state.playlist_data is not None and not st.session_state.playlist_data.empty:
        st.divider()
        st.subheader("Your Curated Mix")

        st.dataframe(
            st.session_state.playlist_data.drop(columns=["Similarity"], errors="ignore"),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Cover":   st.column_config.ImageColumn("Art", width=50),
                "Preview": st.column_config.LinkColumn("Preview"),
                "Link":    st.column_config.LinkColumn("Link"),
                "BPM":     st.column_config.NumberColumn("BPM", format="%d"),
                "Pop (T)": st.column_config.ProgressColumn("Track Pop",  min_value=0, max_value=100),
                "Pop (A)": st.column_config.ProgressColumn("Artist Pop", min_value=0, max_value=100),
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
            value=f"Generated by Music Matcher | Genre: {genre_group}",
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
