import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime

import spotipy
from spotipy.oauth2 import SpotifyOAuth

# -----------------------
# CONFIG
# -----------------------
st.set_page_config(page_title="Music Matcher", page_icon="🎧", layout="wide")

# IMPORTANT: set these in .streamlit/secrets.toml (recommended)
# [spotify]
# client_id="..."
# client_secret="..."
# redirect_uri="http://localhost:8501"
SPOTIFY_CLIENT_ID = st.secrets["spotify"]["client_id"]
SPOTIFY_CLIENT_SECRET = st.secrets["spotify"]["client_secret"]
SPOTIFY_REDIRECT_URI = st.secrets["spotify"]["redirect_uri"]

SCOPE = "user-read-private"

# Map your UI genres to Spotify recommendation seed genres (must be from Spotify's allowed list)
GENRE_MAP = {
    "Pop": "pop",
    "Hip-Hop": "hip-hop",
    "Rock": "rock",
    "Electro": "electronic",
    "Jazz": "jazz",
    "Indie": "indie",
    "R&B": "r-n-b",
    "Rap": "rap",
}

# -----------------------
# HELPERS
# -----------------------
def format_duration(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"

def get_spotify_client() -> spotipy.Spotify:
    """
    Spotify OAuth inside Streamlit:
    - Creates an auth URL
    - Reads ?code=... from st.query_params after Spotify redirects back
    """
    auth = SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=SPOTIFY_REDIRECT_URI,
        scope=SCOPE,
        cache_handler=spotipy.cache_handler.MemoryCacheHandler(),
        show_dialog=False,
    )

    qp = st.query_params
    code = qp.get("code", None)

    if "sp_token" not in st.session_state:
        st.session_state.sp_token = None

    if st.session_state.sp_token is None:
        if code:
            token_info = auth.get_access_token(code, as_dict=True)
            st.session_state.sp_token = token_info["access_token"]
            # Remove code from URL for cleanliness
            st.query_params.clear()
            st.rerun()
        else:
            login_url = auth.get_authorize_url()
            st.info("Log in to Spotify to load tracks.")
            st.link_button("Log in to Spotify", login_url)
            st.stop()

    return spotipy.Spotify(auth=st.session_state.sp_token)

def safe_year(track_obj) -> int:
    # album.release_date can be 'YYYY', 'YYYY-MM-DD', etc.
    album = track_obj.get("album", {})
    rd = album.get("release_date", "")
    try:
        return int(str(rd)[:4])
    except:
        return 2020

@st.cache_data(ttl=3600)
def get_advanced_filtered_tracks_spotify(
    genre: str,
    target_bpm: int,
    min_dur: int,
    max_dur: int,
    min_year: int,
    max_year: int,
    limit: int = 12,
    market: str = "CA",
):
    sp = get_spotify_client()

    seed_genre = GENRE_MAP.get(genre, "pop")

    def attempt_fetch(bpm, d_min, d_max, y_min, y_max, n=75):
        # Use recommendations to bias toward tempo/duration
        rec = sp.recommendations(
            seed_genres=[seed_genre],
            limit=min(n, 100),
            target_tempo=float(bpm),
            min_duration_ms=int(d_min * 1000),
            max_duration_ms=int(d_max * 1000),
            market=market,
        )

        tracks = rec.get("tracks", [])
        if not tracks:
            return []

        ids = [t["id"] for t in tracks if t.get("id")]
        feats = sp.audio_features(ids)  # returns list aligned with ids
        feat_by_id = {f["id"]: f for f in feats if f and f.get("id")}

        pool = []
        for t in tracks:
            tid = t.get("id")
            if not tid:
                continue

            f = feat_by_id.get(tid)
            if not f:
                continue

            tempo = f.get("tempo", 0.0)          # BPM
            loudness = f.get("loudness", None)   # dB
            dur_s = (t.get("duration_ms", 0) / 1000.0)
            year = safe_year(t)

            # Filters (duration + year)
            if not (d_min <= dur_s <= d_max):
                continue
            if not (y_min <= year <= y_max):
                continue

            pool.append({
                "track_obj": t,
                "bpm": tempo,
                "diff": abs(tempo - bpm),
                "year": year,
                "duration": dur_s,
                "loudness": loudness,
            })

        pool.sort(key=lambda x: x["diff"])
        return pool

    # Attempt 1: strict
    matches = attempt_fetch(target_bpm, min_dur, max_dur, min_year, max_year)

    # Attempt 2: relaxed
    if len(matches) < 5:
        matches = attempt_fetch(
            target_bpm,
            max(0, min_dur - 60), max_dur + 60,
            min_year - 5, max_year + 5
        )

    # Attempt 3: fallback (ignore year/duration)
    if len(matches) < 5:
        matches = attempt_fetch(target_bpm, 0, 9999, 1900, 2100)

    final_tracks = []
    for item in matches[:limit]:
        t = item["track_obj"]
        artists = ", ".join(a["name"] for a in t.get("artists", [])) if t.get("artists") else "Unknown"
        images = t.get("album", {}).get("images", [])
        cover = images[1]["url"] if len(images) > 1 else (images[0]["url"] if images else None)

        final_tracks.append({
            "ID": t.get("id"),
            "Artist": artists,
            "Song Title": t.get("name"),
            "BPM": int(round(item["bpm"])) if item["bpm"] else 0,
            "Duration": format_duration(item["duration"]),
            "Release Year": item["year"],
            "Gain": item["loudness"],  # Spotify uses loudness (dB) in audio features
            "Preview": t.get("preview_url"),
            "Cover": cover,
            "Link": t.get("external_urls", {}).get("spotify"),
        })

    return final_tracks

# -----------------------
# SESSION STATE
# -----------------------
if "playlist_data" not in st.session_state: st.session_state.playlist_data = None
if "swiper_songs" not in st.session_state: st.session_state.swiper_songs = []
if "current_index" not in st.session_state: st.session_state.current_index = 0
if "liked_songs" not in st.session_state: st.session_state.liked_songs = []

# -----------------------
# NAVIGATION
# -----------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🎛️ Playlist Generator", "🔥 Song Swiper"])

# ==========================================
# PAGE 1: PLAYLIST GENERATOR
# ==========================================
if page == "🎛️ Playlist Generator":
    st.sidebar.header("Generator Settings")
    genre = st.sidebar.selectbox("Select Genre", ["Pop", "Hip-Hop", "Rock", "Electro", "Jazz", "Indie", "R&B"])

    st.sidebar.subheader("Attributes")
    tempo = st.sidebar.slider("🥁 BPM (Target)", 60, 200, 120)
    dur_range = st.sidebar.slider("⏳ Duration (Seconds)", 60, 600, (120, 300))
    year_range = st.sidebar.slider("📅 Release Year", 1980, 2026, (2000, 2026))
    gain_threshold = st.sidebar.slider("🔊 Min Gain (Loudness, dB)", -60.0, 0.0, -15.0)

    st.title("🎵 AI Playlist Generator (Spotify)")
    st.markdown(f"Create a **{genre}** playlist with your custom filters.")

    if st.button("✨ Generate Playlist", type="primary"):
        with st.spinner(f"Filtering {genre} tracks..."):
            real_data = get_advanced_filtered_tracks_spotify(
                genre, tempo,
                dur_range[0], dur_range[1],
                year_range[0], year_range[1],
                12
            )

            # Apply gain/loudness filter (Spotify loudness: closer to 0 is louder)
            if real_data:
                filtered = [r for r in real_data if (r["Gain"] is None) or (r["Gain"] >= gain_threshold)]
                st.session_state.playlist_data = pd.DataFrame(filtered if filtered else real_data)

                if len(st.session_state.playlist_data) >= 5:
                    st.success(f"Playlist created with {len(st.session_state.playlist_data)} songs!")
                else:
                    st.warning("Found fewer than 5 songs after filtering; try relaxing filters.")
            else:
                st.error("No songs found. Try a different genre.")

    if st.session_state.playlist_data is not None and len(st.session_state.playlist_data) > 0:
        st.divider()
        st.subheader("Your Curated Mix")

        st.dataframe(
            st.session_state.playlist_data,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Cover": st.column_config.ImageColumn("Art", width=50),
                "Preview": st.column_config.LinkColumn("Preview"),
                "Link": st.column_config.LinkColumn("Link"),
                "BPM": st.column_config.NumberColumn("BPM", format="%d")
            }
        )

        first_track_link = st.session_state.playlist_data.iloc[0]["Link"]
        if first_track_link:
            st.link_button("🎧 Open on Spotify", first_track_link)

# ==========================================
# PAGE 2: SONG SWIPER
# ==========================================
elif page == "🔥 Song Swiper":
    st.title("🔥 Song Swiper (Spotify)")

    with st.expander("Filter Swiper"):
        swiper_genre = st.selectbox("Genre", ["Pop", "Rock", "Rap", "Electro"], index=0)
        swiper_bpm = st.slider("Target BPM", 60, 180, 120)
        if st.button("🔄 Reload Stack"):
            st.session_state.swiper_songs = get_advanced_filtered_tracks_spotify(
                swiper_genre, swiper_bpm,
                0, 900, 1990, 2026, 15
            )
            st.session_state.current_index = 0
            st.rerun()

    if not st.session_state.swiper_songs:
        st.session_state.swiper_songs = get_advanced_filtered_tracks_spotify("Pop", 120, 0, 900, 1990, 2026, 15)

    if st.session_state.swiper_songs and st.session_state.current_index < len(st.session_state.swiper_songs):
        current_song = st.session_state.swiper_songs[st.session_state.current_index]

        with st.container(border=True):
            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                if current_song.get("Cover"):
                    st.image(current_song["Cover"], width=300)
                st.subheader(current_song["Song Title"])
                st.write(f"**{current_song['Artist']}**")
                st.caption(f"📅 {current_song['Release Year']} • ⏱ {current_song['Duration']}")
                if current_song.get("Preview"):
                    st.audio(current_song["Preview"], format="audio/mp3")
                else:
                    st.caption("No preview available for this track on Spotify.")

        col_pass, col_space, col_like = st.columns([1, 2, 1])
        with col_pass:
            if st.button("❌ Pass", use_container_width=True):
                st.session_state.current_index += 1
                st.rerun()
        with col_like:
            if st.button("💚 Like", type="primary", use_container_width=True):
                st.session_state.liked_songs.append(current_song)
                st.session_state.current_index += 1
                st.rerun()

        st.progress(st.session_state.current_index / len(st.session_state.swiper_songs))

    else:
        st.balloons()
        st.success("Stack empty!")

        if len(st.session_state.liked_songs) > 0:
            st.subheader("Your Liked Songs")
            df_liked = pd.DataFrame(st.session_state.liked_songs)
            st.dataframe(df_liked, use_container_width=True)

        if st.button("Restart"):
            st.session_state.current_index = 0
            st.rerun()
