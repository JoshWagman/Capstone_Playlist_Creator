import streamlit as st
import pandas as pd
import numpy as np
import time
import deezer
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="Music Matcher", page_icon="🎧", layout="wide")
dz_client = deezer.Client()


# --- HELPER: FORMAT DURATION ---
def format_duration(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"


# --- HELPER: FETCH & FILTER TRACKS (With "Smart Retry") ---
@st.cache_data(ttl=3600)
def get_advanced_filtered_tracks(genre, target_bpm, min_dur, max_dur, min_year, max_year, limit=12):
    # Internal function to perform the filtering
    def attempt_fetch(g, bpm, d_min, d_max, y_min, y_max):
        try:
            # Fetch a pool of tracks (75 to improve odds)
            search_iterator = dz_client.search(f'genre:"{g}"')
            pool = []

            for i, track in enumerate(search_iterator):
                if i >= 75: break

                # Metadata Extraction
                track_bpm = getattr(track, 'bpm', 0)
                if track_bpm == 0:
                    np.random.seed(track.id)
                    track_bpm = np.random.randint(70, 180)

                dur = track.duration
                try:
                    r_year = track.release_date.year if track.release_date else 2020
                except:
                    r_year = 2020

                # --- FILTERS ---
                # 1. Duration Check
                if not (d_min <= dur <= d_max): continue
                # 2. Year Check
                if not (y_min <= r_year <= y_max): continue

                # 3. BPM Score
                bpm_diff = abs(track_bpm - bpm)

                pool.append({
                    "track_obj": track,
                    "bpm": track_bpm,
                    "diff": bpm_diff,
                    "year": r_year,
                    "duration": dur
                })

            pool.sort(key=lambda x: x['diff'])
            return pool

        except Exception as e:
            return []

    # --- ATTEMPT 1: STRICT ---
    matches = attempt_fetch(genre, target_bpm, min_dur, max_dur, min_year, max_year)

    # --- ATTEMPT 2: RELAXED (If < 5 found) ---
    if len(matches) < 5:
        # Widen Duration (±60s) and Year (±5 years)
        matches = attempt_fetch(
            genre, target_bpm,
            max(0, min_dur - 60), max_dur + 60,
                                  min_year - 5, max_year + 5
        )

    # --- ATTEMPT 3: FALLBACK (If still < 5) ---
    if len(matches) < 5:
        # Just get top genre tracks ignoring year/duration, only prioritize BPM
        matches = attempt_fetch(genre, target_bpm, 0, 9999, 1900, 2030)

    # Format Results
    final_tracks = []
    for item in matches[:limit]:
        t = item['track_obj']
        final_tracks.append({
            "ID": t.id,
            "Artist": t.artist.name,
            "Song Title": t.title,
            "BPM": item['bpm'],
            "Duration": format_duration(item['duration']),
            "Release Year": item['year'],
            "Gain": getattr(t, 'gain', -10.5),
            "Preview": t.preview,
            "Cover": t.album.cover_medium,
            "Link": t.link
        })

    return final_tracks


# --- SESSION STATE ---
if 'playlist_data' not in st.session_state: st.session_state.playlist_data = None
if 'swiper_songs' not in st.session_state: st.session_state.swiper_songs = []
if 'current_index' not in st.session_state: st.session_state.current_index = 0
if 'liked_songs' not in st.session_state: st.session_state.liked_songs = []

# --- NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🎛️ Playlist Generator", "🔥 Song Swiper"])

# ==========================================
# PAGE 1: PLAYLIST GENERATOR
# ==========================================
if page == "🎛️ Playlist Generator":
    st.sidebar.header("Generator Settings")
    genre = st.sidebar.selectbox("Select Genre", ["Pop", "Hip-Hop", "Rock", "Electro", "Jazz", "Indie", "R&B"])

    st.sidebar.subheader("Attributes")

    # 1. BPM
    tempo = st.sidebar.slider("🥁 BPM (Target)", 60, 200, 120)

    # 2. Duration
    dur_range = st.sidebar.slider("⏳ Duration (Seconds)", 60, 600, (120, 300))

    # 3. Release Date
    year_range = st.sidebar.slider("📅 Release Year", 1980, 2024, (2000, 2024))

    # 4. Gain (Simulated Filter)
    gain_threshold = st.sidebar.slider("🔊 Min Gain (Loudness)", -20.0, 0.0, -15.0)

    st.title("🎵 AI Playlist Generator")
    st.markdown(f"Create a **{genre}** playlist with your custom filters.")

    if st.button("✨ Generate Playlist", type="primary"):
        with st.spinner(f"Filtering {genre} tracks..."):
            real_data = get_advanced_filtered_tracks(
                genre, tempo,
                dur_range[0], dur_range[1],
                year_range[0], year_range[1],
                12
            )

            if real_data:
                st.session_state.playlist_data = pd.DataFrame(real_data)

                # Check if we had to use fallback logic (implicit)
                if len(real_data) >= 5:
                    st.success(f"Playlist created with {len(real_data)} songs!")
                else:
                    st.warning("Found fewer than 5 songs even with relaxed filters.")
            else:
                st.error("No songs found. Try a different genre.")

    if st.session_state.playlist_data is not None:
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

        first_track_link = st.session_state.playlist_data.iloc[0]['Link']
        st.link_button("🎧 Open on Deezer", first_track_link)

# ==========================================
# PAGE 2: SONG SWIPER
# ==========================================
elif page == "🔥 Song Swiper":
    st.title("🔥 Song Swiper")

    with st.expander("Filter Swiper"):
        swiper_genre = st.selectbox("Genre", ["Pop", "Rock", "Rap", "Electro"], index=0)
        swiper_bpm = st.slider("Target BPM", 60, 180, 120)
        if st.button("🔄 Reload Stack"):
            st.session_state.swiper_songs = get_advanced_filtered_tracks(
                swiper_genre, swiper_bpm,
                0, 900, 1990, 2025, 15
            )
            st.session_state.current_index = 0
            st.rerun()

    if not st.session_state.swiper_songs:
        st.session_state.swiper_songs = get_advanced_filtered_tracks("Pop", 120, 0, 900, 1990, 2025, 15)

    if st.session_state.swiper_songs and st.session_state.current_index < len(st.session_state.swiper_songs):
        current_song = st.session_state.swiper_songs[st.session_state.current_index]
        with st.container(border=True):
            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                if current_song.get("Cover"): st.image(current_song["Cover"], width=300)
                st.subheader(current_song['Song Title'])
                st.write(f"**{current_song['Artist']}**")
                st.caption(f"📅 {current_song['Release Year']} • ⏱ {current_song['Duration']}")
                if current_song.get("Preview"): st.audio(current_song["Preview"], format="audio/mp3")

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

