import streamlit as st
import pandas as pd
import numpy as np
import time

# --- Page Configuration ---
st.set_page_config(page_title="Music Matcher", page_icon="🎧", layout="wide")


# --- Helper Functions ---
def get_random_songs(n=10):
    """Generates a list of random songs for the swiper."""
    artists = ["The Weeknd", "Taylor Swift", "Kendrick Lamar", "Daft Punk", "Tame Impala", "Arctic Monkeys", "SZA",
               "Drake", "Pink Floyd", "Radiohead"]
    titles = ["Midnight City", "Vibe Check", "Neon Lights", "Space Walk", "Retro Grade", "Night Drive", "Solar Power",
              "Deep Dive", "Echoes", "Pulse"]

    songs = []
    for _ in range(n):
        songs.append({
            "Artist": np.random.choice(artists),
            "Title": np.random.choice(titles),
            "Genre": np.random.choice(["Pop", "Indie", "Hip-Hop", "Electronic"]),
            "CoverColor": np.random.choice(["#FF4B4B", "#1DB954", "#4B9CFF", "#FFA500", "#9C27B0"])
        })
    return songs


# --- Session State Initialization ---
if 'playlist_data' not in st.session_state:
    st.session_state.playlist_data = None
if 'swiper_songs' not in st.session_state:
    st.session_state.swiper_songs = get_random_songs(15)
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'liked_songs' not in st.session_state:
    st.session_state.liked_songs = []

# --- Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🎛️ Playlist Generator", "🔥 Song Swiper"])

# ==========================================
# PAGE 1: PLAYLIST GENERATOR
# ==========================================
if page == "🎛️ Playlist Generator":

    # -- Sidebar Controls --
    st.sidebar.header("Generator Settings")
    genre = st.sidebar.selectbox("Select Genre", ["Pop", "Hip-Hop", "Rock", "Electronic", "Jazz", "Indie", "R&B"])

    st.sidebar.subheader("Core Vibe")
    danceability = st.sidebar.slider("💃 Danceability", 0.0, 1.0, 0.7)
    energy = st.sidebar.slider("⚡ Energy", 0.0, 1.0, 0.6)
    tempo = st.sidebar.slider("🥁 Tempo (BPM)", 50, 200, 120)

    # -- RE-ADDED ATTRIBUTES (Hidden in Expander to save space) --
    with st.sidebar.expander("🎚️ Advanced Audio Features", expanded=False):
        loudness = st.slider("🔊 Loudness (dB)", -60.0, 0.0, -5.0)
        acousticness = st.slider("🎻 Acousticness", 0.0, 1.0, 0.1)
        speechiness = st.slider("🎤 Speechiness", 0.0, 1.0, 0.05)
        instrumentalness = st.slider("🎹 Instrumentalness", 0.0, 1.0, 0.0)
        liveness = st.slider("🏟️ Liveness", 0.0, 1.0, 0.1)
        valence = st.slider("🌞 Valence (Happiness)", 0.0, 1.0, 0.5)

    # -- Main Content --
    st.title("🎵 AI Playlist Generator")
    st.markdown(f"Create a **{genre}** playlist optimized for your selected vibe.")

    col_action, col_spacer = st.columns([1, 4])
    with col_action:
        if st.button("✨ Generate Playlist", type="primary", use_container_width=True):
            with st.spinner("Analyzing audio features..."):
                time.sleep(1.0)
                # Mock Logic
                mock_data = []
                for i in range(12):  # Increased count to 12
                    mock_data.append({
                        "Artist": "Various Artists",
                        "Song Title": f"Generated Track #{i + 1}",
                        "Genre": genre,
                        "Danceability": round(np.random.normal(danceability, 0.1), 2),
                        "Energy": round(np.random.normal(energy, 0.1), 2),
                        "Tempo": f"{int(np.random.normal(tempo, 5))} BPM",
                        "Valence": round(np.random.normal(valence, 0.1), 2),
                        "Acousticness": round(np.random.normal(acousticness, 0.1), 2)
                    })
                st.session_state.playlist_data = pd.DataFrame(mock_data)
                st.success("Playlist created!")

    if st.session_state.playlist_data is not None:
        st.divider()
        st.subheader("Your Curated Mix")

        # Displaying more columns now
        st.dataframe(
            st.session_state.playlist_data,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Danceability": st.column_config.ProgressColumn("Danceability", min_value=0, max_value=1),
                "Energy": st.column_config.ProgressColumn("Energy", min_value=0, max_value=1),
                "Valence": st.column_config.ProgressColumn("Happiness", min_value=0, max_value=1, format="%.2f"),
                "Acousticness": st.column_config.NumberColumn("Acoustic", format="%.2f")
            }
        )
        c1, c2, c3 = st.columns([1, 1, 1])
        with c2:
            st.button("Add to Spotify", icon="🎧", use_container_width=True)

# ==========================================
# PAGE 2: SONG SWIPER
# ==========================================
elif page == "🔥 Song Swiper":
    st.title("🔥 Song Swiper")
    st.markdown("Discover new music one track at a time.")

    if st.session_state.current_index < len(st.session_state.swiper_songs):
        current_song = st.session_state.swiper_songs[st.session_state.current_index]

        with st.container(border=True):
            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                st.markdown(
                    f"""
                    <div style="background-color:{current_song['CoverColor']}; 
                                height:200px; width:100%; border-radius:10px; margin-bottom:20px;">
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.subheader(current_song['Title'])
                st.write(f"**{current_song['Artist']}** • {current_song['Genre']}")

        col_pass, col_space, col_like = st.columns([1, 2, 1])

        with col_pass:
            if st.button("❌ Pass", use_container_width=True):
                st.session_state.current_index += 1
                st.rerun()

        with col_like:
            if st.button("💚 Like", type="primary", use_container_width=True):
                st.session_state.liked_songs.append(current_song)
                st.session_state.current_index += 1
                st.toast(f"Added '{current_song['Title']}' to your likes!", icon="🔥")
                st.rerun()

        progress = st.session_state.current_index / len(st.session_state.swiper_songs)
        st.progress(progress,
                    text=f"Stack progress: {st.session_state.current_index}/{len(st.session_state.swiper_songs)}")

    else:
        st.balloons()
        st.success("You've gone through all the songs!")

        if len(st.session_state.liked_songs) > 0:
            st.subheader(f"You matched with {len(st.session_state.liked_songs)} songs:")
            df_liked = pd.DataFrame(st.session_state.liked_songs)
            st.dataframe(df_liked[["Artist", "Title", "Genre"]], use_container_width=True, hide_index=True)

            if st.button("Add Matches to Spotify", icon="🎧"):
                st.toast("Syncing with Spotify...", icon="✅")
        else:
            st.write("You didn't like any songs. Tough crowd! 😅")

        if st.button("Restart Swiper"):
            st.session_state.current_index = 0
            st.session_state.liked_songs = []
            st.rerun() # Hello

# Hello this is a github test
