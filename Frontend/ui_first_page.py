"""
Playlist Creator UI

Pipeline:
  1. VAE or AAE (02_encoder_training.ipynb / 02b_aae_encoder_training.ipynb) encodes real
     Spotify song vectors into a latent space. Songs from cleaned_songs_with_groups.csv
     are vectorized and mapped; genres/similarity are characterized by distance.

  2. This UI lets the user tune sliders to express their ideal song type. We build a
     preference vector (user input only—not a real song vector).

  3. The preference vector is sent to an API (TBD) that will:
     - Map the point onto the latent space
     - Select a radius around that point
     - Retrieve real songs within the radius
     - Return them as real song objects (artist, track name, etc.)

  4. The UI displays the returned playlist.

  Future: Song Swiper functionality will be added later.
"""

import streamlit as st
import pandas as pd

# --- Song Vector Format (matches cleaned_songs_with_groups.csv) ---
# Order of metrics in the preference vector:
#   0: danceability (0-1)
#   1: energy (0-1)
#   2: key (0-11, C=0..B=11)
#   3: loudness (dB, typically -60 to 0)
#   4: mode (0=minor, 1=major)
#   5: speechiness (0-1)
#   6: acousticness (0-1)
#   7: instrumentalness (0-1)
#   8: liveness (0-1)
#   9: valence (0-1)
#  10: tempo (BPM)
#  11: artist_pop (0-100)
#  12: track_pop (0-100)
#  13: genre_group (string, e.g. "pop", "hip_hop")


def build_song_vector(danceability, energy, key, loudness, mode, speechiness, acousticness,
                     instrumentalness, liveness, valence, tempo, artist_pop, track_pop, genre_group):
    """Build a preference vector from slider values. Matches cleaned_songs_with_groups.csv format."""
    return [danceability, energy, key, loudness, mode, speechiness, acousticness,
            instrumentalness, liveness, valence, tempo, artist_pop, track_pop, genre_group]


def get_playlist_from_preference(preference_vector):
    """
    Placeholder for the API that maps the preference vector onto the latent space,
    finds real songs within a radius, and returns them.

    Expected API behavior:
      - Input: preference_vector (list of 14 elements, see build_song_vector)
      - Output: list of dicts, each with at least:
          - artist_name or "Artist"
          - track_name or "Song Title"
          - (optional: id, link, preview_url, cover, metrics, etc.)

    Returns empty list until the API is implemented.
    """
    # TODO: Replace with actual API call, e.g.:
    #   response = requests.post(API_URL + "/recommend", json={"preference_vector": preference_vector})
    #   return response.json().get("songs", [])
    return []


# --- Page Configuration ---
st.set_page_config(page_title="Music Matcher", page_icon="🎧", layout="wide")

# --- Session State ---
if "playlist_data" not in st.session_state:
    st.session_state.playlist_data = None
if "preference_vector" not in st.session_state:
    st.session_state.preference_vector = None

# --- Playlist Generator ---
st.title("🎵 AI Playlist Generator")
st.markdown(
    "Tune the sliders to your ideal song type. Your preference is sent to the latent-space "
    "model, which finds real songs near that point and builds your playlist."
)

st.sidebar.header("Generator Settings")
st.sidebar.subheader("Song Metrics")
danceability = st.sidebar.slider("💃 Danceability", 0.0, 1.0, 0.7)
energy = st.sidebar.slider("⚡ Energy", 0.0, 1.0, 0.6)
key = st.sidebar.selectbox(
    "🎹 Key",
    range(12),
    format_func=lambda x: ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"][x],
    index=0,
)
loudness = st.sidebar.slider("🔊 Loudness (dB)", -60.0, 0.0, -5.0)
mode = st.sidebar.selectbox("🎵 Mode", [0, 1], format_func=lambda x: "Minor" if x == 0 else "Major", index=0)
speechiness = st.sidebar.slider("🎤 Speechiness", 0.0, 1.0, 0.05)
acousticness = st.sidebar.slider("🎻 Acousticness", 0.0, 1.0, 0.1)
instrumentalness = st.sidebar.slider("🎹 Instrumentalness", 0.0, 1.0, 0.0)
liveness = st.sidebar.slider("🏟️ Liveness", 0.0, 1.0, 0.1)
valence = st.sidebar.slider("🌞 Valence (Happiness)", 0.0, 1.0, 0.5)
tempo = st.sidebar.slider("🥁 Tempo (BPM)", 50, 200, 120)
artist_pop = st.sidebar.slider("📊 Artist Popularity", 0, 100, 50)
track_pop = st.sidebar.slider("📊 Track Popularity", 0, 100, 50)
genre_group = st.sidebar.selectbox(
    "Genre Group",
    ["pop", "rock", "hip_hop", "electronic", "jazz", "indie", "r&b", "latin", "metal"],
    index=0,
)

if st.button("✨ Generate Playlist", type="primary"):
    preference_vector = build_song_vector(
        danceability, energy, key, loudness, mode, speechiness, acousticness,
        instrumentalness, liveness, valence, tempo, artist_pop, track_pop, genre_group,
    )
    st.session_state.preference_vector = preference_vector

    with st.spinner("Mapping to latent space and finding songs..."):
        songs = get_playlist_from_preference(preference_vector)

    if songs:
        st.session_state.playlist_data = songs
        st.success(f"Playlist created with {len(songs)} songs!")
    else:
        st.session_state.playlist_data = None
        st.info(
            "No songs returned yet. The recommendation API is not yet connected. "
            "Your preference vector has been recorded—connect the latent-space API to see real playlists."
        )

if st.session_state.preference_vector is not None:
    with st.expander("📐 Recorded preference vector (sent to API)"):
        st.code(
            "# Order: danceability, energy, key, loudness, mode, speechiness, acousticness, "
            "instrumentalness, liveness, valence, tempo, artist_pop, track_pop, genre_group\n"
            f"{st.session_state.preference_vector}"
        )

if st.session_state.playlist_data is not None and len(st.session_state.playlist_data) > 0:
    st.divider()
    st.subheader("Your Curated Mix")

    # Normalize keys (API may use artist_name/track_name or Artist/Song Title)
    songs = st.session_state.playlist_data
    if isinstance(songs[0], dict):
        normalized = []
        for s in songs:
            row = dict(s)
            row["Artist"] = row.get("Artist") or row.get("artist_name", "—")
            row["Song Title"] = row.get("Song Title") or row.get("track_name", "—")
            normalized.append(row)
        df = pd.DataFrame(normalized)
    else:
        df = pd.DataFrame(songs)

    # Prefer Artist, Song Title first; then other columns
    all_cols = list(df.columns)
    priority = ["Artist", "Song Title"]
    display_cols = [c for c in priority if c in all_cols] + [c for c in all_cols if c not in priority]
    df_display = df[display_cols]

    st.dataframe(df_display, use_container_width=True, hide_index=True)
