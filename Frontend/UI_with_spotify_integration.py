import streamlit as st
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import requests

# -----------------------
# CONFIG
# -----------------------
st.set_page_config(page_title="Music Matcher (Direct HTTP)", page_icon="⚡", layout="wide")

SPOTIFY_CLIENT_ID = st.secrets["spotify"]["client_id"]
SPOTIFY_CLIENT_SECRET = st.secrets["spotify"]["client_secret"]
SPOTIFY_REDIRECT_URI = st.secrets["spotify"]["redirect_uri"]
SCOPE = "user-read-private"

GENRE_MAP = {
    "Pop": "pop", "Hip-Hop": "hip-hop", "Rock": "rock", "Electro": "electronic",
    "Jazz": "jazz", "Indie": "indie", "R&B": "r-n-b", "Rap": "hip-hop"
}


# -----------------------
# AUTH & CLIENT
# -----------------------
def get_token():
    auth_manager = SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=SPOTIFY_REDIRECT_URI,
        scope=SCOPE,
        cache_handler=spotipy.cache_handler.MemoryCacheHandler()
    )

    qp = st.query_params
    code = qp.get("code")

    if "token_info" not in st.session_state:
        st.session_state.token_info = None

    if not st.session_state.token_info:
        if code:
            token_info = auth_manager.get_access_token(code)
            st.session_state.token_info = token_info
            st.query_params.clear()
            st.rerun()
        else:
            st.info("Please log in.")
            st.link_button("Log in", auth_manager.get_authorize_url())
            st.stop()

    return st.session_state.token_info['access_token']


# -----------------------
# APP
# -----------------------
st.title("⚡ Music Matcher (Direct HTTP)")

try:
    token = get_token()
    st.success("Authenticated!")
except Exception as e:
    st.error(f"Login failed: {e}")
    st.stop()

genre = st.selectbox("Genre", list(GENRE_MAP.keys()))
bpm = st.slider("BPM", 60, 200, 120)

if st.button("Generate"):
    seed = GENRE_MAP[genre]

    st.write(f"Fetching: Seed={seed}, BPM={bpm}")

    # DIRECT REQUEST (Bypassing Spotipy)
    headers = {
        "Authorization": f"Bearer {token}"
    }
    params = {
        "limit": 10,
        "seed_genres": seed,
        "target_tempo": bpm,
        "market": "US"  # Force US market to be safe
    }

    try:
        r = requests.get("https://api.spotify.com/v1/recommendations", headers=headers, params=params)

        if r.status_code == 200:
            data = r.json()
            tracks = data.get("tracks", [])

            if tracks:
                st.success(f"✅ Found {len(tracks)} tracks!")
                df = pd.DataFrame([{
                    "Name": t['name'],
                    "Artist": t['artists'][0]['name'],
                    "Link": t['external_urls']['spotify']
                } for t in tracks])
                st.dataframe(df)
            else:
                st.warning("⚠️ Success (200 OK), but Spotify returned 0 tracks.")
        else:
            st.error(f"❌ API Error {r.status_code}")
            st.json(r.json())  # Print the exact error from Spotify

    except Exception as e:
        st.error(f"Network Error: {e}")
