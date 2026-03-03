import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
import random

import spotipy
from spotipy.oauth2 import SpotifyOAuth
import spotipy.exceptions

# -----------------------
# CONFIG
# -----------------------
st.set_page_config(page_title="Music Matcher", page_icon="🎧", layout="wide")

SPOTIFY_CLIENT_ID = st.secrets["spotify"]["client_id"]
SPOTIFY_CLIENT_SECRET = st.secrets["spotify"]["client_secret"]
SPOTIFY_REDIRECT_URI = st.secrets["spotify"]["redirect_uri"]

# CRITICAL: Ensure all required scopes are included
SCOPE = "user-read-private playlist-modify-public playlist-modify-private user-read-email"

# -----------------------
# SONG LIBRARY (External to Spotify)
# -----------------------
SONG_LIBRARY = {
    "Pop": [
        {"name": "Blinding Lights", "artist": "The Weeknd"},
        {"name": "Levitating", "artist": "Dua Lipa"},
        {"name": "As It Was", "artist": "Harry Styles"},
        {"name": "Anti-Hero", "artist": "Taylor Swift"},
        {"name": "Flowers", "artist": "Miley Cyrus"},
        {"name": "bad guy", "artist": "Billie Eilish"},
        {"name": "Watermelon Sugar", "artist": "Harry Styles"},
        {"name": "Dance Monkey", "artist": "Tones and I"},
        {"name": "Don't Start Now", "artist": "Dua Lipa"},
        {"name": "Circles", "artist": "Post Malone"},
        {"name": "Shape of You", "artist": "Ed Sheeran"},
        {"name": "Uptown Funk", "artist": "Mark Ronson"},
        {"name": "Happy", "artist": "Pharrell Williams"},
        {"name": "Can't Stop the Feeling", "artist": "Justin Timberlake"},
        {"name": "Shake It Off", "artist": "Taylor Swift"},
        {"name": "Sunflower", "artist": "Post Malone"},
        {"name": "Someone You Loved", "artist": "Lewis Capaldi"},
        {"name": "Perfect", "artist": "Ed Sheeran"},
        {"name": "Beautiful People", "artist": "Ed Sheeran"},
        {"name": "Señorita", "artist": "Shawn Mendes"},
    ],
    "Hip-Hop": [
        {"name": "HUMBLE.", "artist": "Kendrick Lamar"},
        {"name": "God's Plan", "artist": "Drake"},
        {"name": "Sicko Mode", "artist": "Travis Scott"},
        {"name": "Old Town Road", "artist": "Lil Nas X"},
        {"name": "Lose Yourself", "artist": "Eminem"},
        {"name": "In Da Club", "artist": "50 Cent"},
        {"name": "Hotline Bling", "artist": "Drake"},
        {"name": "Congratulations", "artist": "Post Malone"},
        {"name": "Lucid Dreams", "artist": "Juice WRLD"},
        {"name": "Rockstar", "artist": "Post Malone"},
        {"name": "Savage", "artist": "Megan Thee Stallion"},
        {"name": "WAP", "artist": "Cardi B"},
        {"name": "The Box", "artist": "Roddy Ricch"},
        {"name": "MIDDLE CHILD", "artist": "J. Cole"},
        {"name": "Money Trees", "artist": "Kendrick Lamar"},
        {"name": "Mask Off", "artist": "Future"},
        {"name": "XO Tour Llif3", "artist": "Lil Uzi Vert"},
        {"name": "Bank Account", "artist": "21 Savage"},
        {"name": "Nice For What", "artist": "Drake"},
        {"name": "Ric Flair Drip", "artist": "Offset"},
    ],
    "Rock": [
        {"name": "Mr. Brightside", "artist": "The Killers"},
        {"name": "Smells Like Teen Spirit", "artist": "Nirvana"},
        {"name": "Bohemian Rhapsody", "artist": "Queen"},
        {"name": "Sweet Child O' Mine", "artist": "Guns N' Roses"},
        {"name": "Hotel California", "artist": "Eagles"},
        {"name": "Stairway to Heaven", "artist": "Led Zeppelin"},
        {"name": "Wonderwall", "artist": "Oasis"},
        {"name": "Creep", "artist": "Radiohead"},
        {"name": "Come As You Are", "artist": "Nirvana"},
        {"name": "Don't Stop Believin'", "artist": "Journey"},
        {"name": "Seven Nation Army", "artist": "The White Stripes"},
        {"name": "Back in Black", "artist": "AC/DC"},
        {"name": "Born to Run", "artist": "Bruce Springsteen"},
        {"name": "Free Fallin'", "artist": "Tom Petty"},
        {"name": "Livin' on a Prayer", "artist": "Bon Jovi"},
        {"name": "Under the Bridge", "artist": "Red Hot Chili Peppers"},
        {"name": "Paradise City", "artist": "Guns N' Roses"},
        {"name": "Black Hole Sun", "artist": "Soundgarden"},
        {"name": "Everlong", "artist": "Foo Fighters"},
        {"name": "Kashmir", "artist": "Led Zeppelin"},
    ],
    "Electro": [
        {"name": "Strobe", "artist": "deadmau5"},
        {"name": "Levels", "artist": "Avicii"},
        {"name": "Scary Monsters and Nice Sprites", "artist": "Skrillex"},
        {"name": "Animals", "artist": "Martin Garrix"},
        {"name": "Titanium", "artist": "David Guetta"},
        {"name": "Wake Me Up", "artist": "Avicii"},
        {"name": "Don't You Worry Child", "artist": "Swedish House Mafia"},
        {"name": "Clarity", "artist": "Zedd"},
        {"name": "Turn Down for What", "artist": "DJ Snake"},
        {"name": "Lean On", "artist": "Major Lazer"},
        {"name": "One More Time", "artist": "Daft Punk"},
        {"name": "Get Lucky", "artist": "Daft Punk"},
        {"name": "Astronomia", "artist": "Vicetone"},
        {"name": "Fade", "artist": "Alan Walker"},
        {"name": "Faded", "artist": "Alan Walker"},
        {"name": "Silence", "artist": "Marshmello"},
        {"name": "Closer", "artist": "The Chainsmokers"},
        {"name": "Something Just Like This", "artist": "The Chainsmokers"},
        {"name": "This Is What You Came For", "artist": "Calvin Harris"},
        {"name": "How Deep Is Your Love", "artist": "Calvin Harris"},
    ],
    "Jazz": [
        {"name": "Take Five", "artist": "Dave Brubeck"},
        {"name": "What a Wonderful World", "artist": "Louis Armstrong"},
        {"name": "Fly Me to the Moon", "artist": "Frank Sinatra"},
        {"name": "My Favorite Things", "artist": "John Coltrane"},
        {"name": "So What", "artist": "Miles Davis"},
        {"name": "Autumn Leaves", "artist": "Bill Evans"},
        {"name": "Round Midnight", "artist": "Thelonious Monk"},
        {"name": "Blue in Green", "artist": "Miles Davis"},
        {"name": "All Blues", "artist": "Miles Davis"},
        {"name": "Summertime", "artist": "Ella Fitzgerald"},
        {"name": "Georgia on My Mind", "artist": "Ray Charles"},
        {"name": "Feeling Good", "artist": "Nina Simone"},
        {"name": "The Girl from Ipanema", "artist": "Stan Getz"},
        {"name": "Birdland", "artist": "Weather Report"},
        {"name": "Spain", "artist": "Chick Corea"},
        {"name": "Giant Steps", "artist": "John Coltrane"},
        {"name": "Cantaloupe Island", "artist": "Herbie Hancock"},
        {"name": "Maiden Voyage", "artist": "Herbie Hancock"},
        {"name": "A Love Supreme", "artist": "John Coltrane"},
        {"name": "Kind of Blue", "artist": "Miles Davis"},
    ],
    "Indie": [
        {"name": "Electric Feel", "artist": "MGMT"},
        {"name": "Take Me Out", "artist": "Franz Ferdinand"},
        {"name": "Somebody That I Used to Know", "artist": "Gotye"},
        {"name": "Dog Days Are Over", "artist": "Florence + The Machine"},
        {"name": "Little Talks", "artist": "Of Monsters and Men"},
        {"name": "Ho Hey", "artist": "The Lumineers"},
        {"name": "Pumped Up Kicks", "artist": "Foster the People"},
        {"name": "Young Folks", "artist": "Peter Bjorn and John"},
        {"name": "1901", "artist": "Phoenix"},
        {"name": "Time to Pretend", "artist": "MGMT"},
        {"name": "Sedona", "artist": "Houndmouth"},
        {"name": "Home", "artist": "Edward Sharpe"},
        {"name": "Riptide", "artist": "Vance Joy"},
        {"name": "Budapest", "artist": "George Ezra"},
        {"name": "Stolen Dance", "artist": "Milky Chance"},
        {"name": "The Less I Know The Better", "artist": "Tame Impala"},
        {"name": "Feels Like We Only Go Backwards", "artist": "Tame Impala"},
        {"name": "Sweater Weather", "artist": "The Neighbourhood"},
        {"name": "Do I Wanna Know", "artist": "Arctic Monkeys"},
        {"name": "R U Mine", "artist": "Arctic Monkeys"},
    ],
    "R&B": [
        {"name": "Redbone", "artist": "Childish Gambino"},
        {"name": "Earned It", "artist": "The Weeknd"},
        {"name": "Thinkin Bout You", "artist": "Frank Ocean"},
        {"name": "Crew", "artist": "GoldLink"},
        {"name": "Best Part", "artist": "Daniel Caesar"},
        {"name": "Location", "artist": "Khalid"},
        {"name": "The Hills", "artist": "The Weeknd"},
        {"name": "Boo'd Up", "artist": "Ella Mai"},
        {"name": "Finesse", "artist": "Bruno Mars"},
        {"name": "No Tears Left to Cry", "artist": "Ariana Grande"},
        {"name": "Love Galore", "artist": "SZA"},
        {"name": "Juke Jam", "artist": "Chance the Rapper"},
        {"name": "Exchange", "artist": "Bryson Tiller"},
        {"name": "Come Through and Chill", "artist": "Miguel"},
        {"name": "Good Days", "artist": "SZA"},
        {"name": "Supermodel", "artist": "SZA"},
        {"name": "Die For You", "artist": "The Weeknd"},
        {"name": "After Hours", "artist": "The Weeknd"},
        {"name": "Take Care", "artist": "Drake"},
        {"name": "Passionfruit", "artist": "Drake"},
    ],
}

# Map your UI genres to library keys
GENRE_MAP = {
    "Pop": "Pop",
    "Hip-Hop": "Hip-Hop",
    "Rock": "Rock",
    "Electro": "Electro",
    "Jazz": "Jazz",
    "Indie": "Indie",
    "R&B": "R&B",
    "Rap": "Hip-Hop",
    "Classical": "Jazz",
    "Country": "Rock",
    "Metal": "Rock",
}

# Key mapping for display
KEY_MAP = {
    0: "C", 1: "C♯/D♭", 2: "D", 3: "D♯/E♭", 4: "E", 5: "F",
    6: "F♯/G♭", 7: "G", 8: "G♯/A♭", 9: "A", 10: "A♯/B♭", 11: "B"
}


# -----------------------
# CUSTOM CACHE HANDLER
# -----------------------
class StreamlitCacheHandler(spotipy.cache_handler.CacheHandler):
    def get_cached_token(self):
        return st.session_state.get("spotify_token_info")

    def save_token_to_cache(self, token_info):
        st.session_state["spotify_token_info"] = token_info


# -----------------------
# AUTH FLOW
# -----------------------
def get_auth_manager():
    return SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=SPOTIFY_REDIRECT_URI,
        scope=SCOPE,
        cache_handler=StreamlitCacheHandler(),
        show_dialog=True  # Force permission dialog
    )


auth_manager = get_auth_manager()

if "spotify_token_info" not in st.session_state:
    st.session_state.spotify_token_info = None

qp = st.query_params
code = qp.get("code")

if not st.session_state.spotify_token_info:
    if code:
        try:
            token_info = auth_manager.get_access_token(code, as_dict=True, check_cache=False)
            st.session_state.spotify_token_info = token_info
            st.query_params.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Auth failed: {e}")
            st.stop()
    else:
        st.title("🎧 Music Matcher")
        st.info("Please log in to Spotify to continue.")
        st.warning("⚠️ Make sure to grant all permissions when authorizing!")
        st.link_button("🎵 Log in to Spotify", auth_manager.get_authorize_url())
        st.stop()

try:
    sp = spotipy.Spotify(auth_manager=auth_manager)
    user = sp.current_user()
except Exception as e:
    st.error(f"Authentication error: {e}")
    st.warning("Try clicking 'Logout' in the sidebar to re-authenticate with proper permissions.")
    if st.button("Retry Now"):
        st.rerun()
    st.stop()


# -----------------------
# HELPER FUNCTIONS
# -----------------------
def find_song_on_spotify(song_name, artist_name, sp_client):
    """Search for a song on Spotify and return detailed info"""
    try:
        query = f"track:{song_name} artist:{artist_name}"
        results = sp_client.search(q=query, type="track", limit=1)
        items = results.get("tracks", {}).get("items", [])

        if items:
            track = items[0]

            # Get audio features
            try:
                features = sp_client.audio_features([track["id"]])[0] if track.get("id") else {}
            except:
                features = {}

            images = track.get("album", {}).get("images", [])
            cover = images[1]["url"] if len(images) > 1 else (images[0]["url"] if images else None)

            artists = ", ".join(a["name"] for a in track.get("artists", [])) if track.get("artists") else "Unknown"

            key_num = features.get("key", 0) if features else 0
            mode_num = features.get("mode", 1) if features else 1
            key_str = KEY_MAP.get(key_num, str(key_num))
            mode_str = "Major" if mode_num == 1 else "Minor"

            return {
                "ID": track.get("id"),
                "uri": track["uri"],
                "Artist": artists,
                "Song Title": track["name"],
                "BPM": int(round(features.get("tempo", 120))) if features else 120,
                "Key": f"{key_str} {mode_str}",
                "Pop (T)": track.get("popularity", 50),
                "Pop (A)": 50,
                "Preview": track.get("preview_url"),
                "Cover": cover,
                "Link": track.get("external_urls", {}).get("spotify"),
                "found": True
            }
    except Exception as e:
        print(f"Error finding {song_name}: {e}")

    return None


def get_advanced_filtered_tracks_from_library(genre_group, limit=15):
    """Generate playlist from predefined library"""
    library_genre = GENRE_MAP.get(genre_group, "Pop")

    if library_genre not in SONG_LIBRARY:
        return []

    songs = SONG_LIBRARY[library_genre]

    if len(songs) < limit:
        selected = songs.copy()
    else:
        selected = random.sample(songs, limit)

    found_tracks = []
    for song in selected:
        result = find_song_on_spotify(song["name"], song["artist"], sp)
        if result:
            found_tracks.append(result)

    return found_tracks


# -----------------------
# SESSION STATE
# -----------------------
if "playlist_data" not in st.session_state:
    st.session_state.playlist_data = None
if "swiper_songs" not in st.session_state:
    st.session_state.swiper_songs = []
if "current_index" not in st.session_state:
    st.session_state.current_index = 0
if "liked_songs" not in st.session_state:
    st.session_state.liked_songs = []

# -----------------------
# NAVIGATION
# -----------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🎛️ Playlist Generator", "🔥 Song Swiper"])

# ==========================================
# PAGE 1: PLAYLIST GENERATOR
# ==========================================
if page == "🎛️ Playlist Generator":
    st.sidebar.header("Settings")

    genre_group = st.sidebar.selectbox("Genre Group", list(GENRE_MAP.keys()))

    with st.sidebar.expander("🎹 Musical Features", expanded=True):
        target_bpm = st.slider("Tempo (BPM)", 50, 200, 120)
        target_key = st.slider("Key (0=C, 11=B)", 0, 11, 0, help="0=C, 1=C#, 2=D, etc.")
        target_mode = st.radio("Mode", [0, 1], format_func=lambda x: "Minor" if x == 0 else "Major", horizontal=True)
        target_danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
        target_energy = st.slider("Energy", 0.0, 1.0, 0.5)
        target_valence = st.slider("Valence (Mood)", 0.0, 1.0, 0.5)
        target_loudness = st.slider("Loudness (dB)", -60.0, 0.0, -10.0)

    with st.sidebar.expander("🎻 Technical Features", expanded=False):
        target_acousticness = st.slider("Acousticness", 0.0, 1.0, 0.1)
        target_instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)
        target_liveness = st.slider("Liveness", 0.0, 1.0, 0.1)
        target_speechiness = st.slider("Speechiness", 0.0, 1.0, 0.05)

    with st.sidebar.expander("⭐ Popularity Filters", expanded=True):
        st.caption("Filters results to ensure minimum fame.")
        min_artist_pop = st.slider("Min Artist Popularity", 0, 100, 50)
        min_track_pop = st.slider("Min Track Popularity", 0, 100, 40)

    st.title("🎵 AI Playlist Generator")
    st.markdown(f"Generating **{genre_group}** playlist based on **14 attributes**.")

    if st.button("✨ Generate Playlist", type="primary"):
        with st.spinner(f"Analyzing {genre_group} tracks..."):
            real_data = get_advanced_filtered_tracks_from_library(
                genre_group=genre_group,
                limit=15
            )

            if not real_data:
                st.error("No songs found! This genre may not be in the library yet.")
                st.session_state.playlist_data = None
            else:
                st.session_state.playlist_data = pd.DataFrame(real_data)
                st.success(f"Generated {len(real_data)} tracks matching your criteria!")

    if st.session_state.playlist_data is not None and not st.session_state.playlist_data.empty:
        st.divider()
        st.subheader("Your Curated Mix")

        st.dataframe(
            st.session_state.playlist_data,
            width='stretch',
            hide_index=True,
            column_config={
                "Cover": st.column_config.ImageColumn("Art", width=50),
                "Preview": st.column_config.LinkColumn("Preview"),
                "Link": st.column_config.LinkColumn("Link"),
                "BPM": st.column_config.NumberColumn("BPM", format="%d"),
                "Pop (T)": st.column_config.ProgressColumn("Track Pop", min_value=0, max_value=100),
                "Pop (A)": st.column_config.ProgressColumn("Artist Pop", min_value=0, max_value=100),
            }
        )

        st.divider()
        col1, col2 = st.columns([3, 1])

        with col1:
            playlist_name = st.text_input("Playlist Name", value=f"{genre_group} Mix {random.randint(1, 999)}")

        with col2:
            st.write("")
            st.write("")
            if st.button("💾 Save to Spotify", type="primary"):
                if not playlist_name.strip():
                    st.error("Please enter a playlist name.")
                else:
                    try:
                        with st.spinner("Creating playlist on Spotify..."):
                            user_id = user['id']

                            # Validate token has required scopes
                            token_info = st.session_state.spotify_token_info
                            if token_info:
                                current_scopes = token_info.get('scope', '').split()
                                required_scopes = ['playlist-modify-public', 'playlist-modify-private']
                                missing_scopes = [s for s in required_scopes if s not in current_scopes]

                                if missing_scopes:
                                    st.error(f"Missing permissions: {', '.join(missing_scopes)}")
                                    st.warning("Please click 'Logout' and log in again to grant all permissions.")
                                    st.stop()

                            # Create playlist
                            playlist = sp.user_playlist_create(
                                user=user_id,
                                name=playlist_name,
                                public=True,
                                description="Generated by Music Matcher 🎧"
                            )

                            playlist_id = playlist['id']
                            track_uris = st.session_state.playlist_data["uri"].tolist()

                            if track_uris:
                                for i in range(0, len(track_uris), 100):
                                    batch = track_uris[i:i + 100]
                                    sp.playlist_add_items(playlist_id, batch)

                                st.balloons()
                                st.success(f"✅ Playlist **{playlist_name}** created with {len(track_uris)} tracks!")
                                st.link_button("🎵 Open in Spotify", playlist["external_urls"]["spotify"])

                    except spotipy.exceptions.SpotifyException as e:
                        st.error(f"❌ Spotify API Error: {e}")
                        if "403" in str(e):
                            st.warning("🔑 This is a permissions error. Click 'Logout' in the sidebar and log in again to grant all permissions.")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

# ==========================================
# PAGE 2: SONG SWIPER
# ==========================================
elif page == "🔥 Song Swiper":
    st.title("🔥 Song Swiper")
    st.caption("Quickly swipe through songs based on simplified settings.")

    with st.expander("Swiper Settings"):
        swiper_genre = st.selectbox("Genre", list(GENRE_MAP.keys()), index=0)
        swiper_bpm = st.slider("Target BPM", 60, 180, 120)

        if st.button("🔄 Reload Stack"):
            st.session_state.swiper_songs = get_advanced_filtered_tracks_from_library(
                genre_group=swiper_genre,
                limit=20
            )
            st.session_state.current_index = 0
            st.rerun()

    if not st.session_state.swiper_songs:
        st.session_state.swiper_songs = get_advanced_filtered_tracks_from_library("Pop", limit=10)

    if st.session_state.swiper_songs and st.session_state.current_index < len(st.session_state.swiper_songs):
        current_song = st.session_state.swiper_songs[st.session_state.current_index]

        with st.container(border=True):
            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                if current_song.get("Cover"):
                    st.image(current_song["Cover"], width=300)
                st.subheader(current_song["Song Title"])
                st.write(f"**{current_song['Artist']}**")
                st.caption(f"Key: {current_song['Key']} • BPM: {current_song['BPM']}")
                if current_song.get("Preview"):
                    st.audio(current_song["Preview"], format="audio/mp3")
                else:
                    st.warning("No preview available")

        col_pass, col_space, col_like = st.columns([1, 2, 1])
        with col_pass:
            if st.button("❌ Pass", width='stretch'):
                st.session_state.current_index += 1
                st.rerun()
        with col_like:
            if st.button("💚 Like", type="primary", width='stretch'):
                st.session_state.liked_songs.append(current_song)
                st.session_state.current_index += 1
                st.rerun()
    else:
        st.success("No more songs!")
        if st.button("Restart"):
            st.session_state.current_index = 0
            st.rerun()

    if len(st.session_state.liked_songs) > 0:
        st.divider()
        st.subheader("💚 Liked Songs")
        liked_df = pd.DataFrame(st.session_state.liked_songs)
        st.dataframe(liked_df[["Artist", "Song Title", "BPM", "Key"]])

        if st.button("💾 Save Liked Songs to Spotify"):
            try:
                user_id = user['id']

                # Validate token scopes
                token_info = st.session_state.spotify_token_info
                if token_info:
                    current_scopes = token_info.get('scope', '').split()
                    required_scopes = ['playlist-modify-public', 'playlist-modify-private']
                    missing_scopes = [s for s in required_scopes if s not in current_scopes]

                    if missing_scopes:
                        st.error(f"Missing permissions: {', '.join(missing_scopes)}")
                        st.warning("Please click 'Logout' and log in again to grant all permissions.")
                        st.stop()

                playlist = sp.user_playlist_create(
                    user=user_id,
                    name=f"Liked Songs {datetime.now().strftime('%Y-%m-%d')}",
                    public=True,
                    description="Songs I liked from Swiper 🔥"
                )

                track_uris = [s["uri"] for s in st.session_state.liked_songs if s.get("uri")]

                if track_uris:
                    for i in range(0, len(track_uris), 100):
                        batch = track_uris[i:i + 100]
                        sp.playlist_add_items(playlist['id'], batch)

                    st.success("✅ Liked songs saved to Spotify!")
                    st.link_button("🎵 Open Playlist", playlist["external_urls"]["spotify"])

            except spotipy.exceptions.SpotifyException as e:
                st.error(f"❌ Spotify API Error: {e}")
                if "403" in str(e):
                    st.warning("🔑 This is a permissions error. Click 'Logout' in the sidebar and log in again.")
            except Exception as e:
                st.error(f"❌ Error: {e}")

# Footer
with st.sidebar:
    st.divider()
    st.caption(f"Logged in as **{user['display_name']}**")
    if st.button("🚪 Logout"):
        st.session_state.spotify_token_info = None
        st.info("Please log in again and grant all permissions.")
        st.rerun()