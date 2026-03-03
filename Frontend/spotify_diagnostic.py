import spotipy
from spotipy.oauth2 import SpotifyOAuth
import streamlit as st

st.title("🔍 Spotify API Diagnostic Tool")

# Configuration
SPOTIFY_CLIENT_ID = st.secrets["spotify"]["client_id"]
SPOTIFY_CLIENT_SECRET = st.secrets["spotify"]["client_secret"]
SPOTIFY_REDIRECT_URI = st.secrets["spotify"]["redirect_uri"]

st.write("### Configuration Check")
st.write(f"**Client ID:** `{SPOTIFY_CLIENT_ID[:10]}...`")
st.write(f"**Redirect URI:** `{SPOTIFY_REDIRECT_URI}`")

# Test 1: Basic Auth
st.write("---")
st.write("### Test 1: Authentication")

scope = "user-read-private user-read-email"

try:
    auth_manager = SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=SPOTIFY_REDIRECT_URI,
        scope=scope,
        cache_path=".spotify_cache_test"
    )

    sp = spotipy.Spotify(auth_manager=auth_manager)
    user = sp.current_user()

    st.success("✅ Authentication successful!")
    st.json({
        "User ID": user.get('id'),
        "Display Name": user.get('display_name'),
        "Email": user.get('email'),
        "Country": user.get('country')
    })

    # Test 2: Token Info
    st.write("---")
    st.write("### Test 2: Token Scopes")
    token_info = auth_manager.get_cached_token()
    if token_info:
        scopes = token_info.get('scope', '').split()
        st.write("**Granted Scopes:**")
        for scope in scopes:
            st.write(f"- {scope}")

    # Test 3: Basic API Call
    st.write("---")
    st.write("### Test 3: Search API (No special permissions)")
    try:
        results = sp.search(q="The Weeknd", type="track", limit=1)
        if results and results.get('tracks', {}).get('items'):
            st.success("✅ Search API works!")
        else:
            st.error("❌ Search returned no results")
    except Exception as e:
        st.error(f"❌ Search API failed: {e}")

    # Test 4: Audio Features
    st.write("---")
    st.write("### Test 4: Audio Features API")
    try:
        # Use a known track ID
        features = sp.audio_features(['3n3Ppam7vgaVa1iaRUc9Lp'])
        if features:
            st.success("✅ Audio Features API works!")
        else:
            st.error("❌ Audio Features returned None")
    except Exception as e:
        st.error(f"❌ Audio Features failed: {e}")
        st.error("**This means your account is NOT authorized in the Spotify Developer Dashboard**")

    # Test 5: Playlist Creation (requires permissions)
    st.write("---")
    st.write("### Test 5: Playlist Creation")

    if st.button("Test Create Playlist"):
        try:
            # Need to re-auth with playlist permissions
            auth_manager_full = SpotifyOAuth(
                client_id=SPOTIFY_CLIENT_ID,
                client_secret=SPOTIFY_CLIENT_SECRET,
                redirect_uri=SPOTIFY_REDIRECT_URI,
                scope="user-read-private playlist-modify-public playlist-modify-private",
                cache_path=".spotify_cache_full"
            )
            sp_full = spotipy.Spotify(auth_manager=auth_manager_full)

            test_playlist = sp_full.user_playlist_create(
                user=user['id'],
                name="API Test Playlist (Delete Me)",
                public=True,
                description="Testing API access"
            )
            st.success(f"✅ Playlist created! ID: {test_playlist['id']}")
            st.write(f"[Open Playlist]({test_playlist['external_urls']['spotify']})")
        except Exception as e:
            st.error(f"❌ Playlist creation failed: {e}")

except Exception as e:
    st.error(f"❌ Authentication failed: {e}")
    st.error("**Check your Client ID and Secret in secrets.toml**")

st.write("---")
st.write("### 📋 Troubleshooting Checklist")

st.markdown("""
If you see ❌ errors above, check these in your Spotify Developer Dashboard:

1. **User Management** (CRITICAL):
   - Go to your app settings
   - Find "User Management" or "Users and Access"
   - Your Spotify email MUST be added there
   - App is in Development Mode (max 25 users)

2. **Redirect URI**:
   - Must be exactly: `http://127.0.0.1:8501`
   - Click "Add" then "Save"

3. **App Settings**:
   - Client ID and Secret match your secrets.toml
   - App is not deleted or suspended

4. **Account Match**:
   - Log in with the SAME email added to User Management
   - Case-sensitive name match
""")