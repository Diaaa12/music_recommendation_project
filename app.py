"""
app.py
Streamlit UI for Music Recommendation
"""

import streamlit as st
import pandas as pd
from recommender import (
    load_data, preprocess_features, generate_playlists, build_interaction_matrix,
    train_als, get_similar_tracks, get_playlist_recommendations, get_hybrid_recommendations,
    get_playlist_majority
)

# ===========================
# Title
# ===========================
st.title("🎵 Hybrid Music Recommendation System")

# ===========================
# Load Data and Models
# ===========================
@st.cache_data
def prepare_data():
    df = load_data()
    df, feature_cols = preprocess_features(df)
    feature_matrix = df[feature_cols].values
    playlist_df, num_clusters = generate_playlists(df, feature_cols, num_clusters=50)
    interaction_matrix = build_interaction_matrix(playlist_df, num_clusters, len(df))

    als_model = train_als(interaction_matrix)
    return df, feature_matrix, playlist_df, interaction_matrix, als_model

df, feature_matrix, playlist_df, interaction_matrix, als_model = prepare_data()

# Sidebar for Recommendation Type
option = st.sidebar.radio("Choose Recommendation Type", ("Content-Based", "Collaborative", "Hybrid"))

# ===========================
# Track Search with Dropdown
# ===========================
track_names = df["track_name"].unique().tolist()
search_query = st.multiselect(
    "Search and select a track:",
    options=track_names,
    max_selections=1,
    placeholder="Start typing to search..."
)

if search_query:
    selected_track = search_query[0]

    if option == "Content-Based":
        st.subheader(f"Similar Tracks to: {selected_track}")
        recs = get_similar_tracks(selected_track, df, feature_matrix, top_n=10)
        if recs:
            st.table(pd.DataFrame(recs))
        else:
            st.error("Track not found!")

    else:
        matching_playlists = playlist_df.groupby("playlist_id").filter(
            lambda x: any(selected_track.lower() in df.iloc[i]["track_name"].lower() for i in x["track_idx"])
        )
        playlist_ids = matching_playlists["playlist_id"].unique()

        if len(playlist_ids) == 0:
            st.error("No playlists found (should not happen with guaranteed coverage).")
        else:
            st.write("### Select a Playlist:")
            for pid in playlist_ids[:5]:
                tracks_in_playlist = playlist_df[playlist_df["playlist_id"] == pid]["track_idx"]
                playlist_tracks = df.iloc[tracks_in_playlist][["track_name", "artist_name"]].head(10)
                with st.expander(f"Playlist ID {pid} - Sample Tracks"):
                    st.table(playlist_tracks)

            selected_pid = st.selectbox("Choose Playlist ID for Recommendations", playlist_ids)

            if st.button("Get Recommendations"):
                if option == "Collaborative":
                    recs, indices = get_playlist_recommendations(selected_pid, als_model, interaction_matrix, df, top_n=10)
                else:
                    recs, indices = get_hybrid_recommendations(selected_pid, als_model, interaction_matrix, df, feature_matrix, alpha=0.5, top_n=10)

                st.subheader("Recommended Tracks:")
                st.table(pd.DataFrame(recs))

                # Show most relevant playlist
                playlist_id_majority = get_playlist_majority(indices, playlist_df)
                if playlist_id_majority is not None:
                    st.success(f"🎶 Recommended Playlist: Playlist ID {playlist_id_majority}")
