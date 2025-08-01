"""
recommender.py
Hybrid Music Recommendation System Logic
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

# =========================================================
# 1. LOAD DATA
# =========================================================
def load_data(file_path="data/music.csv"):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=["track_name", "artist_name"])
    df.reset_index(drop=True, inplace=True)
    return df

# =========================================================
# 2. PREPROCESS FOR CONTENT-BASED
# =========================================================
def preprocess_features(df):
    feature_cols = [
        "acousticness", "danceability", "energy",
        "instrumentalness", "liveness", "loudness",
        "speechiness", "tempo", "valence"
    ]
    df = df.dropna(subset=feature_cols)
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df, feature_cols

# =========================================================
# 3. GENERATE PLAYLISTS USING K-MEANS CLUSTERING
# =========================================================
def generate_playlists(df, feature_cols, num_clusters=50):
    """
    Generate playlists using K-Means clustering on track features.
    Each cluster = one playlist.
    """
    feature_matrix = df[feature_cols].values
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(feature_matrix)

    # Assign cluster as playlist ID
    df['cluster'] = cluster_labels

    playlists = []
    for cluster_id in range(num_clusters):
        track_indices = df[df['cluster'] == cluster_id].index.tolist()
        for tid in track_indices:
            playlists.append((cluster_id, tid))

    playlist_df = pd.DataFrame(playlists, columns=["playlist_id", "track_idx"])
    return playlist_df, num_clusters

# =========================================================
# 4. INTERACTION MATRIX FOR ALS
# =========================================================
def build_interaction_matrix(playlist_df, num_playlists, num_tracks):
    row = playlist_df["playlist_id"].values
    col = playlist_df["track_idx"].values
    data = np.ones(len(playlist_df))
    return csr_matrix((data, (row, col)), shape=(num_playlists, num_tracks))

# =========================================================
# 5. TRAIN ALS MODEL
# =========================================================
def train_als(interaction_matrix, factors=50, iterations=10, regularization=0.01):
    model = AlternatingLeastSquares(factors=factors,
                                    iterations=iterations,
                                    regularization=regularization)
    model.fit(interaction_matrix)
    return model

# =========================================================
# 6. CONTENT SIMILARITY
# =========================================================
def get_track_similarity_vector(track_idx, feature_matrix, top_n=50):
    track_features = feature_matrix[track_idx].reshape(1, -1)
    sims = cosine_similarity(track_features, feature_matrix)[0]
    top_indices = sims.argsort()[::-1][1:top_n+1]
    return top_indices

def get_similar_tracks(track_name, df, feature_matrix, top_n=10):
    matches = df[df["track_name"].str.lower() == track_name.lower()]
    if matches.empty:
        return []
    idx = matches.index[0]
    top_indices = get_track_similarity_vector(idx, feature_matrix, top_n)
    return df.iloc[top_indices][["track_name", "artist_name", "genre"]].to_dict("records")

# =========================================================
# 7. COLLABORATIVE RECOMMENDATIONS
# =========================================================
def get_playlist_recommendations(playlist_id, als_model, interaction_matrix, df, top_n=10):
    recs = als_model.recommend(playlist_id, interaction_matrix[playlist_id],
                                N=top_n, filter_already_liked_items=True)

    # Normalize: recs is array (N,2): [track_id, score]
    track_indices = [int(r[0]) for r in recs]
    return df.iloc[track_indices][["track_name", "artist_name", "genre"]].to_dict("records"), track_indices

# =========================================================
# 8. HYBRID RECOMMENDATIONS
# =========================================================
def get_hybrid_recommendations(playlist_id, als_model, interaction_matrix, df, feature_matrix, alpha=0.5, top_n=10):
    collab_recs = als_model.recommend(playlist_id, interaction_matrix[playlist_id],
                                      N=top_n*3, filter_already_liked_items=True)
    collab_indices = [int(r[0]) for r in collab_recs]

    playlist_tracks = interaction_matrix[playlist_id].indices
    playlist_vector = feature_matrix[playlist_tracks].mean(axis=0).reshape(1, -1)
    sims = cosine_similarity(playlist_vector, feature_matrix)[0]
    content_indices = sims.argsort()[::-1][:top_n*3].tolist()

    combined = list(set(collab_indices + content_indices))
    scores = {}
    for idx in combined:
        collab_score = 1 if idx in collab_indices else 0
        content_score = sims[idx] if idx < len(sims) else 0
        scores[idx] = alpha * collab_score + (1 - alpha) * content_score

    sorted_tracks = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_indices = [i for i, _ in sorted_tracks]
    return df.iloc[top_indices][["track_name", "artist_name", "genre"]].to_dict("records"), top_indices

# =========================================================
# 9. FIND MAJORITY PLAYLIST FOR RECOMMENDED TRACKS
# =========================================================
def get_playlist_majority(recommended_indices, playlist_df):
    playlist_counts = playlist_df[playlist_df['track_idx'].isin(recommended_indices)]['playlist_id'].value_counts()
    if not playlist_counts.empty:
        return int(playlist_counts.index[0])
    return None
