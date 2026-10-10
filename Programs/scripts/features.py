"""
W11: Shared feature module — train.py AUR backend/main.py dono yahi
features banate hain. Isse training aur serving mein SAME transformation
guaranteed hoti hai (training-serving consistency). Ye hi hai "leakage-safe
shared pipeline" ka practical roop.
"""

RAW_TRACK_FEATURES = [
    "danceability", "energy", "tempo", "valence", "acousticness",
    "speechiness", "instrumentalness", "playlist_position", "hour_of_day",
    "is_playlist_radio", "listener_avg_skip_rate", "days_since_release",
]

# W4: engineered features — raw features se naye clues banaye
ENGINEERED_FEATURES = [
    "energy_dance", "tempo_norm", "is_night", "is_new_track",
    "position_early", "taste_segment",
]

# W9: taste ke 5 dimensions (tracks aur listeners dono mein hote hain)
TASTE_DIMS = ["danceability", "energy", "acousticness", "tempo", "valence"]

LISTENER_FEATURES = [
    "avg_danceability", "avg_energy", "avg_acousticness", "avg_tempo", "avg_valence",
]

ALL_TRACK_FEATURES = RAW_TRACK_FEATURES + ENGINEERED_FEATURES


def engineer_features(df, taste_kmeans=None):
    """
    Raw features -> engineered features. taste_kmeans diya to W9 ka
    taste_segment label bhi add hota hai (segment-as-feature).
    """
    df = df.copy()
    # W4: interactions + domain logic
    df["energy_dance"] = df["energy"] * df["danceability"]      # catchy combo
    df["tempo_norm"] = (df["tempo"] - 120) / 40                 # centered tempo
    df["is_night"] = ((df["hour_of_day"] >= 20) | (df["hour_of_day"] <= 5)).astype(int)
    df["is_new_track"] = (df["days_since_release"] <= 90).astype(int)
    df["position_early"] = (df["playlist_position"] <= 10).astype(int)

    # W9: taste cluster label ko wapas feature bana do
    if taste_kmeans is not None:
        df["taste_segment"] = taste_kmeans.predict(df[TASTE_DIMS])
    else:
        df["taste_segment"] = 0
    return df