"""
PlaylistPulse — Step 1: Generate synthetic data.
Run:  python scripts/generate_data.py   (project root se)
Creates: data/tracks.csv, data/listeners.csv
"""
import os

import numpy as np
import pandas as pd

# Kisi bhi folder se run karo, path sahi rahega
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

np.random.seed(42)  # same data har baar (reproducible)
N_TRACKS = 5000
N_LISTENERS = 500

# ---------------- TRACKS ----------------
# Audio features (Spotify audio-features jaise realistic ranges)
danceability = np.random.beta(2, 1.5, N_TRACKS)                # 0-1
energy = np.random.beta(2, 2, N_TRACKS)                       # 0-1
tempo = np.clip(np.random.normal(120, 25, N_TRACKS), 60, 200)  # BPM
valence = np.random.beta(2, 2, N_TRACKS)                      # 0-1 (happiness)
acousticness = np.random.beta(1.5, 3, N_TRACKS)              # 0-1
speechiness = np.clip(np.random.beta(1, 8, N_TRACKS) * 0.6, 0, 0.5)
instrumentalness = np.random.beta(1, 10, N_TRACKS)            # 0-1

# Context features
playlist_position = np.random.randint(1, 51, N_TRACKS)        # 1-50
hour_of_day = np.random.randint(0, 24, N_TRACKS)             # 0-23
is_playlist_radio = np.random.randint(0, 2, N_TRACKS)        # 0/1

# History features
listener_avg_skip_rate = np.random.beta(2, 5, N_TRACKS)       # 0-1
days_since_release = np.random.randint(1, 2001, N_TRACKS)    # 1-2000

# TARGET 1: streams — heavy-tailed (lognormal).
# Catchy + energetic + happy + purane gaane = zyada streams. Noise realistic rakhta hai.
log_streams = (
    10.5
    + 1.5 * danceability
    + 1.2 * energy
    + 0.8 * valence
    - 0.6 * acousticness
    - 0.8 * speechiness
    + 0.6 * (days_since_release / 2000)
    + np.random.normal(0, 0.6, N_TRACKS)
)
streams = np.exp(log_streams)  # exp() = heavy-tailed: kuch hits, zyada tar flops

# TARGET 2: skip within 30s — features ka noisy logistic function
z = (
    -2.2
    + 1.6 * energy
    + 0.010 * (tempo - 120)
    + 2.0 * listener_avg_skip_rate
    + 0.8 * (playlist_position - 25) / 25
    + 0.4 * np.sin(2 * np.pi * hour_of_day / 24)
    - 1.0 * danceability
    + np.random.normal(0, 0.6, N_TRACKS)
)
skip_prob = 1 / (1 + np.exp(-z))
skipped_within_30s = (np.random.rand(N_TRACKS) < skip_prob).astype(int)

tracks = pd.DataFrame({
    "danceability": danceability,
    "energy": energy,
    "tempo": tempo,
    "valence": valence,
    "acousticness": acousticness,
    "speechiness": speechiness,
    "instrumentalness": instrumentalness,
    "playlist_position": playlist_position,
    "hour_of_day": hour_of_day,
    "is_playlist_radio": is_playlist_radio,
    "listener_avg_skip_rate": listener_avg_skip_rate,
    "days_since_release": days_since_release,
    "streams": streams.astype(int),
    "skipped_within_30s": skipped_within_30s,
})
tracks.to_csv(os.path.join(DATA_DIR, "tracks.csv"), index=False)
print(f"Saved {len(tracks)} tracks -> data/tracks.csv")
print(f"Skip rate: {skipped_within_30s.mean():.1%} (imbalanced, jaisa design tha)")

# ---------------- LISTENERS ----------------
# 4 taste archetypes + noise -> k-means inhe wapas dhoondh lega
ARCHETYPES = {
    # name: [avg_danceability, avg_energy, avg_acousticness, avg_tempo, avg_valence]
    "High-Energy Party":   [0.85, 0.85, 0.15, 135, 0.80],
    "Chill Acoustic Soul": [0.45, 0.25, 0.85,  95, 0.40],
    "Dance Pop Lover":     [0.80, 0.60, 0.30, 120, 0.75],
    "Focus Instrumental":  [0.30, 0.35, 0.50, 105, 0.25],
}
names = list(ARCHETYPES.keys())

rows = []
for _ in range(N_LISTENERS):
    base = ARCHETYPES[names[np.random.randint(len(names))]]
    rows.append([
        float(np.clip(base[0] + np.random.normal(0, 0.06), 0, 1)),
        float(np.clip(base[1] + np.random.normal(0, 0.06), 0, 1)),
        float(np.clip(base[2] + np.random.normal(0, 0.06), 0, 1)),
        float(np.clip(base[3] + np.random.normal(0, 10), 60, 200)),
        float(np.clip(base[4] + np.random.normal(0, 0.06), 0, 1)),
    ])

listeners = pd.DataFrame(rows, columns=[
    "avg_danceability", "avg_energy", "avg_acousticness", "avg_tempo", "avg_valence",
])
listeners.to_csv(os.path.join(DATA_DIR, "listeners.csv"), index=False)
print(f"Saved {len(listeners)} listeners -> data/listeners.csv")