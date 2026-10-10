"""
PlaylistPulse — FastAPI backend (RUBRIC VERSION — FIX #3).
FIX: taste_kmeans track-column names pe fit tha (danceability, energy...),
backend avg_* bhej raha tha -> sklearn feature-name mismatch.
Ab column names MODEL SE AUTO-DETECT hote hain (feature_names_in_).

Run:  python -m uvicorn backend.main:app --reload
"""
import json
import os
import subprocess
import threading
import time
from collections import deque

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
DATA_DIR = os.path.join(ROOT_DIR, "data")

STATE = {
    "regressor": None, "classifier": None, "kmeans": None, "taste_kmeans": None,
    "track_features": [], "listener_features": [], "cluster_info": {},
    "version": "unknown", "last_retrain": "never",
    "training_ref": {}, "recent": {},
    "requests_total": 0, "predictions_total": 0, "errors_total": 0,
    "latency_sum": 0.0, "retraining": False,
}
LOCK = threading.Lock()



def _taste_predict(f: pd.DataFrame) -> np.ndarray:
    km = STATE["taste_kmeans"]
    cols = list(getattr(km, "feature_names_in_", []))
    if not cols:  # numpy array pe fit hua tha — koi name check nahi
        cols = ["danceability", "energy", "acousticness", "tempo", "valence"]

    # avg_* naam ho toh track column pe map karo (dono cases handle)
    mapping = {}
    for c in cols:
        if c in f.columns:
            mapping[c] = c
        else:
            base = c.replace("avg_", "")
            if base in f.columns:
                mapping[c] = base
    sub = f[list(mapping.values())].copy()
    sub.columns = list(mapping.keys())
    return km.predict(sub)


# ⭐ W4: Feature engineering — single source of truth (prediction + PSI dono)
def engineer_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Raw 12 columns -> 18 engineered columns (vectorized)."""
    f = df.copy()
    f["energy_dance"] = f["energy"] * f["danceability"]
    f["tempo_norm"] = (f["tempo"] - 60) / 140
    f["is_night"] = ((f["hour_of_day"] >= 20) | (f["hour_of_day"] < 6)).astype(int)
    f["is_new_track"] = (f["days_since_release"] <= 30).astype(int)
    f["position_early"] = (f["playlist_position"] <= 5).astype(int)
    f["taste_segment"] = _taste_predict(f)   # ⭐ auto-detect column names
    return f


def load_models():
    """Startup pe + /retrain ke BAAD models load (hot reload)."""
    STATE["regressor"] = joblib.load(os.path.join(MODELS_DIR, "regressor.pkl"))
    STATE["classifier"] = joblib.load(os.path.join(MODELS_DIR, "classifier.pkl"))
    STATE["kmeans"] = joblib.load(os.path.join(MODELS_DIR, "kmeans.pkl"))
    STATE["taste_kmeans"] = joblib.load(os.path.join(MODELS_DIR, "taste_kmeans.pkl"))

    with open(os.path.join(MODELS_DIR, "features.json")) as fh:
        meta = json.load(fh)
    STATE["track_features"] = meta["track_features"]
    STATE["listener_features"] = meta["listener_features"]

    with open(os.path.join(MODELS_DIR, "cluster_labels.json")) as fh:
        STATE["cluster_info"] = json.load(fh)

    vpath = os.path.join(MODELS_DIR, "version.json")
    if os.path.exists(vpath):
        with open(vpath) as fh:
            v = json.load(fh)
        STATE["version"] = v.get("version", "unknown")
        STATE["last_retrain"] = v.get("trained_at", "unknown")

    # ---- W11: PSI reference — tracks.csv -> SAME engineering -> reference ----
    # Monitoring fail ho to bhi API chalegi (fail-safe).
    try:
        tracks = pd.read_csv(os.path.join(DATA_DIR, "tracks.csv"))
        engineered = engineer_frame(tracks)
        STATE["training_ref"] = {
            c: engineered[c].to_numpy()
            for c in STATE["track_features"] if c in engineered.columns
        }
        print(f"[psi] reference ban gayi: {len(STATE['training_ref'])} features")
    except Exception as e:
        print(f"[warn] PSI reference nahi bani — drift monitoring off: {e}")
        STATE["training_ref"] = {}
    STATE["recent"] = {c: deque(maxlen=200) for c in STATE["training_ref"]}


load_models()

app = FastAPI(title="PlaylistPulse API", version="2.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- Request schemas ----------------
class TrackFeatures(BaseModel):
    danceability: float = Field(0.5, ge=0, le=1)
    energy: float = Field(0.5, ge=0, le=1)
    tempo: float = Field(120, ge=60, le=200)
    valence: float = Field(0.5, ge=0, le=1)
    acousticness: float = Field(0.3, ge=0, le=1)
    speechiness: float = Field(0.05, ge=0, le=0.5)
    instrumentalness: float = Field(0.0, ge=0, le=1)
    playlist_position: int = Field(10, ge=1, le=50)
    hour_of_day: int = Field(18, ge=0, le=23)
    is_playlist_radio: int = Field(0, ge=0, le=1)
    listener_avg_skip_rate: float = Field(0.3, ge=0, le=1)
    days_since_release: int = Field(200, ge=1, le=2000)


class ListenerTaste(BaseModel):
    avg_danceability: float = Field(0.5, ge=0, le=1)
    avg_energy: float = Field(0.5, ge=0, le=1)
    avg_acousticness: float = Field(0.3, ge=0, le=1)
    avg_tempo: float = Field(115, ge=60, le=200)
    avg_valence: float = Field(0.5, ge=0, le=1)


# ---------------- W11: PSI — Population Stability Index ----------------
def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """< 0.1 stable | 0.1-0.25 moderate | > 0.25 major drift."""
    if len(actual) < 20:
        return 0.0
    edges = np.percentile(expected, np.linspace(0, 100, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.clip(np.histogram(expected, edges)[0] / len(expected), 1e-6, None)
    a = np.clip(np.histogram(actual, edges)[0] / len(actual), 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def check_drift():
    alerts = []
    for c, ref in STATE["training_ref"].items():
        live = np.array(STATE["recent"][c])
        score = psi(ref, live)
        if score > 0.25:
            alerts.append({"feature": c, "psi": round(score, 3), "level": "major"})
        elif score > 0.1:
            alerts.append({"feature": c, "psi": round(score, 3), "level": "moderate"})
    return alerts


# ---------------- Endpoints ----------------
@app.get("/health")
def health():
    return {"status": "ok", "model_version": STATE["version"]}


@app.get("/metrics")
def metrics():
    """W12: monitoring — requests, latency, version, drift."""
    alerts = check_drift()
    n = max(STATE["predictions_total"], 1)
    has_major = any(a["level"] == "major" for a in alerts)
    return {
        "requests_total": STATE["requests_total"],
        "predictions_total": STATE["predictions_total"],
        "errors_total": STATE["errors_total"],
        "avg_latency_ms": round(STATE["latency_sum"] / n * 1000, 2),
        "model_version": STATE["version"],
        "last_retrain": STATE["last_retrain"],
        "retraining_now": STATE["retraining"],
        "drift_status": (
            "MAJOR DRIFT — /retrain call karo" if has_major
            else ("moderate drift — dhyan do" if alerts else "stable")
        ),
        "drift_alerts": alerts,
    }


@app.post("/predict")
def predict(track: TrackFeatures):
    """Popularity + skip-likelihood endpoint (W12 core)."""
    t0 = time.time()
    with LOCK:
        STATE["requests_total"] += 1
    try:
        # 12 raw -> 18 engineered (SAME function jo PSI reference banati hai)
        row_full = engineer_frame(pd.DataFrame([track.model_dump()]))
        row = row_full[STATE["track_features"]]
        f = row_full.iloc[0].to_dict()

        # live values drift buffer mein (PSI ke liye)
        for c in STATE["recent"]:
            STATE["recent"][c].append(f[c])

        log_streams = float(STATE["regressor"].predict(row)[0])
        predicted_streams = int(round(np.expm1(log_streams)))   # log1p undo

        skip_prob = float(STATE["classifier"].predict_proba(row)[0][1])

        with LOCK:
            STATE["predictions_total"] += 1
            STATE["latency_sum"] += time.time() - t0

        return {
            "predicted_streams": predicted_streams,
            "skip_probability": round(skip_prob, 3),
            "skip_prediction": "skip" if skip_prob >= 0.5 else "no-skip",
            "model_version": STATE["version"],
        }
    except Exception as e:
        with LOCK:
            STATE["errors_total"] += 1
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/segment")
def segment(listener: ListenerTaste):
    """Listener taste cluster (W9) — /segment wala kmeans listeners.csv pe fit hai,
    avg_* names — ListenerTaste schema se wahi aate hain, sab sahi hai."""
    row = pd.DataFrame([listener.model_dump()])[STATE["listener_features"]]
    cluster_id = int(STATE["kmeans"].predict(row)[0])
    info = STATE["cluster_info"][str(cluster_id)]
    return {
        "cluster_id": cluster_id,
        "segment_name": info["name"],
        "description": info["description"],
    }


# ---------------- W12: Retraining trigger ----------------
def _retrain_background():
    try:
        subprocess.run(["python", "scripts/train.py"], check=True, cwd=ROOT_DIR)
        load_models()
    finally:
        STATE["retraining"] = False


@app.post("/retrain")
def retrain():
    if STATE["retraining"]:
        return {"status": "already retraining"}
    STATE["retraining"] = True
    threading.Thread(target=_retrain_background, daemon=True).start()
    return {"status": "retraining started", "note": "thodi der baad /metrics check karo"}