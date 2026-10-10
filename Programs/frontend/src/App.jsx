import { useState } from "react";

const API = "http://localhost:8000";

const TRACK_SLIDERS = [
  { key: "danceability", label: "Danceability", min: 0, max: 1, step: 0.01 },
  { key: "energy", label: "Energy", min: 0, max: 1, step: 0.01 },
  { key: "tempo", label: "Tempo (BPM)", min: 60, max: 200, step: 1 },
  { key: "valence", label: "Valence (happiness)", min: 0, max: 1, step: 0.01 },
  { key: "acousticness", label: "Acousticness", min: 0, max: 1, step: 0.01 },
  { key: "speechiness", label: "Speechiness", min: 0, max: 0.5, step: 0.01 },
  { key: "instrumentalness", label: "Instrumentalness", min: 0, max: 1, step: 0.01 },
  { key: "playlist_position", label: "Playlist position", min: 1, max: 50, step: 1 },
  { key: "hour_of_day", label: "Hour of day", min: 0, max: 23, step: 1 },
  { key: "is_playlist_radio", label: "Playlist radio (0 = no, 1 = yes)", min: 0, max: 1, step: 1 },
  { key: "listener_avg_skip_rate", label: "Listener avg skip rate", min: 0, max: 1, step: 0.01 },
  { key: "days_since_release", label: "Days since release", min: 1, max: 2000, step: 1 },
];

const LISTENER_SLIDERS = [
  { key: "avg_danceability", label: "Avg danceability", min: 0, max: 1, step: 0.01 },
  { key: "avg_energy", label: "Avg energy", min: 0, max: 1, step: 0.01 },
  { key: "avg_acousticness", label: "Avg acousticness", min: 0, max: 1, step: 0.01 },
  { key: "avg_tempo", label: "Avg tempo (BPM)", min: 60, max: 200, step: 1 },
  { key: "avg_valence", label: "Avg valence", min: 0, max: 1, step: 0.01 },
];

const DEFAULT_TRACK = {
  danceability: 0.75, energy: 0.85, tempo: 128, valence: 0.7,
  acousticness: 0.15, speechiness: 0.05, instrumentalness: 0.0,
  playlist_position: 3, hour_of_day: 20, is_playlist_radio: 0,
  listener_avg_skip_rate: 0.25, days_since_release: 60,
};

const DEFAULT_LISTENER = {
  avg_danceability: 0.8, avg_energy: 0.7, avg_acousticness: 0.2,
  avg_tempo: 125, avg_valence: 0.7,
};

function formatStreams(n) {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (n >= 1_000) return (n / 1_000).toFixed(1) + "K";
  return String(n);
}

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error("API error");
  return res.json();
}

export default function App() {
  const [track, setTrack] = useState(DEFAULT_TRACK);
  const [listener, setListener] = useState(DEFAULT_LISTENER);
  const [prediction, setPrediction] = useState(null);
  const [segment, setSegment] = useState(null);
  const [loadingPred, setLoadingPred] = useState(false);
  const [loadingSeg, setLoadingSeg] = useState(false);
  const [error, setError] = useState(null);

  async function handlePredict() {
    setLoadingPred(true);
    setError(null);
    try {
      setPrediction(await postJSON(`${API}/predict`, track));
    } catch {
      setError("Prediction failed — backend chalu hai kya? (port 8000)");
    } finally {
      setLoadingPred(false);
    }
  }

  async function handleSegment() {
    setLoadingSeg(true);
    setError(null);
    try {
      setSegment(await postJSON(`${API}/segment`, listener));
    } catch {
      setError("Segmentation failed — backend chalu hai kya? (port 8000)");
    } finally {
      setLoadingSeg(false);
    }
  }

  const skipPct = prediction ? Math.round(prediction.skip_probability * 100) : 0;
  const barColor = prediction && prediction.skip_probability >= 0.5 ? "#ff4d6d" : "#4ade80";

  return (
    <div className="app">
      <header className="header">
        <h1>PlaylistPulse</h1>
        <p>Predict the next hit — and the next skip.</p>
      </header>

      {error && <div className="error">{error}</div>}

      <div className="cards">
        {/* Card 1: Track Predictor */}
        <section className="card">
          <h2>Track Predictor</h2>
          <p className="hint">Track ke features set karo, phir streams & skip risk predict karo.</p>

          {TRACK_SLIDERS.map((s) => (
            <div className="slider-row" key={s.key}>
              <label>{s.label}</label>
              <input
                type="range"
                min={s.min}
                max={s.max}
                step={s.step}
                value={track[s.key]}
                onChange={(e) => setTrack({ ...track, [s.key]: Number(e.target.value) })}
              />
              <span className="slider-value">{track[s.key]}</span>
            </div>
          ))}

          <button onClick={handlePredict} disabled={loadingPred}>
            {loadingPred ? "Predicting..." : "Predict"}
          </button>

          {prediction && (
            <div className="result">
              <span className="big">{formatStreams(prediction.predicted_streams)}</span>
              <span className="sub">predicted streams</span>
              <div className="skip-box">
                <p>
                  Skip probability: <strong>{skipPct}%</strong> —{" "}
                  {prediction.skip_prediction === "skip" ? "likely SKIP ⏭" : "likely KEEP ▶"}
                </p>
                <div className="bar-track">
                  <div
                    className="bar-fill"
                    style={{ width: `${skipPct}%`, background: barColor }}
                  />
                </div>
              </div>
            </div>
          )}
        </section>

        {/* Card 2: Listener Segment */}
        <section className="card">
          <h2>Listener Segment</h2>
          <p className="hint">Listener ki taste describe karo, cluster dhoondo.</p>

          {LISTENER_SLIDERS.map((s) => (
            <div className="slider-row" key={s.key}>
              <label>{s.label}</label>
              <input
                type="range"
                min={s.min}
                max={s.max}
                step={s.step}
                value={listener[s.key]}
                onChange={(e) => setListener({ ...listener, [s.key]: Number(e.target.value) })}
              />
              <span className="slider-value">{listener[s.key]}</span>
            </div>
          ))}

          <button onClick={handleSegment} disabled={loadingSeg}>
            {loadingSeg ? "Finding..." : "Find Segment"}
          </button>

          {segment && (
            <div className="result">
              <p className="segment-name">🎧 {segment.segment_name}</p>
              <p className="segment-desc">{segment.description}</p>
              <p className="segment-id">Cluster #{segment.cluster_id}</p>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}