import { useEffect, useState } from "react";

const API = (import.meta.env.VITE_API_URL || "/api").replace(/\/$/, "");

const TRACK_SLIDERS = [
  { key: "danceability", label: "Danceability", min: 0, max: 1, step: 0.01 },
  { key: "energy", label: "Energy", min: 0, max: 1, step: 0.01 },
  { key: "tempo", label: "Tempo", unit: "BPM", min: 60, max: 200, step: 1 },
  { key: "valence", label: "Valence", hint: "musical positivity", min: 0, max: 1, step: 0.01 },
  { key: "acousticness", label: "Acousticness", min: 0, max: 1, step: 0.01 },
  { key: "speechiness", label: "Speechiness", min: 0, max: 0.5, step: 0.01 },
  { key: "instrumentalness", label: "Instrumentalness", min: 0, max: 1, step: 0.01 },
  { key: "playlist_position", label: "Playlist position", min: 1, max: 50, step: 1 },
  { key: "hour_of_day", label: "Hour of day", min: 0, max: 23, step: 1 },
  { key: "is_playlist_radio", label: "Playlist radio", min: 0, max: 1, step: 1 },
  { key: "listener_avg_skip_rate", label: "Listener skip rate", min: 0, max: 1, step: 0.01 },
  { key: "days_since_release", label: "Days since release", unit: "days", min: 1, max: 2000, step: 1 },
];

const LISTENER_SLIDERS = [
  { key: "avg_danceability", label: "Danceability", min: 0, max: 1, step: 0.01 },
  { key: "avg_energy", label: "Energy", min: 0, max: 1, step: 0.01 },
  { key: "avg_acousticness", label: "Acousticness", min: 0, max: 1, step: 0.01 },
  { key: "avg_tempo", label: "Tempo", unit: "BPM", min: 60, max: 200, step: 1 },
  { key: "avg_valence", label: "Valence", min: 0, max: 1, step: 0.01 },
];

const DEFAULT_TRACK = {
  danceability: 0.75, energy: 0.85, tempo: 128, valence: 0.7,
  acousticness: 0.15, speechiness: 0.05, instrumentalness: 0,
  playlist_position: 3, hour_of_day: 20, is_playlist_radio: 0,
  listener_avg_skip_rate: 0.25, days_since_release: 60,
};

const DEFAULT_LISTENER = {
  avg_danceability: 0.8, avg_energy: 0.7, avg_acousticness: 0.2,
  avg_tempo: 125, avg_valence: 0.7,
};

const TRACK_PRESETS = [
  { name: "Dance pop", values: DEFAULT_TRACK },
  {
    name: "Acoustic chill",
    values: {
      ...DEFAULT_TRACK, danceability: 0.42, energy: 0.28, tempo: 92,
      valence: 0.38, acousticness: 0.86, speechiness: 0.04,
      playlist_position: 18, hour_of_day: 22, days_since_release: 220,
    },
  },
  {
    name: "Late-night radio",
    values: {
      ...DEFAULT_TRACK, danceability: 0.62, energy: 0.54, tempo: 112,
      valence: 0.48, acousticness: 0.42, playlist_position: 12,
      hour_of_day: 1, is_playlist_radio: 1, listener_avg_skip_rate: 0.48,
    },
  },
];

const LISTENER_PRESETS = [
  { name: "Party", values: { avg_danceability: 0.85, avg_energy: 0.85, avg_acousticness: 0.15, avg_tempo: 135, avg_valence: 0.8 } },
  { name: "Chill", values: { avg_danceability: 0.45, avg_energy: 0.25, avg_acousticness: 0.85, avg_tempo: 95, avg_valence: 0.4 } },
];

function formatStreams(value) {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}K`;
  return new Intl.NumberFormat().format(value);
}

async function fetchJSON(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Request failed (${response.status})`);
  }
  return response.json();
}

async function postJSON(url, body) {
  return fetchJSON(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export default function App() {
  const [track, setTrack] = useState(DEFAULT_TRACK);
  const [listener, setListener] = useState(DEFAULT_LISTENER);
  const [prediction, setPrediction] = useState(null);
  const [segment, setSegment] = useState(null);
  const [apiOnline, setApiOnline] = useState(false);
  const [loadingPred, setLoadingPred] = useState(false);
  const [loadingSeg, setLoadingSeg] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    async function checkAPI() {
      try {
        await fetchJSON(`${API}/health`);
        if (active) setApiOnline(true);
      } catch {
        if (active) setApiOnline(false);
      }
    }
    checkAPI();
    const timer = window.setInterval(checkAPI, 30_000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  async function handlePredict() {
    setLoadingPred(true);
    setError("");
    try {
      setPrediction(await postJSON(`${API}/predict`, track));
    } catch {
      setError("Could not get a prediction. Check that the API is running, then try again.");
    } finally {
      setLoadingPred(false);
    }
  }

  async function handleSegment() {
    setLoadingSeg(true);
    setError("");
    try {
      setSegment(await postJSON(`${API}/segment`, listener));
    } catch {
      setError("Could not find a listener segment. Check that the API is running, then try again.");
    } finally {
      setLoadingSeg(false);
    }
  }

  const skipPct = prediction ? Math.round(prediction.skip_probability * 100) : 0;

  return (
    <main className="app">
      <nav className="topbar" aria-label="Main">
        <a className="brand" href="#" aria-label="PlaylistPulse home">
          <span className="brand-mark" aria-hidden="true">♫</span>
          <span>Playlist<span className="brand-accent">Pulse</span></span>
        </a>
        <span className={`api-status ${apiOnline ? "online" : "offline"}`}>
          <span className="status-dot" />
          {apiOnline ? "API connected" : "API offline"}
        </span>
      </nav>

      <header className="hero">
        <div className="hero-copy">
          <p className="eyebrow">MUSIC INTELLIGENCE · PROJECT 12</p>
          <h1>Find the next<br /><span>track worth hearing.</span></h1>
          <p className="hero-description">
            Explore how sound, listening context, and personal taste shape a track&apos;s
            reach — and the chance someone skips.
          </p>
          <div className="hero-tags">
            <span>01 · STREAMS</span><span>02 · SKIP RISK</span><span>03 · TASTE</span>
          </div>
        </div>
        <div className="hero-art" aria-hidden="true">
          <div className="art-disc"><span>♫</span></div>
          <span className="art-note note-one">♪</span>
          <span className="art-note note-two">♫</span>
          <span className="art-caption">SOUND / SIGNAL</span>
        </div>
      </header>

      <section className="section-heading">
        <div>
          <p className="eyebrow">YOUR WORKSPACE</p>
          <h2>Make a prediction</h2>
        </div>
        <p className="demo-note"><span className="demo-dot" /> Demo uses synthetic training data</p>
      </section>

      {error && <div className="error" role="alert">{error}</div>}

      <div className="workspace">
        <section className="card track-card">
          <div className="card-heading">
            <div className="card-icon blue-icon">♫</div>
            <div>
              <p className="card-kicker">TRACK ANALYSIS</p>
              <h3>Sound &amp; context</h3>
            </div>
            <span className="step-number">01</span>
          </div>
          <p className="card-description">Adjust the track profile or start with a quick example.</p>

          <div className="preset-row" aria-label="Track examples">
            {TRACK_PRESETS.map((preset) => (
              <button
                className="preset-button"
                key={preset.name}
                onClick={() => { setTrack(preset.values); setPrediction(null); }}
                type="button"
              >
                {preset.name}
              </button>
            ))}
          </div>

          <div className="slider-list">
            {TRACK_SLIDERS.map((slider) => (
              <label className="slider-row" key={slider.key}>
                <span className="slider-label">
                  {slider.label}
                  {slider.hint && <small>{slider.hint}</small>}
                </span>
                <input
                  aria-label={slider.label}
                  type="range"
                  min={slider.min}
                  max={slider.max}
                  step={slider.step}
                  value={track[slider.key]}
                  onChange={(event) => {
                    setTrack({ ...track, [slider.key]: Number(event.target.value) });
                    setPrediction(null);
                  }}
                />
                <span className="slider-value">
                  {track[slider.key]}{slider.unit ? ` ${slider.unit}` : ""}
                </span>
              </label>
            ))}
          </div>

          <button className="primary-button" onClick={handlePredict} disabled={loadingPred || !apiOnline}>
            {loadingPred ? "Analysing track…" : "Predict streams & skip risk"}
            {!loadingPred && <span aria-hidden="true">→</span>}
          </button>

          {prediction && (
            <div className="prediction-result" aria-live="polite">
              <div className="result-topline">
                <span>MODEL PREDICTION</span>
                <span>{prediction.model_version}</span>
              </div>
              <div className="result-stats">
                <div className="stream-stat">
                  <strong>{formatStreams(prediction.predicted_streams)}</strong>
                  <span>expected streams</span>
                </div>
                <div className="skip-stat">
                  <div className="skip-label">
                    <span>Skip likelihood</span><strong>{skipPct}%</strong>
                  </div>
                  <div
                    className="bar-track"
                    role="progressbar"
                    aria-label="Skip likelihood"
                    aria-valuenow={skipPct}
                    aria-valuemin="0"
                    aria-valuemax="100"
                  >
                    <div className={`bar-fill ${skipPct >= 50 ? "high-risk" : ""}`} style={{ width: `${skipPct}%` }} />
                  </div>
                  <span className="skip-outcome">
                    {prediction.skip_prediction === "skip" ? "Higher skip risk" : "More likely to keep listening"}
                  </span>
                </div>
              </div>
            </div>
          )}
        </section>

        <section className="card listener-card">
          <div className="card-heading">
            <div className="card-icon lavender-icon">◎</div>
            <div>
              <p className="card-kicker">TASTE DISCOVERY</p>
              <h3>Listener profile</h3>
            </div>
            <span className="step-number">02</span>
          </div>
          <p className="card-description">Describe a listener&apos;s usual sound to discover their taste cluster.</p>

          <div className="preset-row" aria-label="Listener examples">
            {LISTENER_PRESETS.map((preset) => (
              <button
                className="preset-button"
                key={preset.name}
                onClick={() => { setListener(preset.values); setSegment(null); }}
                type="button"
              >
                {preset.name} listener
              </button>
            ))}
          </div>

          <div className="slider-list listener-sliders">
            {LISTENER_SLIDERS.map((slider) => (
              <label className="slider-row" key={slider.key}>
                <span className="slider-label">{slider.label}</span>
                <input
                  aria-label={`Average ${slider.label}`}
                  type="range"
                  min={slider.min}
                  max={slider.max}
                  step={slider.step}
                  value={listener[slider.key]}
                  onChange={(event) => {
                    setListener({ ...listener, [slider.key]: Number(event.target.value) });
                    setSegment(null);
                  }}
                />
                <span className="slider-value">
                  {listener[slider.key]}{slider.unit ? ` ${slider.unit}` : ""}
                </span>
              </label>
            ))}
          </div>

          <div className="taste-note">
            <span aria-hidden="true">✳</span>
            <p>Clustering compares several listening habits together — not just one slider at a time.</p>
          </div>
          <button className="secondary-button" onClick={handleSegment} disabled={loadingSeg || !apiOnline}>
            {loadingSeg ? "Finding your segment…" : "Discover listener segment"}
            {!loadingSeg && <span aria-hidden="true">→</span>}
          </button>

          {segment && (
            <div className="segment-result" aria-live="polite">
              <span className="segment-orbit" aria-hidden="true">♫</span>
              <div>
                <p className="card-kicker">YOUR TASTE CLUSTER · #{segment.cluster_id}</p>
                <h4>{segment.segment_name}</h4>
                <p>{segment.description}</p>
              </div>
            </div>
          )}
        </section>
      </div>

      <footer className="footer">
        <span>PlaylistPulse <span className="footer-separator">/</span> ML capstone</span>
        <span>Predictions are educational examples, not real-world guarantees.</span>
      </footer>
    </main>
  );
}
