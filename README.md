# PlaylistPulse

**Predict the next hit — and the next skip.**

PlaylistPulse is a learning-focused music analytics demo. It combines track audio/context features and listener history to estimate stream popularity, estimate skip-within-30-seconds risk, and group listeners by taste. A React interface calls a FastAPI service that loads saved scikit-learn/XGBoost models.

> **Important:** the included CSVs are generated synthetic data, not Spotify data or evidence about real listeners. This project is suitable for a class demonstration, not real release, ranking, or listener decisions. Real deployment needs consented, representative play/skip logs, privacy controls, monitoring, and a governed retraining process.

## 1. Start here

From the `Programs` directory, the fastest reproducible run is Docker Compose:

```powershell
docker compose up --build
```

Open:

- **PlaylistPulse UI:** http://localhost:5173
- **API health:** http://localhost:8000/health
- **Interactive API docs:** http://localhost:8000/docs
- **API metrics and drift:** http://localhost:8000/metrics

Stop it with `Ctrl+C`; remove stopped containers with `docker compose down`.

The containers use the model files already in `models/`. To regenerate synthetic data and retrain before starting, use the local steps below, then rebuild/restart Compose. Retraining takes longer than simply starting the demo.

### Run without Docker (Windows PowerShell)

Use two terminals, both opened in `Programs`:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts\generate_data.py
python scripts\train.py
python -m uvicorn backend.main:app --reload
```

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

The Vite development server serves the UI on port 5173 and proxies `/api/*` requests to the API on port 8000. API calls use the same proxy when running in Docker, so the browser does not need a hard-coded backend hostname.

## 2. What the project predicts

Each training row represents a track/listening context. The current demo has 5,000 generated track rows and 500 generated listener profiles.

### A. Expected streams: regression

- **Target:** `streams`, transformed with `log1p(streams)`.
- **Why transform it?** Streams are heavy-tailed: a few tracks can have very large counts. Taking a logarithm makes multiplicative differences easier to model and stops the biggest hits from dominating the loss.
- **How to read the output:** the model predicts log-streams; the API applies `expm1` to return an approximate count. The frontend formats that count for display.
- **Main metric:** RMSE on the log target. An error of 1 on this scale means a multiplicative error of about `e` in the original count, not “one stream.”

The synthetic generator adds independent Gaussian noise with standard deviation **0.6** to its `log_streams` formula. Since generated counts are large, `log1p(streams)` is approximately `log_streams`; therefore **about 0.6 log-stream RMSE is the known noise-floor reference** when predicting only from the generated features. It is a property of this simulator, not a guaranteed error floor for real music data.

### B. Skip within 30 seconds: classification

- **Target:** `skipped_within_30s`, where 1 means skipped and 0 means kept listening.
- **Output:** a probability between 0 and 1, plus a demo label using a 0.5 threshold.
- **Metrics:** ROC-AUC ranks likely skips above likely keeps; Brier score checks the quality of probabilities; the classification report shows precision/recall.
- **Imbalance:** if skips and keeps are not equally common, raw accuracy can look good while missing most skips. Training uses class weighting/positive weighting, and evaluation reports metrics beyond accuracy.

The simulator adds random noise to the skip logit and then samples a Bernoulli outcome. Even a perfect model of its conditional probability cannot know each random outcome in advance. Its theoretical classification error floor is the **Bayes error** `mean(min(p, 1-p))`, where `p` is the probability conditional on the observable features. For this generator, estimate it by integrating over its known 0.6 logit-noise distribution:

```python
import numpy as np
import pandas as pd
from scipy.special import expit

tracks = pd.read_csv("data/tracks.csv")
rng = np.random.default_rng(7)
noise = rng.normal(0, 0.6, size=(len(tracks), 500))
z = (
    -2.2
    + 1.6 * tracks["energy"].to_numpy()
    + 0.010 * (tracks["tempo"].to_numpy() - 120)
    + 2.0 * tracks["listener_avg_skip_rate"].to_numpy()
    + 0.8 * (tracks["playlist_position"].to_numpy() - 25) / 25
    + 0.4 * np.sin(2 * np.pi * tracks["hour_of_day"].to_numpy() / 24)
    - tracks["danceability"].to_numpy()
)
p = expit(z[:, None] + noise).mean(axis=1)
print("Estimated irreducible skip error:", np.minimum(p, 1 - p).mean())
```

This is a Monte Carlo estimate; using more noise samples makes it less variable. For real logs, the equivalent floor is not known without repeated/experimental observations or a well-validated model of the label-generating process.

### C. Listener taste: clustering

The listener endpoint groups the five average taste measurements (danceability, energy, acousticness, tempo, and valence). It returns a cluster ID and a human-readable archetype name. A cluster is a useful summary, **not** a permanent identity or a verified psychological category.

## 3. How the ML pieces fit together

**A model learns a mapping from examples.** During training, features (`X`) are paired with a target (`y`). A fitted model learns patterns from the training rows; evaluation checks how well those patterns generalize to rows not used to fit them.

### The 12-week roadmap in this repository

| Week | Concept | Where to look / what it does |
|---|---|---|
| W1 | Define the two targets, inspect data, think about the irreducible error floor | `scripts/generate_data.py` makes example targets. The formulas above explain their known noise. |
| W2 | Ingest, validate, explore, handle missingness/outliers, prevent leakage | Generated CSVs are clean by construction; the API validates ranges with Pydantic. This is **not** a real-data cleaning pipeline. The trainer keeps a stratified track test set out of feature fitting, tuning, and model selection. |
| W3 | Ordinary least squares (OLS) and gradient descent | `LinearRegression` fits a least-squares line; `SGDRegressor` approximates it with repeated small updates. MLflow records validation metrics. |
| W4 | Ridge, Lasso, logistic regression, feature engineering | Ridge shrinks coefficients; Lasso can shrink some to zero. Logistic regression estimates skip probability. The shared feature module adds interactions and context indicators. |
| W5 | Cross-validation and the one-standard-error rule | Five folds compare Ridge strengths. The one-SE rule chooses the strongest regularization still within one standard error of the best mean score (a simpler model when scores are similar). |
| W6 | Decision tree | A shallow tree makes if/then splits that are easier to explain, but can be unstable or overfit. |
| W7 | Random forest, OOB error, permutation importance | Many bootstrapped trees reduce variance; out-of-bag rows provide an internal check. Permutation importance measures how much validation AUC falls when one feature is shuffled. |
| W8 | XGBoost, early stopping, SHAP | Boosted trees learn residual patterns stage by stage. Early stopping selects the number of rounds on validation data. SHAP estimates feature contributions for individual predictions. |
| W9 | K-means, DBSCAN, PCA, segment feature | K-means finds a chosen number of compact groups; DBSCAN can mark dense groups and noise; PCA projects features into two dimensions. A track taste-cluster ID is included in the prediction features. |
| W10 | Nested CV, calibration, imbalance-aware metrics, significance | Nested CV estimates tuning performance without using the final test set. Sigmoid calibration is checked with Brier score. A paired test compares per-row losses on the final test set; its p-value is evidence, not proof of practical importance. |
| W11 | Shared features, versioned artifacts, PSI drift check | The API imports the same feature builder used by training. `models/version.json` records the selected model names/metrics; PSI compares recent request features with a reference distribution. |
| W12 | FastAPI, health, metrics, drift and retraining trigger | `/predict`, `/segment`, `/health`, `/metrics`, and a token-protected `/retrain` endpoint. Docker packages the API and web UI. |

### Why these models?

- **OLS / SGD:** understandable first baselines. Complex models should beat a sensible simple model, not just look impressive.
- **Ridge / Lasso:** useful when features overlap or coefficients are unstable. Ridge shrinks; Lasso can also remove features.
- **Logistic regression:** a transparent probability baseline for a binary skip label.
- **Decision tree:** demonstrates readable nonlinear rules.
- **Random forest:** averages many trees, often reducing the variance of a single tree.
- **XGBoost:** a strong tabular-data candidate that learns nonlinearities and feature interactions; it still needs validation and should be compared against simpler baselines.
- **K-means / DBSCAN / PCA:** unsupervised tools for grouping and visualizing preference profiles. They do not use the target labels to define groups.

### Leakage-safe model evaluation

The training script first reserves 20% of tracks as the test set, stratified by skip label. It fits the track-taste scaler/cluster model on the remaining training portion, uses a validation slice and cross-validation for choices, selects champions on validation metrics, refits those models on the full training portion, and evaluates the test set at the end. Nested CV currently estimates Ridge tuning performance specifically; it is not nested model-family selection for every candidate. Listener K-means/DBSCAN/PCA are exploratory fits on the full unlabeled listener table because the demo does not define a listener-cluster test target. Do not keep changing features or thresholds based on test results; doing so turns the test set into another validation set.

## 4. Feature engineering and data

`data/tracks.csv` includes audio-like values, playlist/time context, a listener-history summary, streams, and the skip label. `data/listeners.csv` contains aggregate preference profiles.

The important track columns are:

| Kind | Columns | Meaning |
|---|---|---|
| Audio inputs | `danceability`, `energy`, `tempo`, `valence`, `acousticness`, `speechiness`, `instrumentalness` | Generated audio-like descriptors. Most are scaled to 0–1; tempo is BPM. |
| Context inputs | `playlist_position`, `hour_of_day`, `is_playlist_radio`, `days_since_release` | Where/when the track is heard and how old the release is. |
| Listener history | `listener_avg_skip_rate` | A generated summary of the listener's past skip behavior. |
| Regression target | `streams` | Generated positive stream count; transformed for training. |
| Classification target | `skipped_within_30s` | Binary label: 1 = skip, 0 = keep listening. |

An initial exploratory data analysis (EDA) should answer: Are values in plausible ranges? Which inputs are missing? How skewed is `streams`? What fraction of examples are skips? Do relationships look plausible? Example quick inspection:

```powershell
python -c "import pandas as pd; d=pd.read_csv('data/tracks.csv'); print(d.describe().T); print('Stream quantiles:'); print(d.streams.quantile([0,.5,.9,.99,1])); print('Skip rate:', d.skipped_within_30s.mean()); print('Missing values:', d.isna().sum().sum())"
```

Here the generator deliberately produces valid bounded data, so the expected missing-value count is zero. In a real ingestion pipeline, investigate missingness and impossible values rather than silently filling everything; document imputation, and inspect outliers before clipping or removing them. A log target handles skew but does not make every outlier “wrong.”

`scripts/features.py` is the shared feature definition. Its extra track features include:

- `energy_dance`: energy × danceability interaction.
- `tempo_norm`: centered/scaled tempo.
- `is_night`, `is_new_track`, `position_early`: simple context flags.
- `taste_segment`: the fitted track-taste cluster.

This shared module matters: if training transforms tempo one way but the API transforms it another way, the model receives a different input distribution at prediction time. The backend and training script now call the same function.

The synthetic generator uses a fixed random seed, so it is reproducible. It does **not** simulate real artist IDs, actual catalog popularity, actual session sequences, or missing fields. If you replace it with real data, document units and provenance; validate ranges and IDs; split by time, listener, or track as appropriate; prevent future information from entering features; and make privacy/consent controls part of the ingestion design.

## 5. MLflow: what it is and how this project uses it

**MLflow is an experiment-tracking tool.** It records a run's parameters (for example model family or regularization strength), metrics (for example validation RMSE/AUC), tags, and optionally files/model artifacts. Its UI lets you compare runs rather than relying on memory or overwritten console output. MLflow does not choose a good model for you, prevent leakage automatically, or make synthetic data realistic.

`scripts/train.py` logs named experiments to the local MLflow tracking store when MLflow is available. If logging fails, it writes a JSON fallback under `models/experiments.json`. From `Programs`, launch the local run-comparison UI with:

```powershell
mlflow ui --host 127.0.0.1 --port 5000
```

Then open http://127.0.0.1:5000 and select the `PlaylistPulse` experiment. Set `MLFLOW_TRACKING_URI` before running training and the UI if you want to point both at a different MLflow store. Keep the tracking store and model files backed up in a real deployment.

## 6. API contract

Open http://localhost:8000/docs for the generated Swagger UI.

### `POST /predict`

Accepts bounded audio/context features such as `danceability`, `tempo`, `playlist_position`, and `listener_avg_skip_rate`. Returns:

```json
{
  "predicted_streams": 123456,
  "skip_probability": 0.37,
  "skip_prediction": "no-skip",
  "model_version": "v2-example"
}
```

### `POST /segment`

Accepts listener averages (`avg_danceability`, `avg_energy`, `avg_acousticness`, `avg_tempo`, `avg_valence`) and returns `cluster_id`, `segment_name`, and a description.

### `GET /health` and `GET /metrics`

`/health` reports model readiness/version. `/metrics` returns JSON request counters, average prediction latency, model version, the last training timestamp, drift status, and per-feature PSI alerts. PSI below 0.1 is treated as stable, 0.1–0.25 as moderate, and above 0.25 as major in this demonstration. PSI is a feature-distribution warning; it is not a measure of model accuracy.

### `POST /retrain`

Disabled unless `RETRAIN_API_KEY` is configured. When enabled, send the same value in the `X-Retrain-Token` header. This demonstration retrains in a background thread from the bundled data. It is **not** a production retraining workflow: a real service should queue a job, validate newly labelled data, evaluate candidate artifacts, approve them, then deploy atomically. Do not expose the demo retraining key publicly.

## 7. Train and inspect locally

From `Programs`:

```powershell
python scripts\generate_data.py
python scripts\train.py
```

The trainer prints model comparisons and writes the selected artifacts and metadata to `models/`. It includes heavier steps (nested CV, forests, XGBoost, SHAP); a first full training run may take a while. You can omit `shap` from `requirements.txt` only if you also intentionally remove/skip the SHAP demonstration.

Artifacts include:

- `regressor.pkl`, `classifier.pkl`: stream and skip models.
- `taste_kmeans.pkl`, `kmeans.pkl`: track taste feature and listener clustering models.
- `features.json`, `cluster_labels.json`: feature order and cluster descriptions.
- `version.json`: version, training timestamp, selected model names, and evaluation metrics.
- `psi_ref.pkl`: saved reference feature distributions.

Only load trusted `.pkl`/joblib artifacts: deserializing a malicious pickle can execute code.

## 8. Docker deployment

### Local or a Docker host

1. Install Docker Desktop (Windows/macOS) or Docker Engine + the Compose plugin (Linux).
2. In the `Programs` directory, run `docker compose up --build`.
3. Visit `http://localhost:5173`. Check `http://localhost:8000/health` and `http://localhost:8000/docs`.
4. For a hosted VM, copy the `Programs` directory and model artifacts to the host, install Docker, then run `docker compose up --build -d`. Configure the host firewall/reverse proxy and HTTPS before public access.

The backend image runs as a non-root user and has a container health check. The frontend is a static production build served by Nginx; Nginx proxies `/api/` to the backend container. Compose waits for the API health check before starting the UI.

Before exposing this demo publicly:

- Put it behind HTTPS and an access-controlled reverse proxy; add rate limiting and authentication for any non-public endpoints.
- Set `CORS_ORIGINS` to only the actual browser origin(s), comma-separated if needed.
- Leave `RETRAIN_API_KEY` unset to keep retraining disabled. If you enable it, use a long random secret, store it outside source control, and never expose it through the frontend.
- Use persistent, versioned model/data storage and a managed tracking/monitoring service. Compose's container filesystem is ephemeral.
- Replace synthetic training data with consented production data and rerun offline evaluation before any real use.

For a custom frontend domain, route that domain's `/api/` path to the backend (or configure a secure reverse proxy) and set `CORS_ORIGINS` to the frontend's exact origin when using direct cross-origin API calls. Do not use `*` as a production CORS policy.

### Deploy on Render (one API service + one static frontend)

Render should host the FastAPI API as a Docker web service and the React app as a static site. These steps deploy the currently committed artifacts; changing source code locally will not update Render until the changes are pushed to the connected GitHub branch.

1. Push this project to GitHub. In the Render Dashboard, select **New → Web Service**, connect the repository, and select the branch to deploy.
2. Configure the API service:
   - **Root Directory:** `Programs`
   - **Runtime:** `Docker`
   - **Dockerfile Path:** `Dockerfile`
   - **Health Check Path:** `/health`
   - **Environment:** add `CORS_ORIGINS` with the exact frontend origin, for example `https://playlistpulse-ui.onrender.com`. No trailing slash.
   - Leave `RETRAIN_API_KEY` unset unless you deliberately need the protected demo endpoint.
   - Create the service and wait for the first deployment. Copy its public HTTPS URL, such as `https://playlistpulse-api.onrender.com`. The Docker command listens on Render's injected `PORT`.
3. Select **New → Static Site** and connect the same repository and branch:
   - **Root Directory:** `Programs/frontend`
   - **Build Command:** `npm install && npm run build`
   - **Publish Directory:** `dist`
   - **Environment:** add `VITE_API_URL` and set it to the API's public URL from step 2, e.g. `https://playlistpulse-api.onrender.com` (no trailing slash).
   - Name the site `playlistpulse-ui` if you want the example `https://playlistpulse-ui.onrender.com` address.
4. Once the static site is created, check its actual `onrender.com` URL. If it differs from the origin entered for `CORS_ORIGINS`, update the API service's value to the exact origin (scheme + hostname, no path or trailing slash), then manually redeploy the API.
5. Open the static site's URL. Verify that `https://<api-service>.onrender.com/health` returns `"models_loaded": true`, then try both prediction cards. If the UI says API offline, confirm `VITE_API_URL` is the API URL and trigger a **Clear build cache & deploy** for the frontend after changing it. If the browser console reports a CORS error, correct `CORS_ORIGINS` and redeploy the API.

`VITE_API_URL` is embedded into the frontend at **build time**, so changing it requires a new frontend build/deploy. `CORS_ORIGINS` is read by the API at startup, so changing it requires an API redeploy. Render's free web services may sleep when idle; the first API request can take longer while the service wakes. For an always-on public demo, use an appropriate paid instance. Store secrets only in Render's Environment settings; never put them in GitHub or frontend variables.

## 9. Two-minute project-review explanation

> “PlaylistPulse has two supervised tasks and one unsupervised task. For popularity, we predict `log1p(streams)` because streams are heavy-tailed. For skips, we predict the probability of a skip within 30 seconds, and evaluate with ROC-AUC and Brier score because classes can be imbalanced and probability quality matters. Separately, K-means groups listener taste profiles. We compare simple linear/logistic baselines with regularized models, trees, random forests, and XGBoost; tuning uses training/validation data, and the final test set is held back for evaluation. FastAPI loads versioned model files, runs the shared feature builder, and exposes predictions, health, JSON metrics, and PSI drift status. Docker Compose runs the API and the light-blue React UI together. The current data is synthetic, so this proves the pipeline and demo behavior—not real-world recommendation accuracy.”

## 10. Project map

```text
Programs/
├── backend/main.py          # FastAPI endpoints, validation, inference, PSI
├── scripts/generate_data.py # Reproducible synthetic dataset
├── scripts/features.py      # Shared training/serving feature engineering
├── scripts/train.py         # Training, validation, comparison, artifact export
├── data/                    # Synthetic tracks and listeners
├── models/                  # Saved model artifacts and metadata
├── frontend/                # React + Vite interface and Nginx image
├── Dockerfile               # API image
├── compose.yaml             # Local two-container deployment
└── requirements.txt         # Python dependencies
```
