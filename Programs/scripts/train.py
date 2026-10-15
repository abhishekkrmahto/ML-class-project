"""
PlaylistPulse — Step 2 (FULL RUBRIC): W3 baseline se W11 versioned artefact tak.
Run:  python scripts/train.py   (project root se)
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import (Lasso, LinearRegression, LogisticRegression,
                                  Ridge, SGDRegressor)
from sklearn.metrics import (brier_score_loss, classification_report,
                             mean_squared_error, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
from xgboost import XGBClassifier, XGBRegressor

try:
    from features import (ALL_TRACK_FEATURES, LISTENER_FEATURES,
                          RAW_TRACK_FEATURES, TASTE_DIMS, engineer_features)
except ImportError:
    from scripts.features import (ALL_TRACK_FEATURES, LISTENER_FEATURES,
                                  RAW_TRACK_FEATURES, TASTE_DIMS, engineer_features)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

ARCHETYPES = {
    "High-Energy Party":   [0.85, 0.85, 0.15, 135, 0.80],
    "Chill Acoustic Soul": [0.45, 0.25, 0.85,  95, 0.40],
    "Dance Pop Lover":     [0.80, 0.60, 0.30, 120, 0.75],
    "Focus Instrumental":  [0.30, 0.35, 0.50, 105, 0.25],
}
CLUSTER_DESCRIPTIONS = {
    "High-Energy Party":   "Loves loud, fast, feel-good bangers — the life of every party.",
    "Chill Acoustic Soul": "Prefers soft, acoustic, laid-back tracks for late-night unwinding.",
    "Dance Pop Lover":     "Hooked on catchy, upbeat, highly danceable pop hits.",
    "Focus Instrumental":  "Low-key listener — calm instrumental music for work and focus.",
}


# ---------------- W3: experiment tracking (MLflow, JSON fallback) ----------------
def log_run(name, params, metrics):
    """Har model ka run log karo — MLflow hai toh wahan, warna experiments.json."""
    try:
        import mlflow
        mlflow.set_experiment("PlaylistPulse")
        with mlflow.start_run(run_name=name):
            mlflow.log_params({k: str(v) for k, v in params.items()})
            for k, v in metrics.items():
                mlflow.log_metric(k, float(v))
        print(f"   [mlflow] run logged: {name}")
    except Exception:
        path = os.path.join(MODELS_DIR, "experiments.json")
        runs = json.load(open(path)) if os.path.exists(path) else []
        runs.append({"run": name, "params": params, "metrics": metrics,
                     "ts": time.strftime("%Y-%m-%d %H:%M:%S")})
        with open(path, "w") as f:
            json.dump(runs, f, indent=2)
        print(f"   [json-fallback] run logged: {name}")


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# ---------------- Data load ----------------
tracks = pd.read_csv(os.path.join(DATA_DIR, "tracks.csv"))
listeners = pd.read_csv(os.path.join(DATA_DIR, "listeners.csv"))

# Keep test tracks out of the unsupervised taste-feature fit too.
train_idx, test_idx = train_test_split(
    tracks.index,
    test_size=0.2,
    random_state=42,
    stratify=tracks["skipped_within_30s"],
)

# W9 (pehle, kyunki ye feature banega): track-level taste segments
print("=" * 60)
print("W9 | Taste segment model (ye W4 features mein jayega)")
taste_kmeans = make_pipeline(StandardScaler(), KMeans(n_clusters=4, random_state=42, n_init=10))
taste_kmeans.fit(tracks.loc[train_idx, TASTE_DIMS])

# W4: feature engineering — shared module se
X_df = engineer_features(tracks, taste_kmeans)
X = X_df[ALL_TRACK_FEATURES]
y_reg = pd.Series(np.log1p(tracks["streams"]), index=tracks.index)  # heavy tail -> log1p
y_cls = tracks["skipped_within_30s"]

# W2: the test set stays sealed until final evaluation.
X_tmp, X_te = X.loc[train_idx], X.loc[test_idx]
y_tmp, y_te = y_reg.loc[train_idx], y_reg.loc[test_idx]
Xc_tmp, Xc_te = X_tmp, X_te
yc_tmp, yc_te = y_cls.loc[train_idx], y_cls.loc[test_idx]
X_tr, X_val, y_tr, y_val = train_test_split(
    X_tmp, y_tmp, test_size=0.15, random_state=42
)
Xc_tr, Xc_val = X_tr, X_val
yc_tr, yc_val = y_cls.loc[X_tr.index], y_cls.loc[X_val.index]

# ---------------- W3: OLS + gradient descent baselines ----------------
print("=" * 60)
print("W3 | Linear regression baselines (log-streams)")
ols = make_pipeline(StandardScaler(), LinearRegression()).fit(X_tr, y_tr)
rmse_ols = rmse(y_val, ols.predict(X_val))
sgd = make_pipeline(StandardScaler(), SGDRegressor(max_iter=2000, random_state=42)).fit(X_tr, y_tr)
rmse_sgd = rmse(y_val, sgd.predict(X_val))
print(f"   OLS validation RMSE:            {rmse_ols:.3f}")
print(f"   GradientDescent validation RMSE: {rmse_sgd:.3f}")
log_run("ols-baseline", {"model": "OLS"}, {"validation_rmse": rmse_ols})
log_run("sgd-baseline", {"model": "SGD"}, {"validation_rmse": rmse_sgd})

# ---------------- W4: Ridge/Lasso + logistic + FE ----------------
print("=" * 60)
print("W4 | Ridge/Lasso + logistic skip classifier + engineered features")
ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X_tr, y_tr)
rmse_ridge = rmse(y_val, ridge.predict(X_val))
lasso = make_pipeline(StandardScaler(), Lasso(alpha=0.01)).fit(X_tr, y_tr)
rmse_lasso = rmse(y_val, lasso.predict(X_val))
n_zero = int(np.sum(lasso[-1].coef_ == 0))
print(f"   Ridge validation RMSE: {rmse_ridge:.3f} | Lasso validation RMSE: {rmse_lasso:.3f}")
print(f"   Lasso ne {n_zero} features ko zero kar diya (sparsity — irrelevant features out)")
logreg = make_pipeline(StandardScaler(),
                       LogisticRegression(max_iter=1000, class_weight="balanced")).fit(Xc_tr, yc_tr)
auc_lr = roc_auc_score(yc_val, logreg.predict_proba(Xc_val)[:, 1])
print(f"   Logistic skip validation ROC-AUC: {auc_lr:.3f} (class_weight=balanced)")
log_run("logistic-skip", {"model": "LogisticRegression"}, {"validation_auc": auc_lr})

# ---------------- W5: CV tuning + one-standard-error rule ----------------
print("=" * 60)
print("W5 | 5-fold CV + one-standard-error rule (Ridge alpha)")
ALPHAS = [0.01, 0.1, 1, 10, 100]
cv_rows = []
for a in ALPHAS:
    pipe = make_pipeline(StandardScaler(), Ridge(alpha=a))
    scores = -cross_val_score(pipe, X_tmp, y_tmp, cv=5,
                              scoring="neg_root_mean_squared_error")
    cv_rows.append((a, scores.mean(), scores.std(ddof=1) / np.sqrt(5)))
    print(f"   alpha={a:<6} CV RMSE: {scores.mean():.3f} (+/- {cv_rows[-1][2]:.3f})")
best = min(cv_rows, key=lambda t: t[1])
threshold = best[1] + best[2]  # best + 1 SE
chosen_alpha = max(a for a, m, _ in cv_rows if m <= threshold)  # 1-SE ke andar sabse simple
print(f"   Best alpha: {best[0]} | 1-SE rule ne chuna: {chosen_alpha} (simple model, same performance)")
ridge_final = make_pipeline(StandardScaler(), Ridge(alpha=chosen_alpha)).fit(X_tr, y_tr)
rmse_ridge_w5 = rmse(y_val, ridge_final.predict(X_val))
print(f"   Ridge (1-SE alpha) validation RMSE: {rmse_ridge_w5:.3f}")

# ---------------- W6: decision tree + top splits ----------------
print("=" * 60)
print("W6 | Decision tree for skip — top splits padho")
dt = DecisionTreeClassifier(max_depth=4, random_state=42,
                            class_weight="balanced").fit(Xc_tr, yc_tr)
auc_dt = roc_auc_score(yc_val, np.take(dt.predict_proba(Xc_val), 1, axis=1))
print(f"   DecisionTree validation ROC-AUC: {auc_dt:.3f}")
print(export_text(dt, feature_names=list(X.columns), max_depth=2))
imp = pd.Series(dt.feature_importances_, index=X.columns).sort_values(ascending=False)
print("   Top splits:", ", ".join(f"{k} ({v:.2f})" for k, v in imp.head(3).items()))

# ---------------- W7: random forest + OOB + permutation importance ----------------
print("=" * 60)
print("W7 | Random forest — OOB error + permutation importance")
rf_cls = RandomForestClassifier(n_estimators=200, oob_score=True, random_state=42,
                                n_jobs=-1, class_weight="balanced").fit(Xc_tr, yc_tr)
auc_rf = roc_auc_score(yc_val, rf_cls.predict_proba(Xc_val)[:, 1])
print(f"   RF validation ROC-AUC: {auc_rf:.3f} | OOB error: {1 - rf_cls.oob_score_:.3f}")
perm = permutation_importance(rf_cls, Xc_val.iloc[:500], yc_val.iloc[:500],
                              n_repeats=5, scoring="roc_auc", random_state=42)
perm_s = pd.Series(
    np.asarray(perm["importances"]).mean(axis=1), index=X.columns
).sort_values(ascending=False)
print("   Permutation importance top 5:", ", ".join(f"{k} ({v:.3f})" for k, v in perm_s.head(5).items()))
rf_reg = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1).fit(X_tr, y_tr)
rmse_rf = rmse(y_val, rf_reg.predict(X_val))

# ---------------- W8: XGBoost champion + early stopping + SHAP ----------------
print("=" * 60)
print("W8 | XGBoost — early stopping (val set pe) + SHAP")
xgb_es = XGBRegressor(n_estimators=800, learning_rate=0.05, max_depth=4, subsample=0.8,
                      colsample_bytree=0.8, early_stopping_rounds=30,
                      eval_metric="rmse", random_state=42, verbosity=0)
xgb_es.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
best_iter = xgb_es.best_iteration + 1 if xgb_es.best_iteration is not None else 800
print(f"   Early stopping at round {best_iter} (val set pe — test pe NAHI, leakage-safe)")
rmse_xgb = rmse(y_val, xgb_es.predict(X_val))
print(f"   XGB validation RMSE: {rmse_xgb:.3f}")

spw = float((yc_tr == 0).sum() / (yc_tr == 1).sum())  # imbalance-aware XGB
xgb_c_es = XGBClassifier(n_estimators=800, learning_rate=0.05, max_depth=4,
                         subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                         early_stopping_rounds=30, eval_metric="logloss",
                         random_state=42, verbosity=0)
xgb_c_es.fit(Xc_tr, yc_tr, eval_set=[(Xc_val, yc_val)], verbose=False)
best_iter_c = xgb_c_es.best_iteration + 1 if xgb_c_es.best_iteration is not None else 800
auc_xgb = roc_auc_score(yc_val, xgb_c_es.predict_proba(Xc_val)[:, 1])
print(f"   XGB skip validation ROC-AUC: {auc_xgb:.3f}")

try:
    import shap
    sv = shap.TreeExplainer(xgb_es).shap_values(X_val.iloc[:200])
    shap_s = pd.Series(np.abs(sv).mean(0), index=X.columns).sort_values(ascending=False)
    print("   SHAP top 5 (streams):", ", ".join(f"{k} ({v:.3f})" for k, v in shap_s.head(5).items()))
except Exception as e:
    print(f"   SHAP skip ({e}) — fallback feature_importances_ used")

# ---------------- Select models on validation; evaluate test once ----------------
print("=" * 60)
print("CHAMPION SELECTION (validation set)")
reg_models = {"OLS": (rmse_ols, ols), "SGD": (rmse_sgd, sgd),
              "Ridge": (rmse_ridge_w5, ridge_final), "Lasso": (rmse_lasso, lasso),
              "RandomForest": (rmse_rf, rf_reg), "XGBoost": (rmse_xgb, xgb_es)}
cls_models = {"Logistic": (auc_lr, logreg), "DecisionTree": (auc_dt, dt),
              "RandomForest": (auc_rf, rf_cls), "XGBoost": (auc_xgb, xgb_c_es)}
champ_reg_name, (_, champ_reg_candidate) = min(
    reg_models.items(), key=lambda kv: kv[1][0]
)
champ_cls_name, (_, champ_cls_candidate) = max(
    cls_models.items(), key=lambda kv: kv[1][0]
)
champ_reg = clone(champ_reg_candidate)
if champ_reg_name == "XGBoost":
    champ_reg.set_params(n_estimators=best_iter, early_stopping_rounds=None)
champ_reg.fit(X_tmp, y_tmp)
champ_cls = clone(champ_cls_candidate)
if champ_cls_name == "XGBoost":
    champ_cls.set_params(n_estimators=best_iter_c, early_stopping_rounds=None)
champ_cls.fit(Xc_tmp, yc_tmp)
rmse_champ = rmse(y_te, champ_reg.predict(X_te))
auc_champ = roc_auc_score(yc_te, champ_cls.predict_proba(Xc_te)[:, 1])
print(f"   Regression champion: {champ_reg_name} (test RMSE {rmse_champ:.3f})")
print(f"   Skip champion:       {champ_cls_name} (test AUC {auc_champ:.3f})")

# ---------------- W9: listener clustering — KMeans + DBSCAN + PCA ----------------
print("=" * 60)
print("W9 | Listener taste clusters — KMeans + DBSCAN + PCA")
km_pipe = make_pipeline(StandardScaler(), KMeans(n_clusters=4, random_state=42, n_init=10))
km_labels = km_pipe.fit_predict(listeners)
cluster_info = {}
for cid in range(4):
    raw_center = listeners[km_labels == cid].mean().to_numpy()
    name = min(ARCHETYPES, key=lambda n: np.linalg.norm(np.array(ARCHETYPES[n]) - raw_center))
    cluster_info[str(cid)] = {"name": name, "description": CLUSTER_DESCRIPTIONS[name]}
print("   KMeans 4 clusters:", {c: cluster_info[c]["name"] for c in cluster_info})

db = make_pipeline(StandardScaler(), DBSCAN(eps=1.2, min_samples=5)).fit(listeners)
db_labels = db[-1].labels_
n_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
print(f"   DBSCAN: {n_db} clusters, {int((db_labels == -1).sum())} noise points (KMeans se alag: shapes khud dhoondhta hai)")

pca = make_pipeline(StandardScaler(), PCA(n_components=2)).fit(listeners)
print(f"   PCA 2 components explain {pca[-1].explained_variance_ratio_.sum():.1%} variance (structure compress)")

# ---------------- W10: nested CV + calibration + significance test ----------------
print("=" * 60)
print("W10 | Nested CV + probability calibration + significance test")
inner = GridSearchCV(make_pipeline(StandardScaler(), Ridge()),
                     {"ridge__alpha": ALPHAS}, cv=3,
                     scoring="neg_root_mean_squared_error")
outer = -cross_val_score(inner, X_tmp, y_tmp, cv=3, scoring="neg_root_mean_squared_error")
print(f"   Nested CV RMSE: {outer.mean():.3f} +/- {outer.std():.3f} (tuning bhi CV ke andar — no optimism bias)")

calibrated = CalibratedClassifierCV(champ_cls, method="sigmoid", cv=3).fit(Xc_tmp, yc_tmp)
p_raw = champ_cls.predict_proba(Xc_te)[:, 1]
p_cal = calibrated.predict_proba(Xc_te)[:, 1]
brier_raw, brier_cal = brier_score_loss(yc_te, p_raw), brier_score_loss(yc_te, p_cal)
print(f"   Brier score: raw {brier_raw:.3f} -> calibrated {brier_cal:.3f} (chhota = better probabilities)")

sq_champ = (y_te - champ_reg.predict(X_te)) ** 2
base_name = "SGD" if champ_reg_name == "OLS" else "OLS"
baseline_reg = clone(reg_models[base_name][1]).fit(X_tmp, y_tmp)
sq_base = (y_te - baseline_reg.predict(X_te)) ** 2
t1, p1 = stats.ttest_rel(sq_champ, sq_base)

def per_sample_logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))

ll_champ = per_sample_logloss(yc_te.to_numpy(), p_cal)
baseline_cls = clone(logreg).fit(Xc_tmp, yc_tmp)
ll_lr = per_sample_logloss(yc_te.to_numpy(), baseline_cls.predict_proba(Xc_te)[:, 1])
t2, p2 = stats.ttest_rel(ll_champ, ll_lr)
print(f"   Significance (sealed test): regression p={p1:.4f}, skip p={p2:.4f}")
print("   (p < 0.05 = champion ka improvement statistically real hai, luck nahi)")
print(classification_report(yc_te, (p_cal >= 0.5).astype(int), target_names=["no-skip", "skip"]))

# ---------------- W11: versioned artefact + PSI reference save ----------------
print("=" * 60)
print("W11 | Versioned artefacts save")
joblib.dump(champ_reg, os.path.join(MODELS_DIR, "regressor.pkl"))
joblib.dump(calibrated, os.path.join(MODELS_DIR, "classifier.pkl"))
joblib.dump(km_pipe, os.path.join(MODELS_DIR, "kmeans.pkl"))
joblib.dump(taste_kmeans, os.path.join(MODELS_DIR, "taste_kmeans.pkl"))
psi_ref = {c: tracks[c].to_numpy() for c in RAW_TRACK_FEATURES}
joblib.dump(psi_ref, os.path.join(MODELS_DIR, "psi_ref.pkl"))  # drift guard ka reference

with open(os.path.join(MODELS_DIR, "features.json"), "w") as f:
    json.dump({"raw_track_features": RAW_TRACK_FEATURES,
               "track_features": ALL_TRACK_FEATURES,
               "listener_features": LISTENER_FEATURES}, f, indent=2)
with open(os.path.join(MODELS_DIR, "cluster_labels.json"), "w") as f:
    json.dump(cluster_info, f, indent=2)

version = {
    "version": f"v2-{int(time.time())}",
    "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "champion_regressor": champ_reg_name,
    "champion_classifier": champ_cls_name,
    "metrics": {"test_rmse_log": round(rmse_champ, 4), "test_roc_auc": round(auc_champ, 4),
                "brier_calibrated": round(brier_cal, 4),
                "nested_cv_rmse": round(float(outer.mean()), 4),
                "significance_p_reg": round(float(p1), 5),
                "significance_p_cls": round(float(p2), 5)},
}
with open(os.path.join(MODELS_DIR, "version.json"), "w") as f:
    json.dump(version, f, indent=2)
log_run("champion", {"regressor": champ_reg_name, "classifier": champ_cls_name},
        version["metrics"])
print(f"   Saved: regressor.pkl, classifier.pkl, kmeans.pkl, taste_kmeans.pkl,")
print(f"          psi_ref.pkl, features.json, cluster_labels.json, version.json")
print(f"   Version: {version['version']} — DONE! Ab: uvicorn backend.main:app --reload")