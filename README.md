# PlaylistPulse — ML ko simple examples se samjho

**Predict the next hit — and the next skip.**

Yeh guide beginner ke liye hai. Iska aim sirf code chalana nahi, balki yeh samajhna hai ki project ka har model **kya karta hai, kyun use hua, aur uska result kaise samjhayen**.

> **Sabse pehle sach:** project ka current data synthetic (computer-generated) hai. Isliye yeh machine-learning workflow aur demo dikhata hai; yeh real Spotify/artist data se trained ya real hits predict karne ka proof nahi hai. Review mein is limitation ko khud batana achhi scientific practice hai.

---

## 1-hour review ke liye padhai ka order

| Samay | Kya samjho |
|---|---|
| 0–5 min | Project ka purpose, 3 outputs, synthetic-data limitation |
| 5–13 min | Features/targets aur generated CSV data |
| 13–25 min | Regression, classification, log transform, baselines |
| 25–35 min | Trees, Random Forest, XGBoost, comparison |
| 35–43 min | Train/validation/test, cross-validation, leakage |
| 43–49 min | K-means, DBSCAN, PCA, SHAP |
| 49–54 min | MLflow, saved model, FastAPI, endpoints |
| 54–58 min | Metrics, calibration, PSI drift |
| 58–60 min | Neeche diya viva answer ek baar bol kar practice karo |

Agar sirf 10 minute hain: sections **2, 3, 5, 8, 10 aur 13** padho.

---

## 1. Project ko ek line mein kaise samjhayen?

**PlaylistPulse audio-style track features aur listening context se expected streams estimate karta hai, 30 seconds ke andar skip hone ki probability classify karta hai, aur listener ke taste profiles ko groups mein cluster karta hai.**

Iske teen alag ML tasks hain:

1. **Regression:** stream count jaisa number estimate karna.
2. **Classification:** skip hoga ya nahi, aur uski probability estimate karna.
3. **Clustering:** bina pehle se diye labels ke similar listener profiles ko group karna.

Yeh “next hit” ka educational prototype hai. Is code mein actual time series, release-date splits, ya real-world future hit labels nahi hain; isliye ise validated future-hit forecasting nahi kehna chahiye.

---

## 2. Dataset: model ko kya dikhaya jaata hai?

`data/tracks.csv` mein **5,000 synthetic track rows** aur `data/listeners.csv` mein **500 synthetic listener profiles** hain. Track data mein 12 input features aur 2 targets hain.

### Inputs = features = X

| Group | Columns | Simple meaning |
|---|---|---|
| Audio-like | `danceability`, `energy`, `tempo`, `valence`, `acousticness`, `speechiness`, `instrumentalness` | Gaana kaisa sound karta hai; tempo BPM mein, kai values 0–1 mein |
| Context | `playlist_position`, `hour_of_day`, `is_playlist_radio`, `days_since_release` | Gaana kahaan/kis waqt suna ja raha hai |
| Listener history | `listener_avg_skip_rate` | Listener ka average past skip rate |

**Analogy:** restaurant reviews mein cuisine, price aur location features ho sakte hain; gaane ke project mein sound aur listening context features hain.

### Targets = answers = y

- `streams`: stream count, regression ka target.
- `skipped_within_30s`: 1 = skip, 0 = listener ne sunna jaari rakha; classification ka target.

Model training mein model ko examples `(X, y)` dikhaye jaate hain. Model patterns seekhta hai: “aise inputs ke saath aam taur par aisa output dekha gaya.”

### EDA aur missing values

EDA ka matlab **Exploratory Data Analysis**: model se pehle data ko samajhna. Jaise:

- har column ka range sahi hai?
- missing values hain?
- `streams` mein kuch bahut bade outliers hain?
- skip karne wale examples kitne percent hain?
- features aur target ka relation sensible lagta hai?

Current CSV ko code se check kiya gaya: 5,000 tracks, 500 listeners, track CSV mein missing cells 0, aur skip rate **23.72%**. `streams` ka median lagbhag **256K**, 99th percentile lagbhag **1.58M**, aur maximum lagbhag **3.36M** hai. Isse popularity ka long/heavy tail dikhta hai: zyada tracks moderate hain, kuch bahut bade.

Ye values `scripts/generate_data.py` se aati hain, real listening logs se nahi. Real data mein missing values ko samajhkar handle karna padega; bina soch-samajh har missing value ko zero bharna sahi nahi hota.

---

## 3. Synthetic data kaise banta hai?

`scripts/generate_data.py` NumPy ke random generators se features banata hai. `np.random.seed(42)` hone se data dobara generate karne par wahi sample milta hai—isse **reproducibility** kehte hain.

### Streams target

Generator pehle ek noisy log-stream formula banata hai:

```text
log_streams =
  10.5
  + 1.5 × danceability
  + 1.2 × energy
  + 0.8 × valence
  - 0.6 × acousticness
  - 0.8 × speechiness
  + 0.6 × release_age_fraction
  + random_noise
```

Phir `streams = exp(log_streams)` karta hai. `exp` (exponential) ki wajah se count ka scale bada aur skewed hota hai. Matlab: generator ne hi banaya hai ki zyada danceable/energetic/happy examples ko average par zyada streams mil sakte hain—lekin noise ki wajah se har row formula ke bilkul barabar nahi hoti.

### Skip target

Skip ke liye pehle inputs se ek score `z` banta hai, us par **sigmoid** lagti hai:

```text
p = 1 / (1 + exp(-z))
```

Sigmoid kisi bhi real score ko 0 se 1 ke beech probability banati hai. Phir random draw se `skipped_within_30s` 0 ya 1 banta hai. Isliye ek hi probability 0.3 waale kai plays mein kabhi skip, kabhi no-skip ho sakta hai. Prediction probability hai, guarantee nahi.

### Listener profiles

Generator chaar designed taste archetypes ke around listener values banata hai: party, chill/acoustic, dance-pop aur focus/instrumental. Har profile mein thoda random noise add hota hai. Clustering ka demo isi known structure ko discover karne ke liye banaya gaya hai.

---

## 4. ML ke buniyadi shabd

- **Sample/row:** ek example, yahan ek generated track ya listener.
- **Feature (`X`):** model ko diya input clue, jaise energy.
- **Target/label (`y`):** model ko sikhaya jaane wala answer, jaise streams ya skip.
- **Training:** model ke parameters ko known examples se seekhna.
- **Prediction/inference:** trained model ko naye input par chalana.
- **Model:** learned mapping jo input se output estimate karti hai.
- **Parameter:** training ke dauraan model khud seekhta hai, jaise linear-regression ka coefficient.
- **Hyperparameter:** training se pehle hum set/tune karte hain, jaise tree ki depth ya Ridge ka alpha.
- **Loss/error:** prediction aur actual answer ka difference.
- **Generalization:** naye/unseen data par bhi theek kaam karna.

### Supervised aur unsupervised learning

**Supervised learning** mein target labels diye hue hain. Stream regression aur skip classification supervised hain.

**Unsupervised learning** mein pre-existing answer label nahi diya jaata. K-means listener profiles ke similar groups khud dhoondhta hai. Clustering se milne wala group “sahi/galat” tabhi kaha ja sakta hai jab business ya human evaluation ho.

---

## 5. Regression: streams kyun log scale par predict hote hain?

Regression ka output number hota hai. Is project ka target `streams` hai, lekin training mein:

```python
y_reg = log1p(streams)
```

`log1p(x)` ka matlab `log(1 + x)`. 0 ke liye bhi valid hai aur bade counts ko compress karta hai.

### Chhota example

- 1,000 streams: `log1p(1000)` lagbhag 6.91.
- 100,000 streams: `log1p(100000)` lagbhag 11.51.

Original count mein 99,000 ka farq hai, log scale par farq lagbhag 4.60 hai. Isse ek-do mega-hit examples loss par poori tarah haavi nahi hote.

API predicted log value ko `expm1` se reverse karti hai:

```python
predicted_streams = round(expm1(predicted_log_streams))
```

**Important nuance:** `expm1(predicted_log)` aam taur par expected original stream count ka unbiased estimate nahi hota, kyunki log transform ke baad reverse transform karne mein statistical bias aa sakta hai. Demo ke liye yeh simple conversion hai.

### Regression metric: RMSE

RMSE = squared errors ka average lekar uska square root. Is project mein RMSE log-stream scale par hai.

Example, true log-streams `[10, 12]`, predicted `[11, 11]`:

```text
errors = [+1, -1]
RMSE = sqrt((1² + (-1)²) / 2) = 1
```

**RMSE chhota behtar.** Bade errors ko square karne ki wajah se RMSE unhe zyada punish karta hai. Log scale par RMSE ko original “streams ka error” mat samajhna.

### Noise floor / irreducible error

`generate_data.py` streams ke `log_streams` formula mein standard deviation **0.6** ka Gaussian random noise add karta hai. Formula ke doosre saare inputs pata hon tab bhi random noise ka exact value predict nahi ho sakta. Isliye is synthetic setup mein lagbhag **0.6 log-scale RMSE ek useful noise-floor reference** hai; `log1p` aur integer conversion ki wajah se exact theoretical number bilkul 0.6 hona zaroori nahi.

Real world mein noise floor ka exact number aam taur par pehle se maloom nahi hota. Isse samajhne ke liye repeat measurements, good domain knowledge, ya sahi experimental setup chahiye.

---

## 6. Regression models: kaun kya seekhta hai?

Trainer `scripts/train.py` mein kai candidate compare karta hai.

### OLS / Linear Regression

Ek straight-line relationship fit karta hai:

```text
predicted_log_streams =
  intercept
  + coefficient_1 × danceability
  + coefficient_2 × energy
  + ...
```

OLS (Ordinary Least Squares) squared errors ka sum chhota karne wale coefficients dhoondhta hai. Samajhne mein aasaan baseline hai. “Baseline” ka matlab pehla simple comparison point—complex model ko isse behtar hona chahiye.

### SGDRegressor / gradient descent

SGD (Stochastic Gradient Descent) examples/batches ke through chhote parameter updates karke loss kam karta hai. Ise pahadi par neeche jaane jaisa socho: slope dekhkar har step mein neeche ki taraf badhte hain. `max_iter=2000` maximum iterations ka hyperparameter hai.

### Ridge

Linear model ke coefficients ko bada hone par penalty deta hai:

```text
objective ≈ prediction_error + alpha × sum(coefficient²)
```

Ridge large/unstable coefficients ko **shrink** karta hai. Similar ya correlated features hon toh overfitting ka risk kam kar sakta hai. `alpha` penalty ki strength hai: zyada alpha, zyada shrinkage.

### Lasso

Lasso penalty absolute coefficient sizes par hoti hai:

```text
objective ≈ prediction_error + alpha × sum(abs(coefficient))
```

Yeh kuch coefficients ko bilkul zero kar sakta hai, yani feature selection jaisa effect. Par correlated inputs mein kaun zero hoga, woh kabhi unstable ho sakta hai.

### Scaling kyun?

`StandardScaler` feature ko roughly mean 0, standard deviation 1 scale par laata hai. Tempo 60–200 mein hai, jabki danceability 0–1 mein. Scaling khaas taur par Logistic/Ridge/Lasso/SGD aur distance-based clustering ke liye zaroori/upyogi hai.

**Data leakage se bachne ka tareeka:** scaler ko sirf training rows par `fit` karna chahiye; validation/test par bas `transform`. `make_pipeline(StandardScaler(), model)` cross-validation ke har training fold mein scaler ko sahi fit karta hai. Test ka mean/standard deviation training mein use nahi hona chahiye.

---

## 7. Classification: skip prediction

Classification label/category predict karta hai. Yahan label 0/1 hai. Logistic Regression pehle ek score banata hai aur sigmoid se probability:

```text
skip_probability = sigmoid(score)
```

Example: model 0.73 probability de, toh iska matlab training jaise cases mein lagbhag 73% ke aas-paas skip risk ka estimate—**iska matlab yeh nahi ki is particular play mein 73% ka hissa skip hoga.**

Demo is probability ko `0.5` threshold par label banata hai:

```text
probability >= 0.5  -> "skip"
probability < 0.5   -> "no-skip"
```

0.5 universal rule nahi hai. Product mein missed skip aur false alarm ki cost dekhkar threshold choose karna chahiye.

### Imbalance: accuracy akeli kyun kaafi nahi?

Skip rate 23.72%, yani no-skip examples zyada hain. Agar model **har baar no-skip** bole toh bhi accuracy 76.28% hogi, lekin model ek bhi skip nahi dhoondh payega! Isliye `class_weight="balanced"` minority class ke errors ko zyada weight deta hai.

### Confusion matrix aur metrics

| | Actual skip | Actual no-skip |
|---|---:|---:|
| Predicted skip | True Positive (TP) | False Positive (FP) |
| Predicted no-skip | False Negative (FN) | True Negative (TN) |

- **Precision = TP / (TP + FP):** jinhe skip kaha, unmein kitne waqai skip the? Kam false alarms chahiye toh dekho.
- **Recall = TP / (TP + FN):** actual skips mein se kitne pakde? Kam skips miss karne hain toh dekho.
- **F1:** precision aur recall ka harmonic mean; dono ko ek summary mein balance karta hai.
- **Accuracy:** sabhi sahi predictions ka fraction; imbalanced data mein akeli misleading ho sakti hai.
- **ROC-AUC:** positive example ko negative se zyada score karne ki ranking ability. 0.5 random-ranking level, 1.0 perfect ranking. AUC **probability calibration nahi** batata.
- **Brier score:** predicted probability aur actual 0/1 label ke squared difference ka mean. 0 best. Example: true skip ke liye p=0.8 ka loss `(0.8-1)²=0.04`; p=0.2 ka loss `0.64`.

### Skip ka irreducible error

Skip label mein generator random Gaussian noise add karta hai aur phir probability se random 0/1 sample leta hai. Isliye perfect feature knowledge ke baad bhi har individual play ka result certain nahi.

Theoretical **Bayes error**: agar inputs ke liye true skip probability `p` maloom ho, toh best possible 0/1 decision ka error `min(p, 1-p)` hota hai. Sab rows ke liye iska average ek theoretical lower bound hai. Is generator mein `p` bhi random logit noise integrate karke estimate karni padegi; yeh exact observed model metric nahi hai.

---

## 8. Tree models: Decision Tree, Random Forest, XGBoost

### Decision Tree

If/then questions se data ko split karta hai:

```text
Kya listener_avg_skip_rate > 0.4?
  Haan -> agla question...
  Nahi -> doosri branch...
```

`max_depth=4` tree ko bahut gehra hone se rokta hai. Shallow tree samjhana aasaan; deep tree training examples yaad karke overfit kar sakta hai.

### Random Forest

Bahut se decision trees ko alag bootstrapped samples par train karke unke predictions combine karta hai. Classification mein votes/probabilities average hote hain; regression mein numbers average hote hain.

- Aam taur par ek tree se zyada stable; zyada compute/model complexity.
- **OOB (Out-of-Bag) error:** har bootstrap sample se chhute rows ko kuch trees ke liye internal validation jaisa use karta hai. Yeh alag sealed test ka replacement nahi.
- **Permutation importance:** validation feature ke values ko shuffle karo. Agar ROC-AUC girta hai, feature useful signal de raha ho sakta hai. Correlated features importance ko baant sakte hain; importance cause-effect nahi batata.

### XGBoost

Gradient-boosted decision trees ka implementation. Ek ke baad ek naya chhota tree pichhle model ki errors kam karne ki koshish karta hai. Tabular data par strong candidate hai, par “hamesha best” nahi.

- `learning_rate`: har naye tree ka contribution kitna chhota ho.
- `max_depth`: har tree kitna complex ho.
- `n_estimators`: kitne boosting rounds/trees.
- `subsample`, `colsample_bytree`: training rows/features ka hissa use karke overfit risk kam karne mein madad.
- **Early stopping:** validation score sudharna band kare toh aur trees banana rok do. Test set dekhkar stopping point choose karna galat hoga.
- Skip XGBoost mein `scale_pos_weight` positive/negative class imbalance ka ek adjustment hai.

### SHAP

SHAP ek prediction ko features ke contribution mein explain karne ka tareeka hai. Udaharan: “is track ke model output mein high energy ne baseline ke muqable score ko upar dhakela; playlist position ne neeche.” Yeh model ke behavior ki explanation hai, **yeh sabit nahi karta ki energy badhane se real streams causally badhenge**.

---

## 9. Feature engineering

Feature engineering raw input se upyogi model input banana hai. `scripts/features.py` ek shared feature builder hai, jise training aur backend dono use karte hain. Isse same request training aur serving mein same tarah transform hoti hai.

12 raw columns ke saath 6 engineered columns milkar **18 track features** bante hain:

| Naya feature | Code ka idea | Kyun? |
|---|---|---|
| `energy_dance` | `energy × danceability` | Dono saath high hon toh interaction alag signal de sakta hai |
| `tempo_norm` | `(tempo - 120) / 40` | Tempo ko 120 BPM ke around center/scale karna |
| `is_night` | hour 20 ya uske baad, ya 5 tak | Time-of-day ko simple flag banana |
| `is_new_track` | release age <= 90 days | Nayi release ka flag |
| `position_early` | playlist position <= 10 | Shuru ke playlist slots ka flag |
| `taste_segment` | track K-means cluster ID | Cluster label ko supervised model ke input mein add karna |

**Training-serving skew** tab hota hai jab training mein ek feature formula aur live API mein doosra. Iska model ko pata nahi chalta ki same column ke numbers ka meaning badal gaya. Shared feature module is risk ko kam karta hai.

---

## 10. Data splits, CV aur leakage

Data ko alag rakhna isliye zaroori hai taaki humein pata chale model **unseen examples** par kaam karta hai ya sirf training data yaad karta hai.

- **Train set:** coefficients/tree/model fit hote hain.
- **Validation set:** hyperparameters aur candidate models compare karte hain.
- **Test set:** aakhir mein final generalization estimate; ideally ek hi baar.

Is trainer ka track split 80% development data / 20% held-out test hai; skip label ke hisaab se stratify hota hai. Development portion se validation slice aur 5-fold CV use hote hain. Candidate winner validation par select hota hai; test ka final metric baad mein nikala jaata hai.

### Data leakage

Leakage = model ko aisi information mil jana jo real prediction time par available nahi, ya evaluation data training decisions mein chori se aa jaana.

**Example:** exam se pehle answer key test kar lena. Test score achha dikhega, lekin student ki actual knowledge ka fair measure nahi.

Is project mein test rows ko supervised model selection se alag rakha gaya hai. Track taste K-means scaler/centroids bhi train portion par fit hote hain; phir test par `predict` se labels milte hain. Listener clustering alag unlabeled table par exploratory hai—wahan labeled cluster-test target define nahi kiya gaya.

### Cross-validation

5-fold CV data ko 5 hisson mein baantti hai. Har round mein 4 folds par train, 1 fold par score; har hissa ek baar validation hota hai. Isse ek lucky split par bharosa kam hota hai.

### One-standard-error rule

Ridge ke `alpha` values 0.01, 0.1, 1, 10, 100 compare hote hain. Sabse kam average CV RMSE wala alpha “best” hai. One-SE rule best score ke ek standard error ke andar aane wale options mein **sabse zyada regularized/simple** model leta hai. Agar scores lagbhag barabar, toh simpler/stronger regularization model pasand.

### Nested CV

Nested CV ke **inner folds** hyperparameter choose karte hain, **outer folds** us poore selection process ka score estimate karte hain. Isse tuning ke baad overly optimistic estimate ka risk kam hota hai.

Is code mein nested CV specifically Ridge-alpha tuning ke performance estimate ke liye hai; har alag model family ki poori nested selection nahi. Review mein is scope ko accurately batana.

---

## 11. Clustering aur dimensionality reduction

### K-means

`k=4` group centers choose karta hai aur har listener ko nearest center se assign karta hai. Distance meaningful banane ke liye data pehle StandardScaler se scale hota hai.

**Analogy:** 2D map par 500 logon ko unki music preference ke nazdeek chaar meeting spots mein baantna. K-means ko pehle se `k` batana padta hai; yeh spheres/compact groups ko prefer karta hai, outliers ke liye sensitive ho sakta hai.

Is project mein **do alag K-means uses** hain:

1. Track audio dimensions par ek K-means; uska label `taste_segment` feature ban kar stream/skip model ko diya jaata hai.
2. Listener ke paanch average taste dimensions par doosra K-means; `/segment` endpoint isi se listener group nikalta hai.

Algorithm khud “Party” ya “Chill” naam nahi jaanta. Code har listener cluster ke average center ko synthetic archetype centers se compare karke nearest human-readable naam deta hai. Names simulation ke labels hain, discovered human truth nahi.

### DBSCAN

Density-Based Spatial Clustering of Applications with Noise: aas-paas kaafi points hon toh cluster, warna point ko noise/outlier label `-1`. K-means se farq: cluster count pehle dena zaroori nahi; `eps` aur `min_samples` jaise settings dene padte hain. Features scale karna important hai, warna BPM numeric distance par sab kuch dominate kar sakta hai.

### PCA

Principal Component Analysis paanch ya zyada correlated measurements ko naye axes mein compress karta hai. Code listener data ko **2 dimensions** mein project karke batata hai ki un 2 axes mein kitni variance explain hui. PCA plotting/structure samajhne mein madad karta hai; do components ko asli “taste causes” mat samjho.

---

## 12. Calibration, significance aur model comparison

### Probability calibration

Classifier 0.8 bolta hai—kya aise predictions waale lagbhag 80% examples waqai positive hote hain? Aisa hona calibration hai. **ROC-AUC ranking** check karta hai; **Brier score calibration/probability error** par roshni daalta hai. Dono alag sawal hain.

`CalibratedClassifierCV(method="sigmoid")` cross-validation se raw scores ko probability scale par calibrate karne ki koshish karta hai. Calibration ko held-out data par check karna chahiye.

### Significance test

Code paired per-example loss compare karke t-test karta hai. **p-value** ek statistical assumption ke andar dekhti hai: agar average difference asal mein zero ho, itna ya isse zyada extreme difference kitna plausible hoga?

- Chhoti p-value = is comparison mein zero-difference ke khilaaf evidence.
- Yeh prove nahi karti ki model important/useful/fair hai.
- Large sample mein tiny improvement bhi significant ho sakti hai.
- Sirf ek test set aur assumptions par overclaim nahi karna.

Saved `models/version.json` par **2026-10-07** ka training snapshot likha hai:

- saved champion regressor: **Lasso**, test log-RMSE **0.5995**
- saved champion classifier: **Logistic**, test ROC-AUC **0.6841**
- calibrated Brier **0.1622**
- nested-CV Ridge RMSE **0.6012**

Yeh stored snapshot ke numbers hain—**fresh retraining ke baad ke guaranteed metrics nahi**. 0.6841 AUC ranking mein random 0.5 se better, par perfect nahi. Regression ~0.6 error known ~0.6 synthetic noise scale ke paas hai, jo simulator mein achha signal-learning dikhata hai; real-world performance ka proof nahi.

---

## 13. MLflow kya hai?

MLflow machine-learning experiments ka **record book** hai. Har training run ke liye useful information jama kar sakta hai:

- **Parameters:** model type, alpha, depth jaise settings.
- **Metrics:** validation RMSE/AUC, test metrics, etc.
- **Tags/artifacts:** run metadata, files ya model—agar logging code unhe save kare.

Is project ka `log_run()` named experiment `PlaylistPulse` mein parameters aur metrics log karta hai. MLflow import/logging fail ho, toh demo `models/experiments.json` mein JSON fallback likhta hai. Current code metrics/params log karta hai; automatically har candidate ka model file MLflow Registry mein publish nahi karta.

MLflow:

- experiments compare karna aasaan karta hai;
- khud best model choose, leakage fix, ya data validate nahi karta;
- synthetic data ko real nahi banata.

Agar locally training ki hai, `Programs` folder se MLflow UI:

```powershell
mlflow ui --host 127.0.0.1 --port 5000
```

Phir `http://127.0.0.1:5000` kholo. Remote/Render deploy ke MLflow runs tabhi dikhenge jab tracking store configured ho aur tum us store ko access karo; local UI automatically cloud runs nahi dikhata.

---

## 14. Model artifact, FastAPI aur prediction ka poora raasta

### Training ke baad files

`scripts/train.py` selected models aur metadata `models/` mein rakhta hai:

- `regressor.pkl`: stream log-target predictor.
- `classifier.pkl`: skip probability predictor (calibrated wrapper).
- `taste_kmeans.pkl`: track taste features banane ke liye.
- `kmeans.pkl`: listener segmentation.
- `features.json`: model ke input feature columns/order.
- `cluster_labels.json`: cluster ID se display name/description.
- `version.json`: version, date, chosen model names aur metrics.
- `psi_ref.pkl`: training feature distribution reference (trainer ise save karta hai).

Saved model ko **artifact** kehte hain: training ke baad inference ke kaam aane wali file. Pickle/joblib files ko sirf trusted source se load karo.

### FastAPI request

1. Browser sliders se JSON body banata hai.
2. FastAPI/Pydantic types aur range validate karta hai. Jaise `danceability` 0–1; tempo 60–200. Galat request ko validation error milta hai.
3. Shared `engineer_features()` raw inputs ko 18 model inputs mein badalta hai.
4. `regressor.pkl` log-stream estimate deta hai; API `expm1` se approximate stream count banati hai.
5. `classifier.pkl` skip probability deta hai; API 0.5 threshold se demo label banati hai.
6. JSON response browser ko lautata hai aur UI numbers/bar render karta hai.

### Endpoints

- `GET /health`: model files load hue hain ya nahi, model version.
- `POST /predict`: expected streams + skip probability.
- `POST /segment`: listener profile ke taste cluster ka naam/ID.
- `GET /metrics`: request/error counts, average prediction latency, drift status.
- `POST /retrain`: demo trigger; `RETRAIN_API_KEY` configured aur `X-Retrain-Token` sahi ho tabhi allowed. Hosted public app mein token unset ho toh disabled.
- `GET /docs`: FastAPI ka interactive API documentation.

Example request:

```json
{
  "danceability": 0.75,
  "energy": 0.85,
  "tempo": 128,
  "valence": 0.7,
  "acousticness": 0.15,
  "speechiness": 0.05,
  "instrumentalness": 0.0,
  "playlist_position": 3,
  "hour_of_day": 20,
  "is_playlist_radio": 0,
  "listener_avg_skip_rate": 0.25,
  "days_since_release": 60
}
```

Response mein `predicted_streams`, `skip_probability`, `skip_prediction`, `model_version` aate hain.

### Deployment ka simple map

Frontend Vercel par static React site hai; backend Render par model files load karke API deta hai. Frontend ka `VITE_API_URL` Render URL ko batata hai; Render ka `CORS_ORIGINS` Vercel website ko browser se API call karne ki permission deta hai. Docker alag se app + dependencies ko reproducible container mein package karta hai. **Deployment ka matlab model achha hai, yeh nahi; woh evaluation ka kaam hai.**

---

## 15. PSI aur drift

**Drift** ka matlab live inputs ka pattern training data se badalna. Udaharan: training mein playlist position aksar 1–50 thi, lekin naye usage mein almost sab requests 40–50 positions se aa rahi hain. Purane model ke patterns kam reliable ho sakte hain.

**PSI = Population Stability Index** reference distribution aur recent request distribution ka ek comparison. Code training reference ke bins banata hai, phir recent values ke proportions compare karta hai. Demo thresholds:

```text
PSI < 0.10      stable signal
0.10–0.25       moderate warning
> 0.25          major warning
```

Yeh rough monitoring rules hain, universal laws nahi. PSI **feature drift** batata hai, target accuracy nahi. Drift alert aane ka matlab “automatic retrain zaroori” nahi; labelled ground truth, business checks aur approval chahiye. Is demo mein recent buffer max 200 values rakhta hai aur 20 se kam observations hon toh PSI calculate nahi karta.

Production mein data distribution ke saath **model performance** bhi monitor karo—jaise baad mein actual skip labels milne par AUC/Brier. API ka `/retrain` endpoint simple demo hai; real retraining ko validate, approve, version aur safely deploy karna chahiye.

---

## 16. Code map: kis file mein kya hai?

| File | Role |
|---|---|
| `scripts/generate_data.py` | Synthetic features, stream target aur skip label banata hai |
| `data/tracks.csv` | Generated track rows |
| `data/listeners.csv` | Generated listener taste profiles |
| `scripts/features.py` | Training aur API ke beech shared feature engineering |
| `scripts/train.py` | Splits, baselines, candidate models, evaluation, MLflow logging, artifact save |
| `models/*.pkl` | Saved trained model objects |
| `models/version.json` | Saved model version/metrics ka snapshot |
| `backend/main.py` | FastAPI, request validation, predictions, metrics/drift |
| `frontend/src/App.jsx` | React sliders, API calls, results display |
| `frontend/src/index.css` | Light-blue responsive interface styles |
| `Dockerfile`, `compose.yaml` | Backend/frontend services ko locally container mein chalana |

Local training dobara chalane ke liye `Programs` se:

```powershell
python scripts\generate_data.py
python scripts\train.py
```

Training mein cross-validation, random forests, XGBoost, SHAP aur nested CV hain, isliye model ko serve karne se zyada waqt training mein lag sakta hai. Current live Render model ek saved snapshot hai; local source ya artifacts change karne se hosted service tab tak nahi badlegi jab tak usse redeploy/update na karo.

---

## 17. Instructor ke saamne 90-second explanation

> “PlaylistPulse ek learning-focused music ML prototype hai. Ismein teen tasks hain: log-transformed stream count ki regression, 30-second skip ki probability classification, aur listener preferences ki unsupervised clustering. Track data abhi synthetic hai: 5,000 generated track examples, 500 listener profiles, isliye metrics ko real Spotify accuracy nahi keh sakte. Stream count heavy-tailed hone ke kaaran training `log1p(streams)` par hoti hai aur API output ko `expm1` se approximate count mein convert karti hai. Hum OLS/SGD baseline se shuru karke Ridge/Lasso, Logistic Regression, Decision Tree, Random Forest aur XGBoost compare karte hain. Candidate selection validation data par hoti hai; held-out test set final evaluation ke liye hota hai. K-means listeners ke similar taste profiles group karta hai; DBSCAN density-based alternative hai aur PCA dimensionality reduce karta hai. MLflow experiment settings/metrics record karta hai. FastAPI saved model artifacts load karke `/predict`, `/segment`, `/health` aur `/metrics` provide karta hai. Shared feature module training-serving consistency deta hai; PSI request-feature drift ka alert deta hai. Sabse badi next step real, consented listening logs aur time-aware evaluation hai.”

Isko ratta nahi—har jumle ka matlab samjho. Instructor follow-up kare toh neeche ke jawab use karo.

---

## 18. Viva ke sambhavit sawal

**Q: Regression aur classification mein farq?**

A: Regression number estimate karti hai—jaise streams. Classification class ya positive-class probability estimate karti hai—jaise 30 seconds mein skip.

**Q: Streams ko log kyun kiya?**

A: Distribution skewed/heavy-tailed hai. Log scale bade counts ko compress karta hai aur model ko typical aur huge tracks dono se seekhne deta hai.

**Q: Accuracy kyun nahi kaafi?**

A: Skip label sirf ~23.72% rows mein hai. Har row ko no-skip kehne par bhi ~76.28% accuracy aa jaati, par actual skips mein se koi nahi milta. Isliye precision/recall/F1/ROC-AUC dekhen.

**Q: ROC-AUC aur Brier mein farq?**

A: ROC-AUC probability ki ranking quality dekhta hai; Brier predicted probability aur 0/1 outcome ka squared error dekhta hai. AUC achha hone se calibration pakki nahi.

**Q: Ridge aur Lasso?**

A: Dono linear regression ko regularize karte hain. Ridge coefficients shrink karta hai; Lasso kuch coefficients zero bhi kar sakta hai.

**Q: Random Forest vs XGBoost?**

A: Forest parallel bootstrapped trees average karta hai; XGBoost trees ko sequentially errors improve karne ke liye add karta hai. Dono nonlinear interactions seekh sakte hain; winner validation par decide hota hai.

**Q: Overfitting?**

A: Model training examples ko bahut closely fit/yaad kar leta hai, par unseen examples par kharab karta hai. Train/validation/test splits, regularization, shallow tree, CV aur early stopping is risk ko measure/control karte hain.

**Q: Test set ko tuning mein kyun nahi lena?**

A: Test par baar-baar model choose karna test ko training decision ka hissa bana deta hai. Phir test score independent final estimate nahi rehta.

**Q: K-means ko labels kisne diye?**

A: K-means labels nahi dekhta—distance se 4 groups banata hai. Human-readable names baad mein synthetic archetype centers ke nearest match se assign hote hain.

**Q: MLflow kya solve karta hai?**

A: Run settings/metrics track aur compare karta hai; data quality, model correctness ya leakage apne aap solve nahi karta.

**Q: PSI > 0.25 ka matlab?**

A: Is demo ke rule mein noticeable feature distribution drift warning. Iska matlab prediction galat sabit nahi; labels/performance investigate karni chahiye.

**Q: Is model ko real Spotify par kyun nahi keh sakte?**

A: Training data synthetic hai. Real user behavior, artist/playlist effects, time, label quality aur privacy/consent represent nahi hote. Real data par naya training/evaluation zaroori hai.

**Q: Next improvement kya hogi?**

A: Consent ke saath real, timestamped play/skip events collect; listener/track/time grouping ke hisaab se leakage-safe temporal split; production baseline ke against test; probability calibration/threshold business cost ke hisaab se; privacy, monitoring aur approved model versioning.

---

## 19. Yaad rakhne waali 5 baatein

1. **Features inputs hain; targets answers hain.**
2. **Streams = regression; skip = classification; listener grouping = clustering.**
3. **Train fit karta hai, validation choose karta hai, test final check karta hai.**
4. **ROC-AUC ranking hai; Brier probability quality hai; PSI feature drift hai.**
5. **Synthetic data par achha score real-world success ka proof nahi.**
