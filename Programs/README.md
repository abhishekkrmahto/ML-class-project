File	                    Rubric Weeks	                    Kaam
scripts/generate_data.py	W1–W2	                     Data + targets + EDA stats
scripts/train.py	        W3–W10	                     Baselines → Ridge/Lasso → CV → Trees → RF → XGBoost+SHAP → Clustering → Nested CV
backend/main.py	            W11–W12	                     Versioned artefact load → PSI drift → FastAPI → /metrics → /retrain

✅ W1-W2  → generate_data.py + data/*.csv
✅ W3     → train.py (OLS + SGD + MLflow)          → console output
✅ W4     → train.py (Ridge/Lasso + logistic + engineering)
✅ W5     → train.py (5-fold CV + 1-SE rule)
✅ W6     → train.py (decision tree + top splits)
✅ W7     → train.py (RF + OOB + permutation)
✅ W8     → train.py (XGBoost + early stopping + SHAP)
✅ W9     → train.py (KMeans + DBSCAN + PCA)
✅ W10    → train.py (nested CV + calibration + significance)
✅ W11    → main.py (version.json + psi_ref) + models/*.pkl
✅ W12    → main.py (5 endpoints + drift + retrain)



---

## 3. How to Run

```bash
pip install -r requirements.txt
python scripts/generate_data.py
python scripts/train.py
python -m uvicorn backend.main:app --reload     # API: localhost:8000

# Terminal 2:
cd frontend && npm install && npm run dev        # UI: localhost:5173

