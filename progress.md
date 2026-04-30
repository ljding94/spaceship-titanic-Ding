# Spaceship Titanic — Project Progress

## Last Updated
2026-04-30 (02:18 AM)

## Competition
- Kaggle: Spaceship Titanic (Getting Started)
- Team: ZaneLijieTitanic (znding04 + ljding94)
- Repo: github.com/ljding94/spaceship-titanic-Ding

## Current Status

### ✅ Completed
- Multiple models trained and submitted
- Kaggle auth fixed (Bearer token via KAGGLE_API_TOKEN env var)
- auto_improve.py overhauled with mandatory run+submit+README-update steps

### 📊 LB Scores
| Model | CV | LB |
|-------|----|-----|
| **auto-round 20260430 (feature selection v2)** | TBD | **0.80944** 🆕 |
| auto-round 20260429 (pseudo-label stacking) | 0.8135 | 0.80897 |
| Stacking (LGB+XGB+CatBoost→LR) | 0.813 | 0.80780 |
| CatBoost | 0.813 | 0.80710 |
| Optuna LGB (80 trials) | 0.818 | 0.80056 |
| Baseline XGB | 0.802 | 0.799 |
| auto-round 20260428 | TBD | 0.80687 |

- Best LB: 0.80944 (~top 4-5%)
- Target: 0.82 LB (top 10%)

### 🚧 In Progress
- Nightly auto-improve cron running (6 AM)
- Monday 10 AM LB check cron
- NOTE: cron needs KAGGLE_API_TOKEN env var — read from ~/.zshrc via `source ~/.zshrc` or set explicitly

### ⏳ Pending
- ~~Submit Optuna XGB (CV 0.817)~~ → LB confirmed 0.80056
- Reach 0.82 LB (top 5%)
- Simplify features if overfitting persists

## Project Notes
- Data: 8,693 train rows, 4,277 test rows
- Best approach so far: stacking ensemble with pseudo-labeling
- Auth fix: set KAGGLE_API_TOKEN explicitly (empty env var overrides kaggle.json)
- Working kaggle CLI: ~/.local/pipx/venvs/kaggle/bin/kaggle (2.1.0)

## Round Log
- **2026-04-30**: Claude Code feature selection + regularization attempt. Feature selection v2 with 24 features + stronger regularization + threshold tuning (0.54). LB = **0.80944** — **NEW BEST** (+0.00047). Created feature_selection_v2.py and interaction_features_v2.py.
- **2026-04-29**: auto_improve.py ran pseudo-label stacking (R1+R2 blend, 5-seed LGB+XGB+CatBoost). LB = 0.80897 — **previous best**. Used pseudo-labeling with refined stacking v3. CV=0.8135.
- **2026-04-28**: auto-round baseline. LB = 0.80687.
- **2026-04-27**: Stacking ensemble (LGB+XGB+CatBoost→LR) with CV 0.813. LB = 0.80780.

## Key Files
- auto_improve.py     → nightly auto-train+submit pipeline
- build_model.py      → LGB+XGB ensemble
- stacking_model.py  → stacking ensemble (previous best LB)
- catboost_model.py   → CatBoost variant
- optuna_tuning.py   → hyperparameter search
- feature_selection_v2.py → feature selection + regularization (new best LB)
- interaction_features_v2.py → interaction features attempt
