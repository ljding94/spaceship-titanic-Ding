# Spaceship Titanic — Project Progress

## Last Updated
2026-04-28

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
| Stacking (LGB+XGB+CatBoost→LR) | 0.813 | 0.80780 |
| CatBoost | 0.813 | 0.80710 |
| Optuna LGB (80 trials) | 0.818 | 0.80056 |
| Optuna XGB | 0.817 | TBD |
| Baseline XGB | 0.802 | 0.799 |
| auto-round 20260428 | TBD | 0.80687 |

- Best LB: 0.80780 (~8% top)
- Target: 0.82 LB (top 10%)

### 🚧 In Progress
- Nightly auto-improve cron running (6 AM)
- Monday 10 AM LB check cron
- NOTE: cron needs KAGGLE_API_TOKEN env var — read from ~/.zshrc via `source ~/.zshrc` or set explicitly

### ⏳ Pending
- Submit Optuna XGB (CV 0.817) → needs LB score
- Reach 0.82 LB (top 10%)
- Simplify features if overfitting persists

## Project Notes
- Data: 8,693 train rows, 4,277 test rows
- Best approach so far: stacking ensemble
- Auth fix: set KAGGLE_API_TOKEN explicitly (empty env var overrides kaggle.json)
- Working kaggle CLI: ~/.local/pipx/venvs/kaggle/bin/kaggle (2.1.0)

## Key Files
- auto_improve.py     → nightly auto-train+submit pipeline
- build_model.py      → LGB+XGB ensemble
- stacking_model.py  → stacking ensemble (best LB)
- catboost_model.py   → CatBoost variant
- optuna_tuning.py   → hyperparameter search
