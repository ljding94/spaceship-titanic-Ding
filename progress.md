# Spaceship Titanic — Project Progress

## Last Updated
2026-05-02 (06:11 AM)

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
| **auto-round 20260502 (ImprovedBlend v5, CV 0.817)** | 0.817 | **0.807** 🆕 |
| auto-round 20260430 (feature selection v2) | 0.814 | 0.809 |
| auto-round 20260429 (pseudo-label stacking) | 0.814 | 0.809 |
| Stacking (LGB+XGB+CatBoost→LR) | 0.813 | 0.808 |
| CatBoost | 0.813 | 0.807 |
| Optuna LGB (80 trials) | 0.818 | 0.801 |
| Baseline XGB | 0.802 | 0.799 |

- Best LB: 0.810 (~top 3-4%)
- Target: 0.82 LB (top 10%)

### 🚧 In Progress
- Nightly auto-improve cron running (6 AM)
- Monday 10 AM LB check cron
- NOTE: cron needs KAGGLE_API_TOKEN env var — read from ~/.zshrc via `source ~/.zshrc` or set explicitly

### ⏳ Pending
- Reach 0.82 LB (top 10%)
- Fix shell escaping bug in auto_improve.py round 3 (the `don't` in task string causes syntax error)
- Simplify features if overfitting persists

## Project Notes
- Data: 8,693 train rows, 4,277 test rows
- Best approach so far: CatBoost with native categorical handling + weighted blend (75% CatBoost)
- Key insight: CatBoost native ordered target encoding beats label encoding
- More features = overfitting (smart blend v5 with 40 feat scored worse)
- Threshold 0.5 generalizes better than tuned thresholds
- We're near a plateau — 0.810 barrier is hard to break

## Round Log
- **2026-05-02**: auto_improve.py ran improved_blend_v5.py (4-model ensemble LGB+XGB+CatBoost+HGB, stronger reg, 7 seeds, pseudo-labels, CV 0.817). LB = **0.80664** — REGRESSION from best 0.810. Created multi_catboost_v6.py but didn't run it. Submission made but no improvement.
- **2026-05-01**: auto_improve.py ran 3 rounds:
  - **Round 1**: NativeCat v4 (CatBoost native cats + weighted blend 75% CatBoost + pseudo-label + 5 seeds). LB = **0.80991** — NEW BEST! Committed.
  - **Round 2**: Tried heavy reg XGB, smart blend v5 (40 feat), rank blend v6. Best was rank v6 CatBoost 75% → LB 0.80967 (didn't beat round 1). No commit.
  - **Round 3**: FAILED — shell escaping bug in auto_improve.py (the `don't` in the task string caused a syntax error). Needs fix.
- **2026-04-30**: Claude Code feature selection + regularization attempt. Feature selection v2 with 24 features + stronger regularization + threshold tuning (0.54). LB = **0.80944** — previous best. Created feature_selection_v2.py and interaction_features_v2.py.
- **2026-04-29**: auto_improve.py ran pseudo-label stacking (R1+R2 blend, 5-seed LGB+XGB+CatBoost). LB = 0.80897.
- **2026-04-28**: auto-round baseline. LB = 0.80687.
- **2026-04-27**: Stacking ensemble (LGB+XGB+CatBoost→LR) with CV 0.813. LB = 0.80780.

## Key Files
- auto_improve.py     → nightly auto-train+submit pipeline (3 rounds/night)
- native_cat_blend.py → current best: CatBoost native cats + weighted blend (LB 0.810)
- build_model.py      → LGB+XGB ensemble
- stacking_model.py  → stacking ensemble
- catboost_model.py   → CatBoost variant
- optuna_tuning.py   → hyperparameter search
- feature_selection_v2.py → feature selection + regularization
- heavy_reg_xgb.py    → heavy regularization XGB (didn't help)
- rank_blend_v6.py    → rank-based blending attempt (LB 0.80967)
- smart_blend_v5.py   → smart blend with 40 features (overfit, LB 0.806)

## Known Bugs
- auto_improve.py round 3 task string has shell escaping issue with `don't` → fix needed
