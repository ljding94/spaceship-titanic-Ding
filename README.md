# Spaceship Titanic - Zane + Lijie Kaggle Team 🚀

## Competition Overview
[Kaggle Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) (Getting Started, ongoing leaderboard).

**Task**: Predict which passengers were **transported to an alternate dimension** after the Spaceship Titanic hit a spacetime anomaly in 2912. (Binary classification: Transported True/False ~50/50 balanced).

**Data** (train.csv 8693 rows × 14 cols):
- **Features**: PassengerId (group/ticket), HomePlanet (Earth/Europa/Mars), CryoSleep (bool), Cabin (deck/num/side), Destination, Age (0-79), VIP (bool), spend cols (RoomService, FoodCourt, ShoppingMall, Spa, VRDeck – 0-10k), Name.
- **Target**: Transported (bool).
- `test.csv`: 4277 rows (submit PassengerId + pred).
- **Metric**: Accuracy (public LB averages subs).

**Team**: ZaneLijieTitanic (`znding04` + `ljding94`).

## LB Tracker
| Date | Model/Feat | CV Acc | Public LB | Top % | Notes |
|------|------------|--------|-----------|-------|-------|
| 2026-04-25 | XGBoost Baseline (median fill, label encode) | 0.802 ± 0.008 | 0.799 | 15% | [baseline_xgb.py](baseline_xgb.py) |
| 2026-04-25 | LGB+XGB Ensemble (30 feat: cabin/group/spend/log/ratio) | 0.807 ± 0.007 | 0.796 | ~20% | [build_model.py](build_model.py) |
| 2026-04-26 | Minimal-feat LGB+XGB multi-seed ensemble (15 feat) | 0.804 | TBD | TBD | [minimal_ensemble.py](minimal_ensemble.py) |
| 2026-04-26 | MLP Neural Net (15 feat) | 0.792 | - | - | [mlp_xgb_experiment.py](mlp_xgb_experiment.py) |
| 2026-04-26 | Stacking (LGB+XGB+CatBoost→LR) + smart imputation (21 feat) | 0.813 | **0.808** | ~8% | [stacking_model.py](stacking_model.py) |
| 2026-04-26 | CatBoost native categoricals (21 feat) | 0.813 | **0.807** | ~9% | [catboost_model.py](catboost_model.py) |
| 2026-04-26 | Optuna LGB tuned (80 trials, baseline feat) | 0.818 | **0.808** | ~8% | [optuna_tuning.py](optuna_tuning.py) |
| 2026-04-26 | Optuna XGB tuned (80 trials, baseline feat) | 0.817 | TBD | TBD | [optuna_tuning.py](optuna_tuning.py) |
| **2026-04-27** | **Optuna LGB + Stacking (best LB)** | **0.818** | **0.808** | **~8%** | **New best!** |

## Files
- `train.csv`, `test.csv`, `sample_submission.csv`
- `baseline_xgb.py`: Simple XGBoost baseline (10 features, CV 0.802, LB 0.799)
- `build_model.py`: LGB+XGB ensemble model (30 features, CV 0.807, LB 0.796)
- `minimal_ensemble.py`: Minimal-feature LGB+XGB multi-seed ensemble (15 features, CV 0.804)
- `mlp_xgb_experiment.py`: MLP + regularized XGB experiment
- `stacking_model.py`: Stacking ensemble with smart domain-aware imputation (CV 0.813)
- `optuna_tuning.py`: Optuna hyperparameter tuning for LGB+XGB (LGB CV 0.818, XGB CV 0.817)
- `submission.csv`: Latest submission (stacking ensemble)

## Workflow
1. `git clone https://github.com/ljding94/spaceship-titanic-Ding.git`
2. Edit notebook (e.g. cabin parse `deck/side/num`, group size from PassengerId, spend total, age bins).
3. `git add .; git commit -m "Cabin feats +0.02"; git push`
4. Zane auto-review/merge, retrain/submit team LB (cron).

**Cron**: Mon 10AM auto-submit + git sync (ID: ab7e5317341a).

**Target**: Top 10% (0.82 LB – cabin/group next). Edit/push your tune! 🎯