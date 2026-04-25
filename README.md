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
| Date | Model/Feat | CV Acc | Public LB | Top % | Commit |
|------|------------|--------|-----------|-------|--------|
| 2026-04-25 | XGBoost Baseline (median fill, label encode) | 0.802 ± 0.008 | 0.799 | 15% | [spaceship_baseline.ipynb](spaceship_baseline.ipynb) |
| 2026-04-25 | LGB+XGB Ensemble (30 feat: cabin/group/spend/log/ratio) | 0.807 ± 0.007 | 0.796 | ~20% | [build_model.py](build_model.py) |

## Files
- `train.csv`, `test.csv`, `sample_submission.csv`
- `spaceship_baseline.ipynb`: XGBoost baseline notebook (10 features, CV 0.802, LB 0.799)
- `build_model.py`: LGB+XGB ensemble model (30 features, CV 0.807, LB 0.796)
- `submission.csv`: LB 0.796 sub (latest)

## Workflow
1. `git clone https://github.com/ljding94/spaceship-titanic-Ding.git`
2. Edit notebook (e.g. cabin parse `deck/side/num`, group size from PassengerId, spend total, age bins).
3. `git add .; git commit -m "Cabin feats +0.02"; git push`
4. Zane auto-review/merge, retrain/submit team LB (cron).

**Cron**: Mon 10AM auto-submit + git sync (ID: ab7e5317341a).

**Target**: Top 10% (0.82 LB – cabin/group next). Edit/push your tune! 🎯