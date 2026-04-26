"""
Spaceship Titanic — Optuna hyperparameter tuning for XGBoost + LightGBM
Uses the same baseline feature set from baseline_xgb.py.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import xgboost as xgb
import lightgbm as lgb
import optuna
import warnings
import json

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("=" * 60)
print("SPACESHIP TITANIC — Optuna Hyperparameter Tuning")
print("=" * 60)

# ── Data loading & feature engineering (identical to baseline) ───────────────

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


def baseline_features(df):
    df = df.copy()
    df["Deck"] = df["Cabin"].str.split("/").str[0]
    df["CabinNum"] = df["Cabin"].str.split("/").str[1].astype(float)
    df["Side"] = df["Cabin"].str.split("/").str[2]
    df["GroupId"] = df["PassengerId"].str.split("_").str[0].astype(int)
    df["PersonNum"] = df["PassengerId"].str.split("_").str[1].astype(int)
    df["GroupSize"] = df.groupby("GroupId")["PassengerId"].transform("count")
    df["IsSolo"] = (df["GroupSize"] == 1).astype(int)
    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    df["TotalSpend"] = df[spend_cols].sum(axis=1)
    df["AgeBin"] = pd.cut(
        df["Age"],
        bins=[0, 12, 18, 25, 35, 50, 65, 100],
        labels=[0, 1, 2, 3, 4, 5, 6],
    ).astype(float)
    df["CryoSleep"] = df["CryoSleep"].map(
        {True: 1, False: 0, "True": 1, "False": 0}
    )
    df["VIP"] = df["VIP"].map({True: 1, False: 0, "True": 1, "False": 0})
    for col in ["HomePlanet", "Destination", "Deck", "Side"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df


train["is_train"] = 1
test["is_train"] = 0
test["Transported"] = np.nan
combined = pd.concat([train, test], axis=0, ignore_index=True)

spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
for col in spend_cols:
    combined[col] = combined[col].fillna(combined[col].median())
combined["Age"] = combined["Age"].fillna(combined["Age"].median())
combined["HomePlanet"] = combined["HomePlanet"].fillna(combined["HomePlanet"].mode()[0])
combined["Destination"] = combined["Destination"].fillna(
    combined["Destination"].mode()[0]
)
combined["CryoSleep"] = combined["CryoSleep"].fillna(combined["CryoSleep"].mode()[0])
combined["VIP"] = combined["VIP"].fillna(False)

combined = baseline_features(combined)

train_df = combined[combined["is_train"] == 1].copy()
test_df = combined[combined["is_train"] == 0].copy()

y = train_df["Transported"].astype(int)
drop_cols = ["PassengerId", "Cabin", "Name", "Transported", "is_train", "GroupId"]
feature_cols = [c for c in train_df.columns if c not in drop_cols]
print(f"Features ({len(feature_cols)}): {feature_cols}")

X = train_df[feature_cols].values
X_test = test_df[feature_cols].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ── Optuna objectives ───────────────────────────────────────────────────────

def xgb_objective(trial):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "error",
        "tree_method": "hist",
        "verbosity": 0,
        "random_state": 42,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "early_stopping_rounds": 50,
    }
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        model = xgb.XGBClassifier(**params)
        model.fit(
            X[train_idx],
            y.values[train_idx],
            eval_set=[(X[val_idx], y.values[val_idx])],
            verbose=False,
        )
        pred = model.predict(X[val_idx])
        scores.append(accuracy_score(y.values[val_idx], pred))
    return np.mean(scores)


def lgb_objective(trial):
    params = {
        "objective": "binary",
        "metric": "binary_error",
        "verbosity": -1,
        "random_state": 42,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.1, 10.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
    }
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X[train_idx],
            y.values[train_idx],
            eval_set=[(X[val_idx], y.values[val_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        pred = model.predict(X[val_idx])
        scores.append(accuracy_score(y.values[val_idx], pred))
    return np.mean(scores)


# ── Run studies ──────────────────────────────────────────────────────────────

N_TRIALS = 80

print(f"\n{'─'*60}")
print(f"XGBoost Optuna study ({N_TRIALS} trials)...")
xgb_study = optuna.create_study(direction="maximize", study_name="xgb")
xgb_study.optimize(xgb_objective, n_trials=N_TRIALS, show_progress_bar=True)
xgb_best = xgb_study.best_value
print(f"XGBoost best CV: {xgb_best:.5f}  params: {xgb_study.best_params}")

print(f"\n{'─'*60}")
print(f"LightGBM Optuna study ({N_TRIALS} trials)...")
lgb_study = optuna.create_study(direction="maximize", study_name="lgb")
lgb_study.optimize(lgb_objective, n_trials=N_TRIALS, show_progress_bar=True)
lgb_best = lgb_study.best_value
print(f"LightGBM best CV: {lgb_best:.5f}  params: {lgb_study.best_params}")

# ── Summary ──────────────────────────────────────────────────────────────────

results = {
    "xgb_best_cv": round(xgb_best, 5),
    "xgb_best_params": xgb_study.best_params,
    "lgb_best_cv": round(lgb_best, 5),
    "lgb_best_params": lgb_study.best_params,
}

with open("optuna_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print("SUMMARY")
print(f"  XGBoost best CV:  {xgb_best:.5f}")
print(f"  LightGBM best CV: {lgb_best:.5f}")

# Pick winner and generate submission if CV > 0.805
best_model_name = "xgb" if xgb_best >= lgb_best else "lgb"
best_cv = max(xgb_best, lgb_best)
best_study = xgb_study if best_model_name == "xgb" else lgb_study

print(f"\n  Winner: {best_model_name.upper()} (CV={best_cv:.5f})")

if best_cv > 0.805:
    print("\n  CV > 0.805 — generating submission with best params...")
    test_preds = np.zeros(X_test.shape[0])
    bp = best_study.best_params

    for train_idx, val_idx in skf.split(X, y):
        if best_model_name == "xgb":
            model = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="error",
                tree_method="hist",
                verbosity=0,
                random_state=42,
                early_stopping_rounds=50,
                **bp,
            )
            model.fit(
                X[train_idx],
                y.values[train_idx],
                eval_set=[(X[val_idx], y.values[val_idx])],
                verbose=False,
            )
        else:
            model = lgb.LGBMClassifier(
                objective="binary",
                metric="binary_error",
                verbosity=-1,
                random_state=42,
                **bp,
            )
            model.fit(
                X[train_idx],
                y.values[train_idx],
                eval_set=[(X[val_idx], y.values[val_idx])],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(-1),
                ],
            )
        test_preds += model.predict_proba(X_test)[:, 1] / 5

    submission = pd.DataFrame(
        {"PassengerId": test["PassengerId"], "Transported": (test_preds > 0.5)}
    )
    submission.to_csv("submission_optuna.csv", index=False)
    print(f"  Submission saved: submission_optuna.csv ({submission.shape[0]} rows)")
    print(f"  {submission['Transported'].value_counts().to_dict()}")
else:
    print("\n  CV <= 0.805 — no submission generated.")

print(f"\nResults saved to optuna_results.json")
print("Done!")
