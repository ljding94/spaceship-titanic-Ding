"""
Spaceship Titanic — Refined Stacking v3
Goal: Beat LB 0.808 with better generalization.
Key improvements over stacking_model.py:
  1. Enhanced group-based imputation (Age, CryoSleep for children)
  2. DeckSide interaction feature (proven strong signal)
  3. Individual spend columns retained (but capped for stability)
  4. 5-seed averaging for ALL models including CatBoost
  5. HistGradientBoosting as 4th diverse base model
  6. Rank-based blending in addition to LR meta-learner
  7. Tighter regularization across all models
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("SPACESHIP TITANIC — Refined Stacking v3")
print("=" * 60)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(f"Train: {train.shape}, Test: {test.shape}")


def smart_impute(df):
    """Enhanced domain-aware imputation."""
    df = df.copy()

    # Extract group info
    df['GroupId'] = df['PassengerId'].str.split('_').str[0].astype(int)
    df['PersonNum'] = df['PassengerId'].str.split('_').str[1].astype(int)

    # Extract cabin parts
    df['Deck'] = df['Cabin'].str.split('/').str[0]
    df['CabinNum'] = df['Cabin'].str.split('/').str[1].astype(float)
    df['Side'] = df['Cabin'].str.split('/').str[2]

    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

    # Rule 1: CryoSleep passengers have 0 spending
    cryo_mask = df['CryoSleep'].isin([True, 'True'])
    for col in spend_cols:
        df.loc[cryo_mask & df[col].isna(), col] = 0.0

    # Rule 2: If all spend is 0 (and all known), likely CryoSleep
    total_spend = df[spend_cols].sum(axis=1)
    zero_spend = (total_spend == 0) & (~df[spend_cols].isna().any(axis=1))
    df.loc[zero_spend & df['CryoSleep'].isna(), 'CryoSleep'] = True

    # Rule 3: If any spend > 0, not CryoSleep
    has_spend = (total_spend > 0)
    df.loc[has_spend & df['CryoSleep'].isna(), 'CryoSleep'] = False

    # Rule 4: Children (Age < 13) never have RoomService or VRDeck in practice
    child_mask = df['Age'] < 13
    for col in ['RoomService', 'VRDeck']:
        df.loc[child_mask & df[col].isna(), col] = 0.0

    # Rule 5: Group members share HomePlanet, Destination, Deck, Side
    for col in ['HomePlanet', 'Destination', 'Deck', 'Side']:
        group_mode = df.groupby('GroupId')[col].transform(
            lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
        )
        df[col] = df[col].fillna(group_mode)

    # Rule 6: Group members have similar Age — fill with group median
    group_age = df.groupby('GroupId')['Age'].transform('median')
    df['Age'] = df['Age'].fillna(group_age)

    # Convert booleans
    df['CryoSleep'] = df['CryoSleep'].map(
        {True: 1, False: 0, 'True': 1, 'False': 0}
    )
    df['CryoSleep'] = df['CryoSleep'].fillna(df['CryoSleep'].mode()[0])
    df['VIP'] = df['VIP'].map({True: 1, False: 0, 'True': 1, 'False': 0})
    df['VIP'] = df['VIP'].fillna(0)

    # Fill remaining
    for col in spend_cols:
        df[col] = df[col].fillna(df[col].median())
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['HomePlanet'] = df['HomePlanet'].fillna(df['HomePlanet'].mode()[0])
    df['Destination'] = df['Destination'].fillna(df['Destination'].mode()[0])
    df['Deck'] = df['Deck'].fillna('Unknown')
    df['Side'] = df['Side'].fillna('Unknown')
    df['CabinNum'] = df['CabinNum'].fillna(df['CabinNum'].median())

    return df


def build_features(df):
    """Build features — moderate count, strong signal."""
    df = df.copy()
    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

    # Group features
    df['GroupSize'] = df.groupby('GroupId')['PassengerId'].transform('count')
    df['IsSolo'] = (df['GroupSize'] == 1).astype(int)

    # Spend features
    df['TotalSpend'] = df[spend_cols].sum(axis=1)
    df['HasSpend'] = (df['TotalSpend'] > 0).astype(int)
    df['LuxurySpend'] = df['Spa'] + df['VRDeck'] + df['RoomService']
    df['NecessitySpend'] = df['FoodCourt'] + df['ShoppingMall']
    df['NumSpendCats'] = (df[spend_cols] > 0).sum(axis=1)

    # Log spend (stabilizes variance)
    df['LogTotalSpend'] = np.log1p(df['TotalSpend'])

    # Age features
    df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 18, 25, 50, 100],
                          labels=[0, 1, 2, 3, 4]).astype(float)
    df['IsChild'] = (df['Age'] < 13).astype(int)

    # Family from Name
    df['LastName'] = df['Name'].str.split().str[-1]
    family_size = df.groupby('LastName')['PassengerId'].transform('count')
    df['FamilySize'] = family_size.fillna(1).clip(upper=10)

    # Cabin number buckets
    df['CabinNumBucket'] = pd.qcut(df['CabinNum'], q=10, labels=False,
                                    duplicates='drop')
    df['CabinNumBucket'] = df['CabinNumBucket'].fillna(-1)

    # DeckSide interaction (strong signal)
    df['DeckSide'] = df['Deck'].astype(str) + '_' + df['Side'].astype(str)

    # Age × CryoSleep interaction
    df['AgeCryo'] = df['Age'] * df['CryoSleep']

    # Label encode categoricals
    for col in ['HomePlanet', 'Destination', 'Deck', 'Side', 'DeckSide']:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])

    return df


# Process data
train['is_train'] = 1
test['is_train'] = 0
test['Transported'] = np.nan
combined = pd.concat([train, test], axis=0, ignore_index=True)
combined = smart_impute(combined)
combined = build_features(combined)

train_df = combined[combined['is_train'] == 1].copy()
test_df = combined[combined['is_train'] == 0].copy()

y = train_df['Transported'].astype(int)
drop_cols = ['PassengerId', 'Cabin', 'Name', 'Transported', 'is_train',
             'GroupId', 'LastName', 'PersonNum']
feature_cols = [c for c in train_df.columns if c not in drop_cols
                and c in test_df.columns]
print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")

X = train_df[feature_cols].values.astype(np.float32)
X_test = test_df[feature_cols].values.astype(np.float32)

# ── STAGE 1: Base model OOF predictions ──────────────────────────────────────
print("\n" + "=" * 60)
print("STAGE 1: Base model OOF predictions (5-seed)")
print("=" * 60)

N_SEEDS = 5
seeds = [42, 123, 456, 789, 2024]
N_FOLDS = 5

# Accumulators
lgb_oof_all = np.zeros(len(X))
xgb_oof_all = np.zeros(len(X))
cat_oof_all = np.zeros(len(X))
hgb_oof_all = np.zeros(len(X))

lgb_test_all = np.zeros(len(X_test))
xgb_test_all = np.zeros(len(X_test))
cat_test_all = np.zeros(len(X_test))
hgb_test_all = np.zeros(len(X_test))

for seed_idx, seed in enumerate(seeds):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    lgb_oof = np.zeros(len(X))
    xgb_oof = np.zeros(len(X))
    cat_oof = np.zeros(len(X))
    hgb_oof = np.zeros(len(X))
    lgb_test_s = np.zeros(len(X_test))
    xgb_test_s = np.zeros(len(X_test))
    cat_test_s = np.zeros(len(X_test))
    hgb_test_s = np.zeros(len(X_test))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y.values[tr_idx], y.values[va_idx]

        # LightGBM — tighter regularization
        lgb_m = lgb.LGBMClassifier(
            objective='binary', metric='binary_error',
            learning_rate=0.03, num_leaves=24, max_depth=5,
            min_child_samples=25, feature_fraction=0.75,
            bagging_fraction=0.75, bagging_freq=5,
            reg_alpha=0.5, reg_lambda=3.0,
            n_estimators=1000, verbose=-1, random_state=seed,
        )
        lgb_m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                             lgb.log_evaluation(0)])
        lgb_oof[va_idx] = lgb_m.predict_proba(X_va)[:, 1]
        lgb_test_s += lgb_m.predict_proba(X_test)[:, 1] / N_FOLDS

        # XGBoost — tighter regularization
        xgb_m = xgb.XGBClassifier(
            objective='binary:logistic', eval_metric='error',
            learning_rate=0.03, max_depth=4, min_child_weight=5,
            subsample=0.75, colsample_bytree=0.75,
            reg_alpha=0.3, reg_lambda=2.0, gamma=0.1,
            n_estimators=1000, early_stopping_rounds=50,
            verbosity=0, random_state=seed, tree_method='hist',
        )
        xgb_m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        xgb_oof[va_idx] = xgb_m.predict_proba(X_va)[:, 1]
        xgb_test_s += xgb_m.predict_proba(X_test)[:, 1] / N_FOLDS

        # CatBoost — 5-seed now
        cat_m = CatBoostClassifier(
            iterations=1000, learning_rate=0.03, depth=5,
            l2_leaf_reg=5.0, min_data_in_leaf=15,
            subsample=0.75, colsample_bylevel=0.75,
            eval_metric='Accuracy', random_seed=seed, verbose=0,
            early_stopping_rounds=50,
        )
        cat_m.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0)
        cat_oof[va_idx] = cat_m.predict_proba(X_va)[:, 1]
        cat_test_s += cat_m.predict_proba(X_test)[:, 1] / N_FOLDS

        # HistGradientBoosting — sklearn native, different algorithm
        hgb_m = HistGradientBoostingClassifier(
            learning_rate=0.03, max_iter=500, max_depth=5,
            min_samples_leaf=25, max_leaf_nodes=24,
            l2_regularization=3.0,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=50, random_state=seed,
        )
        hgb_m.fit(X_tr, y_tr)
        hgb_oof[va_idx] = hgb_m.predict_proba(X_va)[:, 1]
        hgb_test_s += hgb_m.predict_proba(X_test)[:, 1] / N_FOLDS

    lgb_cv = accuracy_score(y, (lgb_oof > 0.5).astype(int))
    xgb_cv = accuracy_score(y, (xgb_oof > 0.5).astype(int))
    cat_cv = accuracy_score(y, (cat_oof > 0.5).astype(int))
    hgb_cv = accuracy_score(y, (hgb_oof > 0.5).astype(int))
    print(f"  Seed {seed}: LGB={lgb_cv:.4f} XGB={xgb_cv:.4f} CAT={cat_cv:.4f} HGB={hgb_cv:.4f}")

    lgb_oof_all += lgb_oof / N_SEEDS
    xgb_oof_all += xgb_oof / N_SEEDS
    cat_oof_all += cat_oof / N_SEEDS
    hgb_oof_all += hgb_oof / N_SEEDS
    lgb_test_all += lgb_test_s / N_SEEDS
    xgb_test_all += xgb_test_s / N_SEEDS
    cat_test_all += cat_test_s / N_SEEDS
    hgb_test_all += hgb_test_s / N_SEEDS

# Print averaged base model CVs
for name, oof in [('LGB', lgb_oof_all), ('XGB', xgb_oof_all),
                   ('CAT', cat_oof_all), ('HGB', hgb_oof_all)]:
    cv = accuracy_score(y, (oof > 0.5).astype(int))
    print(f"  {name} (5-seed avg): CV={cv:.4f}")

# ── STAGE 2: Stacking meta-learner ──────────────────────────────────────────
print("\n" + "=" * 60)
print("STAGE 2: Stacking Meta-Learner")
print("=" * 60)

meta_train = np.column_stack([lgb_oof_all, xgb_oof_all, cat_oof_all, hgb_oof_all])
meta_test = np.column_stack([lgb_test_all, xgb_test_all, cat_test_all, hgb_test_all])

# Standardize for LR
meta_scaler = StandardScaler()
meta_train_scaled = meta_scaler.fit_transform(meta_train)
meta_test_scaled = meta_scaler.transform(meta_test)

# CV the meta-learner
skf_meta = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
stack_oof = np.zeros(len(y))
stack_test = np.zeros(len(X_test))

for fold, (tr_idx, va_idx) in enumerate(skf_meta.split(meta_train_scaled, y)):
    meta_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    meta_lr.fit(meta_train_scaled[tr_idx], y.values[tr_idx])
    stack_oof[va_idx] = meta_lr.predict_proba(meta_train_scaled[va_idx])[:, 1]
    stack_test += meta_lr.predict_proba(meta_test_scaled)[:, 1] / 5

stack_cv = accuracy_score(y, (stack_oof > 0.5).astype(int))
print(f"Stacking (LR) CV: {stack_cv:.4f}")

# Also try rank-based blending
def rank_blend(oofs, tests, weights):
    """Blend using percentile ranks — more robust to calibration differences."""
    from scipy.stats import rankdata
    blend_oof = np.zeros(len(oofs[0]))
    blend_test = np.zeros(len(tests[0]))
    for oof, tst, w in zip(oofs, tests, weights):
        blend_oof += w * rankdata(oof) / len(oof)
        blend_test += w * rankdata(tst) / len(tst)
    return blend_oof, blend_test

oofs = [lgb_oof_all, xgb_oof_all, cat_oof_all, hgb_oof_all]
tests = [lgb_test_all, xgb_test_all, cat_test_all, hgb_test_all]

# Grid search weights
best_rank_cv = 0
best_weights = None
for w_lgb in np.arange(0.20, 0.45, 0.05):
    for w_xgb in np.arange(0.15, 0.40, 0.05):
        for w_cat in np.arange(0.15, 0.40, 0.05):
            w_hgb = 1.0 - w_lgb - w_xgb - w_cat
            if w_hgb < 0.05 or w_hgb > 0.40:
                continue
            rb_oof, _ = rank_blend(oofs, tests, [w_lgb, w_xgb, w_cat, w_hgb])
            cv = accuracy_score(y, (rb_oof > 0.5).astype(int))
            if cv > best_rank_cv:
                best_rank_cv = cv
                best_weights = (w_lgb, w_xgb, w_cat, w_hgb)

print(f"Rank blend CV: {best_rank_cv:.4f} (weights: LGB={best_weights[0]:.2f} XGB={best_weights[1]:.2f} CAT={best_weights[2]:.2f} HGB={best_weights[3]:.2f})")
rank_oof, rank_test = rank_blend(oofs, tests, best_weights)

# Simple average
avg_oof = np.mean(meta_train, axis=1)
avg_test = np.mean(meta_test, axis=1)
avg_cv = accuracy_score(y, (avg_oof > 0.5).astype(int))
print(f"Simple avg CV: {avg_cv:.4f}")

# ── Summary and best selection ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

results = {
    'LGB (5-seed)': (accuracy_score(y, (lgb_oof_all > 0.5).astype(int)), lgb_test_all, lgb_oof_all),
    'XGB (5-seed)': (accuracy_score(y, (xgb_oof_all > 0.5).astype(int)), xgb_test_all, xgb_oof_all),
    'CAT (5-seed)': (accuracy_score(y, (cat_oof_all > 0.5).astype(int)), cat_test_all, cat_oof_all),
    'HGB (5-seed)': (accuracy_score(y, (hgb_oof_all > 0.5).astype(int)), hgb_test_all, hgb_oof_all),
    'Simple Average': (avg_cv, avg_test, avg_oof),
    'Stacking (LR)': (stack_cv, stack_test, stack_oof),
    'Rank Blend': (best_rank_cv, rank_test, rank_oof),
}

for name, (cv, _, _) in sorted(results.items(), key=lambda x: -x[1][0]):
    print(f"  {name:25s}: CV={cv:.4f}")

best_name = max(results, key=lambda k: results[k][0])
best_cv, best_preds, best_oof = results[best_name]
print(f"\nBest: {best_name} (CV={best_cv:.4f})")
print(f"Previous best: stacking_model.py CV=0.813, LB=0.808")

# Threshold sweep
print("\n--- Threshold Sweep ---")
best_thr, best_thr_acc = 0.5, best_cv
for thr in np.arange(0.46, 0.55, 0.005):
    acc = accuracy_score(y, (best_oof > thr).astype(int))
    if acc > best_thr_acc:
        best_thr, best_thr_acc = thr, acc

if best_thr != 0.5:
    print(f"Optimal threshold: {best_thr:.3f} (CV={best_thr_acc:.4f} vs 0.5={best_cv:.4f})")
else:
    print(f"Threshold 0.5 is optimal (CV={best_cv:.4f})")

# Save submissions — both thresholds
# Version 1: threshold 0.5 (safer, less prone to overfitting)
sub_05 = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Transported': (best_preds > 0.5)
})
sub_05.to_csv('submission_refined_05.csv', index=False)
print(f"\nsubmission_refined_05.csv (thr=0.5, CV={best_cv:.4f}):")
print(sub_05['Transported'].value_counts())

# Version 2: tuned threshold
sub_thr = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Transported': (best_preds > best_thr)
})
sub_thr.to_csv('submission_refined_thr.csv', index=False)
print(f"\nsubmission_refined_thr.csv (thr={best_thr:.3f}, CV={best_thr_acc:.4f}):")
print(sub_thr['Transported'].value_counts())

# Use 0.5 threshold as main submission (safer for LB)
sub_05.to_csv('submission.csv', index=False)
print(f"\nSaved submission.csv with thr=0.5 (CV={best_cv:.4f})")

print("\nDone!")
