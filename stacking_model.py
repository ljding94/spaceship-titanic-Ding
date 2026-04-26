"""
Spaceship Titanic — Stacking Ensemble with Smart Imputation
Goal: Beat LB 0.799 by reducing overfitting.
Key ideas:
  1. Domain-aware imputation (CryoSleep→0 spend, group-based fill, name-based families)
  2. Stacking: LGB + XGB + CatBoost → LogisticRegression meta-learner
  3. Keep features moderate — no ratio/log bloat
  4. Multi-seed for stability
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("SPACESHIP TITANIC — Stacking Ensemble")
print("=" * 60)

# Load
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(f"Train: {train.shape}, Test: {test.shape}")

# ── Smart Domain-Aware Imputation ──────────────────────────────────────────
def smart_impute(df):
    """Fill missing values using domain knowledge before feature engineering."""
    df = df.copy()

    # Extract group info first (needed for group-based imputation)
    df['GroupId'] = df['PassengerId'].str.split('_').str[0].astype(int)

    # Extract cabin parts
    df['Deck'] = df['Cabin'].str.split('/').str[0]
    df['CabinNum'] = df['Cabin'].str.split('/').str[1].astype(float)
    df['Side'] = df['Cabin'].str.split('/').str[2]

    # Rule 1: CryoSleep passengers have 0 spending
    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    cryo_mask = df['CryoSleep'].isin([True, 'True'])
    for col in spend_cols:
        df.loc[cryo_mask & df[col].isna(), col] = 0.0

    # Rule 2: If all spend is 0, likely CryoSleep
    total_spend = df[spend_cols].sum(axis=1)
    zero_spend = (total_spend == 0) & (~df[spend_cols].isna().any(axis=1))
    df.loc[zero_spend & df['CryoSleep'].isna(), 'CryoSleep'] = True

    # Rule 3: If any spend > 0, not CryoSleep
    has_spend = (total_spend > 0)
    df.loc[has_spend & df['CryoSleep'].isna(), 'CryoSleep'] = False

    # Rule 4: Group members share HomePlanet and Destination
    for col in ['HomePlanet', 'Destination']:
        group_mode = df.groupby('GroupId')[col].transform(
            lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
        )
        df[col] = df[col].fillna(group_mode)

    # Rule 5: Group members share Deck and Side
    for col in ['Deck', 'Side']:
        group_mode = df.groupby('GroupId')[col].transform(
            lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
        )
        df[col] = df[col].fillna(group_mode)

    # Fill remaining with mode/median
    df['CryoSleep'] = df['CryoSleep'].map(
        {True: 1, False: 0, 'True': 1, 'False': 0}
    )
    df['CryoSleep'] = df['CryoSleep'].fillna(df['CryoSleep'].mode()[0])
    df['VIP'] = df['VIP'].map({True: 1, False: 0, 'True': 1, 'False': 0})
    df['VIP'] = df['VIP'].fillna(0)

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
    """Build features after imputation."""
    df = df.copy()

    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

    # Group features
    df['GroupSize'] = df.groupby('GroupId')['PassengerId'].transform('count')
    df['IsSolo'] = (df['GroupSize'] == 1).astype(int)

    # Spend features (keep it simple)
    df['TotalSpend'] = df[spend_cols].sum(axis=1)
    df['HasSpend'] = (df['TotalSpend'] > 0).astype(int)

    # Age bins
    df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 18, 25, 50, 100],
                          labels=[0, 1, 2, 3, 4]).astype(float)
    df['IsChild'] = (df['Age'] < 13).astype(int)

    # Family from Name (last name = family)
    df['LastName'] = df['Name'].str.split().str[-1]
    family_size = df.groupby('LastName')['PassengerId'].transform('count')
    df['FamilySize'] = family_size.fillna(1)
    df['FamilySize'] = df['FamilySize'].clip(upper=10)  # cap outliers

    # Cabin number buckets
    df['CabinNumBucket'] = pd.qcut(df['CabinNum'], q=10, labels=False,
                                    duplicates='drop')
    df['CabinNumBucket'] = df['CabinNumBucket'].fillna(-1)

    # Label encode categoricals
    for col in ['HomePlanet', 'Destination', 'Deck', 'Side']:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])

    return df


# ── Process data ────────────────────────────────────────────────────────────
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
# Remove PersonNum if it exists
feature_cols = [c for c in train_df.columns if c not in drop_cols
                and c in test_df.columns]
print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")

X = train_df[feature_cols].values.astype(np.float32)
X_test = test_df[feature_cols].values.astype(np.float32)

# ── Stacking: Generate OOF predictions from base models ────────────────────
print("\n" + "=" * 60)
print("STAGE 1: Base model OOF predictions")
print("=" * 60)

N_SEEDS = 3
seeds = [42, 123, 456]

# Accumulators for multi-seed averaging
lgb_oof_all = np.zeros(len(X))
xgb_oof_all = np.zeros(len(X))
lgb_test_all = np.zeros(len(X_test))
xgb_test_all = np.zeros(len(X_test))

for seed in seeds:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    lgb_oof = np.zeros(len(X))
    xgb_oof = np.zeros(len(X))
    lgb_test = np.zeros(len(X_test))
    xgb_test = np.zeros(len(X_test))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y.values[tr_idx], y.values[va_idx]

        # LightGBM
        lgb_m = lgb.LGBMClassifier(
            objective='binary', metric='binary_error',
            learning_rate=0.03, num_leaves=31, max_depth=5,
            min_child_samples=20, feature_fraction=0.8,
            bagging_fraction=0.8, bagging_freq=5,
            reg_alpha=0.3, reg_lambda=2.0,
            n_estimators=800, verbose=-1, random_state=seed,
        )
        lgb_m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                             lgb.log_evaluation(0)])
        lgb_oof[va_idx] = lgb_m.predict_proba(X_va)[:, 1]
        lgb_test += lgb_m.predict_proba(X_test)[:, 1] / 5

        # XGBoost
        xgb_m = xgb.XGBClassifier(
            objective='binary:logistic', eval_metric='error',
            learning_rate=0.05, max_depth=5, min_child_weight=3,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            n_estimators=500, early_stopping_rounds=50,
            verbosity=0, random_state=seed, tree_method='hist',
        )
        xgb_m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        xgb_oof[va_idx] = xgb_m.predict_proba(X_va)[:, 1]
        xgb_test += xgb_m.predict_proba(X_test)[:, 1] / 5

    lgb_cv = accuracy_score(y, (lgb_oof > 0.5).astype(int))
    xgb_cv = accuracy_score(y, (xgb_oof > 0.5).astype(int))
    print(f"  Seed {seed}: LGB={lgb_cv:.4f}, XGB={xgb_cv:.4f}")

    lgb_oof_all += lgb_oof / N_SEEDS
    xgb_oof_all += xgb_oof / N_SEEDS
    lgb_test_all += lgb_test / N_SEEDS
    xgb_test_all += xgb_test / N_SEEDS

# CatBoost (single seed — slow)
print("\n  CatBoost...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cat_oof = np.zeros(len(X))
cat_test = np.zeros(len(X_test))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y.values[tr_idx], y.values[va_idx]

    cat_m = CatBoostClassifier(
        iterations=800, learning_rate=0.05, depth=6,
        l2_leaf_reg=3.0, min_data_in_leaf=10,
        subsample=0.8, colsample_bylevel=0.8,
        eval_metric='Accuracy', random_seed=42, verbose=0,
        early_stopping_rounds=50,
    )
    cat_m.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0)
    cat_oof[va_idx] = cat_m.predict_proba(X_va)[:, 1]
    cat_test += cat_m.predict_proba(X_test)[:, 1] / 5
    acc = accuracy_score(y_va, (cat_oof[va_idx] > 0.5).astype(int))
    print(f"    Fold {fold+1}: {acc:.4f}")

cat_cv = accuracy_score(y, (cat_oof > 0.5).astype(int))
print(f"  CatBoost CV: {cat_cv:.4f}")

# ── Stage 2: Stacking meta-learner ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STAGE 2: Stacking Meta-Learner")
print("=" * 60)

# Stack OOF predictions as features for meta-learner
meta_train = np.column_stack([lgb_oof_all, xgb_oof_all, cat_oof])
meta_test = np.column_stack([lgb_test_all, xgb_test_all, cat_test])

# Logistic Regression meta-learner (simple → less overfitting)
meta_scaler = StandardScaler()
meta_train_scaled = meta_scaler.fit_transform(meta_train)
meta_test_scaled = meta_scaler.transform(meta_test)

# CV the meta-learner too
skf_meta = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
stack_oof = np.zeros(len(y))
stack_test = np.zeros(len(X_test))
stack_scores = []

for fold, (tr_idx, va_idx) in enumerate(skf_meta.split(meta_train_scaled, y)):
    m_tr = meta_train_scaled[tr_idx]
    m_va = meta_train_scaled[va_idx]
    y_tr, y_va = y.values[tr_idx], y.values[va_idx]

    meta_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    meta_lr.fit(m_tr, y_tr)
    stack_oof[va_idx] = meta_lr.predict_proba(m_va)[:, 1]
    stack_test += meta_lr.predict_proba(meta_test_scaled)[:, 1] / 5
    acc = accuracy_score(y_va, (stack_oof[va_idx] > 0.5).astype(int))
    stack_scores.append(acc)

stack_cv = accuracy_score(y, (stack_oof > 0.5).astype(int))
print(f"Stacking CV: {stack_cv:.4f} (per-fold: {[f'{s:.4f}' for s in stack_scores]})")

# ── Also try simple averaging for comparison ───────────────────────────────
avg_oof = (lgb_oof_all + xgb_oof_all + cat_oof) / 3
avg_test = (lgb_test_all + xgb_test_all + cat_test) / 3
avg_cv = accuracy_score(y, (avg_oof > 0.5).astype(int))

# Weighted average (favor the better models)
w_lgb, w_xgb, w_cat = 0.35, 0.35, 0.30
wavg_oof = w_lgb * lgb_oof_all + w_xgb * xgb_oof_all + w_cat * cat_oof
wavg_test = w_lgb * lgb_test_all + w_xgb * xgb_test_all + w_cat * cat_test
wavg_cv = accuracy_score(y, (wavg_oof > 0.5).astype(int))

# ── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

lgb_cv_final = accuracy_score(y, (lgb_oof_all > 0.5).astype(int))
xgb_cv_final = accuracy_score(y, (xgb_oof_all > 0.5).astype(int))

results = {
    'LGB (multi-seed)': (lgb_cv_final, lgb_test_all),
    'XGB (multi-seed)': (xgb_cv_final, xgb_test_all),
    'CatBoost': (cat_cv, cat_test),
    'Simple Average': (avg_cv, avg_test),
    'Weighted Average': (wavg_cv, wavg_test),
    'Stacking (LR)': (stack_cv, stack_test),
}

for name, (cv, _) in sorted(results.items(), key=lambda x: -x[1][0]):
    print(f"  {name:25s}: CV={cv:.4f}")

# Pick best
best_name = max(results, key=lambda k: results[k][0])
best_cv = results[best_name][0]
best_preds = results[best_name][1]
print(f"\nBest: {best_name} (CV={best_cv:.4f})")
print(f"Previous best: baseline_xgb.py CV=0.802, LB=0.799")

# Threshold sweep on best
print("\n--- Threshold Sweep ---")
best_thr, best_thr_acc = 0.5, best_cv
if best_name == 'Stacking (LR)':
    sweep_oof = stack_oof
else:
    sweep_oof = results[best_name][1]  # doesn't apply for OOF
    # Use the OOF corresponding to best method
    if 'Average' in best_name or 'Stacking' in best_name:
        if best_name == 'Simple Average':
            sweep_oof = avg_oof
        elif best_name == 'Weighted Average':
            sweep_oof = wavg_oof
        else:
            sweep_oof = stack_oof
    elif 'LGB' in best_name:
        sweep_oof = lgb_oof_all
    elif 'XGB' in best_name:
        sweep_oof = xgb_oof_all
    elif 'Cat' in best_name:
        sweep_oof = cat_oof

for thr in np.arange(0.46, 0.55, 0.005):
    acc = accuracy_score(y, (sweep_oof > thr).astype(int))
    if acc > best_thr_acc:
        best_thr, best_thr_acc = thr, acc

if best_thr != 0.5:
    print(f"Optimal threshold: {best_thr:.3f} (CV={best_thr_acc:.4f} vs 0.5 threshold CV={best_cv:.4f})")
else:
    print(f"Threshold 0.5 is already optimal (CV={best_cv:.4f})")

# ── Save submissions ────────────────────────────────────────────────────────
# Main submission: best model at 0.5 threshold
sub = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Transported': (best_preds > 0.5)
})
sub.to_csv('submission_stacking.csv', index=False)
print(f"\nsubmission_stacking.csv saved ({best_name}, thr=0.5)")
print(sub['Transported'].value_counts())

# If threshold helps, save that too
if best_thr != 0.5:
    sub_thr = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Transported': (best_preds > best_thr)
    })
    sub_thr.to_csv('submission_stacking_thr.csv', index=False)
    print(f"\nsubmission_stacking_thr.csv saved (thr={best_thr:.3f})")

# If stacking beats previous best CV (0.802), also save as main submission
if best_cv > 0.802:
    sub.to_csv('submission.csv', index=False)
    print(f"\n** IMPROVED! Saved as submission.csv (CV {best_cv:.4f} > 0.802) **")
else:
    print(f"\nCV {best_cv:.4f} did not beat 0.802 — kept existing submission.csv")

print("\nDone!")
