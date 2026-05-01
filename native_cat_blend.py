"""
Spaceship Titanic — Native CatBoost + Weighted Blend v4
Goal: Beat LB 0.80944.
Key insight: CatBoost with native categorical handling is superior to label encoding.
Strategy:
  1. CatBoost with native categoricals (no encoding)
  2. LGB + XGB with label encoding (as before)
  3. Simple weighted averaging (no stacking meta-learner to reduce overfit)
  4. Pseudo-labeling round 2 (proven to help)
  5. StratifiedKFold (proven better than GroupKFold on LB)
  6. 5 seeds for stability
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("SPACESHIP TITANIC — Native CatBoost + Weighted Blend v4")
print("=" * 60)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(f"Train: {train.shape}, Test: {test.shape}")

CAT_FEATURES = ['HomePlanet', 'Destination', 'Deck', 'Side', 'DeckSide']


def smart_impute(df):
    df = df.copy()
    df['GroupId'] = df['PassengerId'].str.split('_').str[0].astype(int)
    df['Deck'] = df['Cabin'].str.split('/').str[0]
    df['CabinNum'] = df['Cabin'].str.split('/').str[1].astype(float)
    df['Side'] = df['Cabin'].str.split('/').str[2]

    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    cryo_mask = df['CryoSleep'].isin([True, 'True'])
    for col in spend_cols:
        df.loc[cryo_mask & df[col].isna(), col] = 0.0

    total_spend = df[spend_cols].sum(axis=1)
    zero_spend = (total_spend == 0) & (~df[spend_cols].isna().any(axis=1))
    df.loc[zero_spend & df['CryoSleep'].isna(), 'CryoSleep'] = True
    has_spend = (total_spend > 0)
    df.loc[has_spend & df['CryoSleep'].isna(), 'CryoSleep'] = False

    for col in ['HomePlanet', 'Destination', 'Deck', 'Side']:
        group_mode = df.groupby('GroupId')[col].transform(
            lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
        )
        df[col] = df[col].fillna(group_mode)

    group_age = df.groupby('GroupId')['Age'].transform('median')
    df['Age'] = df['Age'].fillna(group_age)

    df['CryoSleep'] = df['CryoSleep'].map({True: 1, False: 0, 'True': 1, 'False': 0})
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
    df = df.copy()
    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

    # Core group features
    df['GroupSize'] = df.groupby('GroupId')['PassengerId'].transform('count')
    df['IsSolo'] = (df['GroupSize'] == 1).astype(int)

    # Core spending features
    df['TotalSpend'] = df[spend_cols].sum(axis=1)
    df['HasSpend'] = (df['TotalSpend'] > 0).astype(int)
    df['TotalSpend_log'] = np.log1p(df['TotalSpend'])
    df['SpendPerPerson'] = df['TotalSpend'] / df['GroupSize']

    # Age features
    df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 18, 25, 50, 100],
                          labels=[0, 1, 2, 3, 4]).astype(float)
    df['IsChild'] = (df['Age'] < 13).astype(int)

    # Family size from name
    df['LastName'] = df['Name'].str.split().str[-1]
    family_size = df.groupby('LastName')['PassengerId'].transform('count')
    df['FamilySize'] = family_size.fillna(1).clip(upper=10)

    # Cabin features
    df['CabinNumBucket'] = pd.qcut(df['CabinNum'], q=10, labels=False, duplicates='drop')
    df['CabinNumBucket'] = df['CabinNumBucket'].fillna(-1)

    # Key interaction: DeckSide
    df['DeckSide'] = df['Deck'].astype(str) + '_' + df['Side'].astype(str)

    # CryoSleep interaction
    df['CryoAndSolo'] = df['CryoSleep'] * df['IsSolo']

    return df


# Process
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

# For CatBoost: keep categoricals as strings
cat_feature_idx = [feature_cols.index(c) for c in CAT_FEATURES if c in feature_cols]
print(f"CatBoost categorical indices: {cat_feature_idx}")

# For LGB/XGB: label encode categoricals
train_enc = train_df.copy()
test_enc = test_df.copy()
for col in CAT_FEATURES:
    if col in feature_cols:
        le = LabelEncoder()
        le.fit(combined[col].astype(str))
        train_enc[col] = le.transform(train_enc[col].astype(str))
        test_enc[col] = le.transform(test_enc[col].astype(str))

X_enc = train_enc[feature_cols].values.astype(np.float32)
X_test_enc = test_enc[feature_cols].values.astype(np.float32)

# For CatBoost: use DataFrames with string categoricals
X_cat = train_df[feature_cols].copy()
X_test_cat = test_df[feature_cols].copy()
for col in CAT_FEATURES:
    if col in feature_cols:
        X_cat[col] = X_cat[col].astype(str)
        X_test_cat[col] = X_test_cat[col].astype(str)
for col in feature_cols:
    if col not in CAT_FEATURES:
        X_cat[col] = X_cat[col].astype(np.float32)
        X_test_cat[col] = X_test_cat[col].astype(np.float32)


def run_models(X_lgb, X_xgb, X_catdf, X_test_lgb, X_test_xgb, X_test_catdf,
               y_train, seeds, n_folds=5, label=""):
    """Run LGB, XGB, CatBoost (native cats) and return OOF + test predictions."""
    N_SEEDS = len(seeds)
    n_train = len(y_train)
    n_test = len(X_test_lgb)

    lgb_oof = np.zeros(n_train)
    xgb_oof = np.zeros(n_train)
    cat_oof = np.zeros(n_train)
    lgb_test = np.zeros(n_test)
    xgb_test = np.zeros(n_test)
    cat_test = np.zeros(n_test)

    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        _lgb_oof = np.zeros(n_train)
        _xgb_oof = np.zeros(n_train)
        _cat_oof = np.zeros(n_train)
        _lgb_test = np.zeros(n_test)
        _xgb_test = np.zeros(n_test)
        _cat_test = np.zeros(n_test)

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_lgb, y_train)):
            y_tr = y_train[tr_idx]
            y_va = y_train[va_idx]

            # LightGBM
            lgb_m = lgb.LGBMClassifier(
                objective='binary', metric='binary_error',
                learning_rate=0.03, num_leaves=31, max_depth=5,
                min_child_samples=25, feature_fraction=0.7,
                bagging_fraction=0.8, bagging_freq=5,
                reg_alpha=0.5, reg_lambda=3.0,
                n_estimators=800, verbose=-1, random_state=seed,
            )
            lgb_m.fit(X_lgb[tr_idx], y_tr, eval_set=[(X_lgb[va_idx], y_va)],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(0)])
            _lgb_oof[va_idx] = lgb_m.predict_proba(X_lgb[va_idx])[:, 1]
            _lgb_test += lgb_m.predict_proba(X_test_lgb)[:, 1] / n_folds

            # XGBoost
            xgb_m = xgb.XGBClassifier(
                objective='binary:logistic', eval_metric='error',
                learning_rate=0.05, max_depth=5, min_child_weight=5,
                subsample=0.8, colsample_bytree=0.7,
                reg_alpha=0.2, reg_lambda=2.0,
                n_estimators=500, early_stopping_rounds=50,
                verbosity=0, random_state=seed, tree_method='hist',
            )
            xgb_m.fit(X_xgb[tr_idx], y_tr, eval_set=[(X_xgb[va_idx], y_va)],
                      verbose=False)
            _xgb_oof[va_idx] = xgb_m.predict_proba(X_xgb[va_idx])[:, 1]
            _xgb_test += xgb_m.predict_proba(X_test_xgb)[:, 1] / n_folds

            # CatBoost with native categoricals
            cat_train_pool = Pool(
                X_catdf.iloc[tr_idx], y_tr,
                cat_features=[c for c in CAT_FEATURES if c in feature_cols]
            )
            cat_val_pool = Pool(
                X_catdf.iloc[va_idx], y_va,
                cat_features=[c for c in CAT_FEATURES if c in feature_cols]
            )
            cat_test_pool = Pool(
                X_test_catdf,
                cat_features=[c for c in CAT_FEATURES if c in feature_cols]
            )

            cat_m = CatBoostClassifier(
                iterations=800, learning_rate=0.05, depth=6,
                l2_leaf_reg=5.0, min_data_in_leaf=15,
                subsample=0.8, colsample_bylevel=0.7,
                one_hot_max_size=10,
                eval_metric='Accuracy', random_seed=seed, verbose=0,
                early_stopping_rounds=50,
            )
            cat_m.fit(cat_train_pool, eval_set=cat_val_pool, verbose=0)
            _cat_oof[va_idx] = cat_m.predict_proba(cat_val_pool)[:, 1]
            _cat_test += cat_m.predict_proba(cat_test_pool)[:, 1] / n_folds

        lgb_oof += _lgb_oof / N_SEEDS
        xgb_oof += _xgb_oof / N_SEEDS
        cat_oof += _cat_oof / N_SEEDS
        lgb_test += _lgb_test / N_SEEDS
        xgb_test += _xgb_test / N_SEEDS
        cat_test += _cat_test / N_SEEDS

        print(f"  {label} Seed {seed}: LGB={accuracy_score(y_train, (_lgb_oof > 0.5).astype(int)):.4f} "
              f"XGB={accuracy_score(y_train, (_xgb_oof > 0.5).astype(int)):.4f} "
              f"CAT={accuracy_score(y_train, (_cat_oof > 0.5).astype(int)):.4f}")

    return lgb_oof, xgb_oof, cat_oof, lgb_test, xgb_test, cat_test


# ── Round 1 ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ROUND 1: Original training data")
print("=" * 60)

seeds_r1 = [42, 123, 456]
lgb_oof_r1, xgb_oof_r1, cat_oof_r1, lgb_test_r1, xgb_test_r1, cat_test_r1 = \
    run_models(X_enc, X_enc, X_cat, X_test_enc, X_test_enc, X_test_cat,
               y.values, seeds_r1, label="R1")

# Weighted average (try different weights)
print("\nR1 individual OOF:")
print(f"  LGB: {accuracy_score(y, (lgb_oof_r1 > 0.5).astype(int)):.4f}")
print(f"  XGB: {accuracy_score(y, (xgb_oof_r1 > 0.5).astype(int)):.4f}")
print(f"  CAT: {accuracy_score(y, (cat_oof_r1 > 0.5).astype(int)):.4f}")

# Find optimal weights via grid search
best_cv_r1 = 0
best_weights = (1/3, 1/3, 1/3)
for w_lgb in np.arange(0.1, 0.8, 0.05):
    for w_xgb in np.arange(0.1, 0.8 - w_lgb, 0.05):
        w_cat = 1 - w_lgb - w_xgb
        if w_cat < 0.1:
            continue
        blend_oof = w_lgb * lgb_oof_r1 + w_xgb * xgb_oof_r1 + w_cat * cat_oof_r1
        cv = accuracy_score(y, (blend_oof > 0.5).astype(int))
        if cv > best_cv_r1:
            best_cv_r1 = cv
            best_weights = (w_lgb, w_xgb, w_cat)

w_lgb, w_xgb, w_cat = best_weights
print(f"\nBest weights: LGB={w_lgb:.2f}, XGB={w_xgb:.2f}, CAT={w_cat:.2f}")
print(f"Weighted avg CV: {best_cv_r1:.4f}")

blend_test_r1 = w_lgb * lgb_test_r1 + w_xgb * xgb_test_r1 + w_cat * cat_test_r1
blend_oof_r1 = w_lgb * lgb_oof_r1 + w_xgb * xgb_oof_r1 + w_cat * cat_oof_r1

# ── Pseudo-labeling ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PSEUDO-LABELING")
print("=" * 60)

CONFIDENCE_THR = 0.90
confident_mask = (blend_test_r1 > CONFIDENCE_THR) | (blend_test_r1 < (1 - CONFIDENCE_THR))
pseudo_labels = (blend_test_r1 > 0.5).astype(int)
n_pseudo = confident_mask.sum()
print(f"Confident predictions: {n_pseudo}/{len(X_test_enc)} ({100*n_pseudo/len(X_test_enc):.1f}%)")

# Augment training data
X_enc_aug = np.vstack([X_enc, X_test_enc[confident_mask]])
y_aug = np.concatenate([y.values, pseudo_labels[confident_mask]])
X_cat_aug = pd.concat([X_cat, X_test_cat[confident_mask]], ignore_index=True)
print(f"Augmented: {len(X_enc_aug)} ({len(X_enc)} + {n_pseudo})")

# ── Round 2 ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ROUND 2: Augmented training data")
print("=" * 60)

seeds_r2 = [42, 123, 456, 789, 2024]
N_SEEDS = len(seeds_r2)
N_FOLDS = 5

lgb_oof_r2 = np.zeros(len(X_enc))
xgb_oof_r2 = np.zeros(len(X_enc))
cat_oof_r2 = np.zeros(len(X_enc))
lgb_test_r2 = np.zeros(len(X_test_enc))
xgb_test_r2 = np.zeros(len(X_test_enc))
cat_test_r2 = np.zeros(len(X_test_enc))

for seed in seeds_r2:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    _lgb_oof = np.zeros(len(X_enc))
    _xgb_oof = np.zeros(len(X_enc))
    _cat_oof = np.zeros(len(X_enc))
    _lgb_test = np.zeros(len(X_test_enc))
    _xgb_test = np.zeros(len(X_test_enc))
    _cat_test = np.zeros(len(X_test_enc))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_enc, y)):
        n_orig = len(X_enc)
        aug_indices = np.arange(n_orig, len(X_enc_aug))
        train_indices = np.concatenate([tr_idx, aug_indices])

        X_tr_enc = X_enc_aug[train_indices]
        y_tr = y_aug[train_indices]
        X_va_enc = X_enc[va_idx]
        y_va = y.values[va_idx]

        # CatBoost with native categoricals
        X_tr_cat = X_cat_aug.iloc[train_indices]
        X_va_cat = X_cat.iloc[va_idx]

        cat_cols = [c for c in CAT_FEATURES if c in feature_cols]
        cat_train_pool = Pool(X_tr_cat, y_tr, cat_features=cat_cols)
        cat_val_pool = Pool(X_va_cat, y_va, cat_features=cat_cols)
        cat_test_pool = Pool(X_test_cat, cat_features=cat_cols)

        # LightGBM
        lgb_m = lgb.LGBMClassifier(
            objective='binary', metric='binary_error',
            learning_rate=0.03, num_leaves=31, max_depth=5,
            min_child_samples=25, feature_fraction=0.7,
            bagging_fraction=0.8, bagging_freq=5,
            reg_alpha=0.5, reg_lambda=3.0,
            n_estimators=800, verbose=-1, random_state=seed,
        )
        lgb_m.fit(X_tr_enc, y_tr, eval_set=[(X_va_enc, y_va)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                             lgb.log_evaluation(0)])
        _lgb_oof[va_idx] = lgb_m.predict_proba(X_va_enc)[:, 1]
        _lgb_test += lgb_m.predict_proba(X_test_enc)[:, 1] / N_FOLDS

        # XGBoost
        xgb_m = xgb.XGBClassifier(
            objective='binary:logistic', eval_metric='error',
            learning_rate=0.05, max_depth=5, min_child_weight=5,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.2, reg_lambda=2.0,
            n_estimators=500, early_stopping_rounds=50,
            verbosity=0, random_state=seed, tree_method='hist',
        )
        xgb_m.fit(X_tr_enc, y_tr, eval_set=[(X_va_enc, y_va)], verbose=False)
        _xgb_oof[va_idx] = xgb_m.predict_proba(X_va_enc)[:, 1]
        _xgb_test += xgb_m.predict_proba(X_test_enc)[:, 1] / N_FOLDS

        # CatBoost (native cats)
        cat_m = CatBoostClassifier(
            iterations=800, learning_rate=0.05, depth=6,
            l2_leaf_reg=5.0, min_data_in_leaf=15,
            subsample=0.8, colsample_bylevel=0.7,
            one_hot_max_size=10,
            eval_metric='Accuracy', random_seed=seed, verbose=0,
            early_stopping_rounds=50,
        )
        cat_m.fit(cat_train_pool, eval_set=cat_val_pool, verbose=0)
        _cat_oof[va_idx] = cat_m.predict_proba(cat_val_pool)[:, 1]
        _cat_test += cat_m.predict_proba(cat_test_pool)[:, 1] / N_FOLDS

    lgb_oof_r2 += _lgb_oof / N_SEEDS
    xgb_oof_r2 += _xgb_oof / N_SEEDS
    cat_oof_r2 += _cat_oof / N_SEEDS
    lgb_test_r2 += _lgb_test / N_SEEDS
    xgb_test_r2 += _xgb_test / N_SEEDS
    cat_test_r2 += _cat_test / N_SEEDS

    print(f"  R2 Seed {seed}: LGB={accuracy_score(y, (_lgb_oof > 0.5).astype(int)):.4f} "
          f"XGB={accuracy_score(y, (_xgb_oof > 0.5).astype(int)):.4f} "
          f"CAT={accuracy_score(y, (_cat_oof > 0.5).astype(int)):.4f}")

# Weighted average R2
blend_oof_r2 = w_lgb * lgb_oof_r2 + w_xgb * xgb_oof_r2 + w_cat * cat_oof_r2
blend_test_r2 = w_lgb * lgb_test_r2 + w_xgb * xgb_test_r2 + w_cat * cat_test_r2
cv_r2 = accuracy_score(y, (blend_oof_r2 > 0.5).astype(int))
print(f"\nR2 Weighted avg CV: {cv_r2:.4f}")

# ── Blend R1 + R2 ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BLEND R1 + R2")
print("=" * 60)

best_blend_cv = 0
best_alpha = 0.5
for alpha in np.arange(0.0, 1.05, 0.05):
    blend_oof = alpha * blend_oof_r1 + (1 - alpha) * blend_oof_r2
    cv = accuracy_score(y, (blend_oof > 0.5).astype(int))
    if cv > best_blend_cv:
        best_blend_cv = cv
        best_alpha = alpha

print(f"Best alpha: {best_alpha:.2f} (R1 weight), CV={best_blend_cv:.4f}")

final_oof = best_alpha * blend_oof_r1 + (1 - best_alpha) * blend_oof_r2
final_test = best_alpha * blend_test_r1 + (1 - best_alpha) * blend_test_r2

# Threshold sweep
best_thr, best_thr_acc = 0.5, best_blend_cv
for thr in np.arange(0.46, 0.55, 0.005):
    acc = accuracy_score(y, (final_oof > thr).astype(int))
    if acc > best_thr_acc:
        best_thr, best_thr_acc = thr, acc

print(f"Threshold: {best_thr:.3f} (CV={best_thr_acc:.4f})")

# ── Save ─────────────────────────────────────────────────────────────────────
sub = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Transported': (final_test > best_thr)
})
sub.to_csv('submission_native_cat_v4.csv', index=False)
print(f"\nsubmission_native_cat_v4.csv (thr={best_thr:.3f}):")
print(sub['Transported'].value_counts())

sub_05 = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Transported': (final_test > 0.5)
})
sub_05.to_csv('submission_native_cat_v4_05.csv', index=False)
print(f"\nsubmission_native_cat_v4_05.csv (thr=0.5):")
print(sub_05['Transported'].value_counts())

sub.to_csv('submission.csv', index=False)
print(f"\nSaved submission.csv (NativeCat v4, CV={best_thr_acc:.4f})")
print("\nDone!")
