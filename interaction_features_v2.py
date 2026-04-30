"""
Spaceship Titanic — Enhanced Feature Engineering
Goal: Beat LB 0.80897 with better feature engineering.
Key improvements:
  1. Interaction features (planet x cryo, deck x side, spending ratios)
  2. Frequency encoding for categorical variables
  3. Spending per person in group
  4. Cabin position features
  5. Keep same pseudo-label stacking approach from best model
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
print("SPACESHIP TITANIC — Enhanced Feature Engineering")
print("=" * 60)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(f"Train: {train.shape}, Test: {test.shape}")


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
    """Enhanced feature engineering with interactions and frequency encoding."""
    df = df.copy()
    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

    # Basic engineered features
    df['GroupSize'] = df.groupby('GroupId')['PassengerId'].transform('count')
    df['IsSolo'] = (df['GroupSize'] == 1).astype(int)
    df['TotalSpend'] = df[spend_cols].sum(axis=1)
    df['HasSpend'] = (df['TotalSpend'] > 0).astype(int)
    df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 18, 25, 50, 100],
                          labels=[0, 1, 2, 3, 4]).astype(float)
    df['IsChild'] = (df['Age'] < 13).astype(int)
    df['LastName'] = df['Name'].str.split().str[-1]
    family_size = df.groupby('LastName')['PassengerId'].transform('count')
    df['FamilySize'] = family_size.fillna(1).clip(upper=10)
    df['CabinNumBucket'] = pd.qcut(df['CabinNum'], q=10, labels=False, duplicates='drop')
    df['CabinNumBucket'] = df['CabinNumBucket'].fillna(-1)

    # === NEW: Interaction Features ===
    
    # Spending ratios (avoid division by zero)
    df['RoomServiceRatio'] = df['RoomService'] / (df['TotalSpend'] + 1)
    df['FoodCourtRatio'] = df['FoodCourt'] / (df['TotalSpend'] + 1)
    df['ShoppingMallRatio'] = df['ShoppingMall'] / (df['TotalSpend'] + 1)
    df['SpaRatio'] = df['Spa'] / (df['TotalSpend'] + 1)
    df['VRDeckRatio'] = df['VRDeck'] / (df['TotalSpend'] + 1)
    
    # Entertainment-related spending (Spa + VRDeck)
    df['EntertainSpend'] = df['Spa'] + df['VRDeck']
    df['EntertainRatio'] = df['EntertainSpend'] / (df['TotalSpend'] + 1)
    
    # Necessity-related spending (RoomService + FoodCourt)
    df['NecessitySpend'] = df['RoomService'] + df['FoodCourt']
    df['NecessityRatio'] = df['NecessitySpend'] / (df['TotalSpend'] + 1)

    # Spending per person in group
    df['SpendPerPerson'] = df['TotalSpend'] / df['GroupSize']
    df['LuxurySpend'] = df['Spa'] + df['VRDeck']
    df['LuxurySpendPerPerson'] = df['LuxurySpend'] / df['GroupSize']

    # Log-transformed spending (helps with outliers)
    for col in spend_cols:
        df[f'{col}_log'] = np.log1p(df[col])
    df['TotalSpend_log'] = np.log1p(df['TotalSpend'])

    # Cabin position features
    df['CabinNumNorm'] = df['CabinNum'] / (df['CabinNum'].max() + 1)
    df['CabinPos'] = np.where(df['Side'] == 'P', df['CabinNum'], -df['CabinNum'])
    
    # Planet-Destination interaction (encoded together)
    df['PlanetDest'] = df['HomePlanet'].astype(str) + '_' + df['Destination'].astype(str)
    
    # CryoSleep interactions
    df['CryoAndVIP'] = df['CryoSleep'] * df['VIP']
    df['CryoAndSolo'] = df['CryoSleep'] * df['IsSolo']
    df['CryoAndChild'] = df['CryoSleep'] * df['IsChild']
    
    # Age-related interactions
    df['AgeSpend'] = df['Age'] * df['HasSpend']
    df['AgeCryo'] = df['Age'] * df['CryoSleep']
    df['YoungAndCryo'] = ((df['Age'] < 18) & (df['CryoSleep'] == 1)).astype(int)
    df['AdultNoCryo'] = ((df['Age'] >= 18) & (df['CryoSleep'] == 0)).astype(int)

    # Deck features
    df['DeckNum'] = df['Deck'].map({
        'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8
    }).fillna(0)
    df['IsFrontDeck'] = df['Deck'].isin(['A', 'B', 'C']).astype(int)
    df['IsMidDeck'] = df['Deck'].isin(['D', 'E', 'F']).astype(int)
    df['IsBackDeck'] = df['Deck'].isin(['G', 'T']).astype(int)

    # Frequency encoding for categoricals (reduces overfitting vs label encoding)
    for col in ['HomePlanet', 'Destination', 'Deck', 'Side', 'PlanetDest']:
        freq = df[col].value_counts(normalize=True)
        df[f'{col}_freq'] = df[col].map(freq)

    # Label encode for model
    for col in ['HomePlanet', 'Destination', 'Deck', 'Side', 'PlanetDest']:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
    
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

X = train_df[feature_cols].values.astype(np.float32)
X_test = test_df[feature_cols].values.astype(np.float32)


def run_stacking(X_train, y_train, X_test_data, seeds, n_folds=5, label=""):
    """Run full stacking pipeline, return OOF and test predictions."""
    N_SEEDS = len(seeds)
    lgb_oof = np.zeros(len(X_train))
    xgb_oof = np.zeros(len(X_train))
    cat_oof = np.zeros(len(X_train))
    lgb_test = np.zeros(len(X_test_data))
    xgb_test = np.zeros(len(X_test_data))
    cat_test = np.zeros(len(X_test_data))

    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        _lgb_oof = np.zeros(len(X_train))
        _xgb_oof = np.zeros(len(X_train))
        _cat_oof = np.zeros(len(X_train))
        _lgb_test = np.zeros(len(X_test_data))
        _xgb_test = np.zeros(len(X_test_data))
        _cat_test = np.zeros(len(X_test_data))

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr, X_va = X_train[tr_idx], X_train[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]

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
            _lgb_oof[va_idx] = lgb_m.predict_proba(X_va)[:, 1]
            _lgb_test += lgb_m.predict_proba(X_test_data)[:, 1] / n_folds

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
            _xgb_oof[va_idx] = xgb_m.predict_proba(X_va)[:, 1]
            _xgb_test += xgb_m.predict_proba(X_test_data)[:, 1] / n_folds

            # CatBoost
            cat_m = CatBoostClassifier(
                iterations=800, learning_rate=0.05, depth=6,
                l2_leaf_reg=3.0, min_data_in_leaf=10,
                subsample=0.8, colsample_bylevel=0.8,
                eval_metric='Accuracy', random_seed=seed, verbose=0,
                early_stopping_rounds=50,
            )
            cat_m.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0)
            _cat_oof[va_idx] = cat_m.predict_proba(X_va)[:, 1]
            _cat_test += cat_m.predict_proba(X_test_data)[:, 1] / n_folds

        lgb_oof += _lgb_oof / N_SEEDS
        xgb_oof += _xgb_oof / N_SEEDS
        cat_oof += _cat_oof / N_SEEDS
        lgb_test += _lgb_test / N_SEEDS
        xgb_test += _xgb_test / N_SEEDS
        cat_test += _cat_test / N_SEEDS

    # Stacking
    meta_train = np.column_stack([lgb_oof, xgb_oof, cat_oof])
    meta_test_arr = np.column_stack([lgb_test, xgb_test, cat_test])

    scaler = StandardScaler()
    meta_train_s = scaler.fit_transform(meta_train)
    meta_test_s = scaler.transform(meta_test_arr)

    skf_meta = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    stack_oof = np.zeros(len(X_train))
    stack_test_preds = np.zeros(len(X_test_data))

    for fold, (tr_idx, va_idx) in enumerate(skf_meta.split(meta_train_s, y_train)):
        meta_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        meta_lr.fit(meta_train_s[tr_idx], y_train[tr_idx])
        stack_oof[va_idx] = meta_lr.predict_proba(meta_train_s[va_idx])[:, 1]
        stack_test_preds += meta_lr.predict_proba(meta_test_s)[:, 1] / 5

    cv = accuracy_score(y_train, (stack_oof > 0.5).astype(int))
    print(f"  {label} Stacking CV: {cv:.4f}")
    return stack_oof, stack_test_preds, cv


# ── Round 1: Train on original data ─────────────────────────────────────────
print("\n" + "=" * 60)
print("ROUND 1: Original training data")
print("=" * 60)

seeds_r1 = [42, 123, 456]
stack_oof_r1, stack_test_r1, cv_r1 = run_stacking(
    X, y.values, X_test, seeds_r1, label="Round 1"
)

# ── Pseudo-labeling: add confident test predictions ──────────────────────────
print("\n" + "=" * 60)
print("PSEUDO-LABELING: Adding confident test predictions")
print("=" * 60)

# Select confident predictions (probability > 0.9 or < 0.1)
CONFIDENCE_THR = 0.90
confident_mask = (stack_test_r1 > CONFIDENCE_THR) | (stack_test_r1 < (1 - CONFIDENCE_THR))
pseudo_labels = (stack_test_r1 > 0.5).astype(int)
n_pseudo = confident_mask.sum()
print(f"Confident predictions: {n_pseudo}/{len(X_test)} ({100*n_pseudo/len(X_test):.1f}%)")
print(f"  Pseudo True: {pseudo_labels[confident_mask].sum()}, "
      f"Pseudo False: {(~pseudo_labels[confident_mask].astype(bool)).sum()}")

# Augmented training set
X_aug = np.vstack([X, X_test[confident_mask]])
y_aug = np.concatenate([y.values, pseudo_labels[confident_mask]])
print(f"Augmented training set: {len(X_aug)} ({len(X)} + {n_pseudo} pseudo)")

# ── Round 2: Train on augmented data ─────────────────────────────────────────
print("\n" + "=" * 60)
print("ROUND 2: Augmented training data (with pseudo labels)")
print("=" * 60)

seeds_r2 = [42, 123, 456, 789, 2024]

N_SEEDS = len(seeds_r2)
lgb_test_r2 = np.zeros(len(X_test))
xgb_test_r2 = np.zeros(len(X_test))
cat_test_r2 = np.zeros(len(X_test))

lgb_oof_r2 = np.zeros(len(X))
xgb_oof_r2 = np.zeros(len(X))
cat_oof_r2 = np.zeros(len(X))

for seed in seeds_r2:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    _lgb_oof = np.zeros(len(X))
    _xgb_oof = np.zeros(len(X))
    _cat_oof = np.zeros(len(X))
    _lgb_test = np.zeros(len(X_test))
    _xgb_test = np.zeros(len(X_test))
    _cat_test = np.zeros(len(X_test))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        n_orig = len(X)
        aug_indices = np.arange(n_orig, len(X_aug))
        train_indices = np.concatenate([tr_idx, aug_indices])

        X_tr = X_aug[train_indices]
        y_tr = y_aug[train_indices]
        X_va = X[va_idx]
        y_va = y.values[va_idx]

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
        _lgb_oof[va_idx] = lgb_m.predict_proba(X_va)[:, 1]
        _lgb_test += lgb_m.predict_proba(X_test)[:, 1] / 5

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
        _xgb_oof[va_idx] = xgb_m.predict_proba(X_va)[:, 1]
        _xgb_test += xgb_m.predict_proba(X_test)[:, 1] / 5

        # CatBoost
        cat_m = CatBoostClassifier(
            iterations=800, learning_rate=0.05, depth=6,
            l2_leaf_reg=3.0, min_data_in_leaf=10,
            subsample=0.8, colsample_bylevel=0.8,
            eval_metric='Accuracy', random_seed=seed, verbose=0,
            early_stopping_rounds=50,
        )
        cat_m.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0)
        _cat_oof[va_idx] = cat_m.predict_proba(X_va)[:, 1]
        _cat_test += cat_m.predict_proba(X_test)[:, 1] / 5

    lgb_oof_r2 += _lgb_oof / N_SEEDS
    xgb_oof_r2 += _xgb_oof / N_SEEDS
    cat_oof_r2 += _cat_oof / N_SEEDS
    lgb_test_r2 += _lgb_test / N_SEEDS
    xgb_test_r2 += _xgb_test / N_SEEDS
    cat_test_r2 += _cat_test / N_SEEDS

    lgb_cv = accuracy_score(y, (_lgb_oof > 0.5).astype(int))
    xgb_cv = accuracy_score(y, (_xgb_oof > 0.5).astype(int))
    cat_cv = accuracy_score(y, (_cat_oof > 0.5).astype(int))
    print(f"  Seed {seed}: LGB={lgb_cv:.4f} XGB={xgb_cv:.4f} CAT={cat_cv:.4f}")

# Stacking meta-learner on round 2 OOF
meta_train_r2 = np.column_stack([lgb_oof_r2, xgb_oof_r2, cat_oof_r2])
meta_test_r2 = np.column_stack([lgb_test_r2, xgb_test_r2, cat_test_r2])

scaler = StandardScaler()
meta_train_r2_s = scaler.fit_transform(meta_train_r2)
meta_test_r2_s = scaler.transform(meta_test_r2)

skf_meta = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
stack_oof_r2 = np.zeros(len(y))
stack_test_r2 = np.zeros(len(X_test))

for fold, (tr_idx, va_idx) in enumerate(skf_meta.split(meta_train_r2_s, y)):
    meta_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    meta_lr.fit(meta_train_r2_s[tr_idx], y.values[tr_idx])
    stack_oof_r2[va_idx] = meta_lr.predict_proba(meta_train_r2_s[va_idx])[:, 1]
    stack_test_r2 += meta_lr.predict_proba(meta_test_r2_s)[:, 1] / 5

cv_r2 = accuracy_score(y, (stack_oof_r2 > 0.5).astype(int))
print(f"\n  Round 2 Stacking CV: {cv_r2:.4f}")

# ── Blend Round 1 and Round 2 ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("BLENDING: Round 1 + Round 2")
print("=" * 60)

best_blend_cv = 0
best_alpha = 0.5
for alpha in np.arange(0.0, 1.05, 0.05):
    blend = alpha * stack_test_r1 + (1 - alpha) * stack_test_r2
    blend_oof = alpha * stack_oof_r1 + (1 - alpha) * stack_oof_r2
    cv = accuracy_score(y, (blend_oof > 0.5).astype(int))
    if cv > best_blend_cv:
        best_blend_cv = cv
        best_alpha = alpha

print(f"Best blend: alpha={best_alpha:.2f} (R1 weight), CV={best_blend_cv:.4f}")
print(f"  Round 1 CV: {cv_r1:.4f}")
print(f"  Round 2 CV: {cv_r2:.4f}")
print(f"  Blend CV: {best_blend_cv:.4f}")

final_oof = best_alpha * stack_oof_r1 + (1 - best_alpha) * stack_oof_r2
final_test = best_alpha * stack_test_r1 + (1 - best_alpha) * stack_test_r2

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
sub.to_csv('submission_interact.csv', index=False)
print(f"\nsubmission_interact.csv saved (thr={best_thr:.3f}):")
print(sub['Transported'].value_counts())

# Also save 0.5 threshold version
sub_05 = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Transported': (final_test > 0.5)
})
sub_05.to_csv('submission_interact_05.csv', index=False)
print(f"\nsubmission_interact_05.csv saved (thr=0.5):")
print(sub_05['Transported'].value_counts())

# Save best as main
sub_05.to_csv('submission.csv', index=False)
print(f"\nSaved submission.csv (thr=0.5, blend CV={best_blend_cv:.4f})")
print("\nDone!")
