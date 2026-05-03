"""
Spaceship Titanic - Multi-CatBoost v6
Goal: Beat LB 0.80967.
Strategy:
  1. Same proven 21-feature set (more = overfit)
  2. Multiple CatBoost configs with different depths/params for diversity
  3. Iterative pseudo-labeling (2 rounds, increasing confidence)
  4. Majority vote across diverse CatBoost models
  5. 7 seeds for stability
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier, Pool
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("SPACESHIP TITANIC - Multi-CatBoost v6")
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

    df['GroupSize'] = df.groupby('GroupId')['PassengerId'].transform('count')
    df['IsSolo'] = (df['GroupSize'] == 1).astype(int)

    df['TotalSpend'] = df[spend_cols].sum(axis=1)
    df['HasSpend'] = (df['TotalSpend'] > 0).astype(int)
    df['TotalSpend_log'] = np.log1p(df['TotalSpend'])
    df['SpendPerPerson'] = df['TotalSpend'] / df['GroupSize']

    df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 18, 25, 50, 100],
                          labels=[0, 1, 2, 3, 4]).astype(float)
    df['IsChild'] = (df['Age'] < 13).astype(int)

    df['LastName'] = df['Name'].str.split().str[-1]
    family_size = df.groupby('LastName')['PassengerId'].transform('count')
    df['FamilySize'] = family_size.fillna(1).clip(upper=10)

    df['CabinNumBucket'] = pd.qcut(df['CabinNum'], q=10, labels=False, duplicates='drop')
    df['CabinNumBucket'] = df['CabinNumBucket'].fillna(-1)

    df['DeckSide'] = df['Deck'].astype(str) + '_' + df['Side'].astype(str)
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

cat_cols = [c for c in CAT_FEATURES if c in feature_cols]

# CatBoost DataFrames with string categoricals
X_cat = train_df[feature_cols].copy()
X_test_cat = test_df[feature_cols].copy()
for col in cat_cols:
    X_cat[col] = X_cat[col].astype(str)
    X_test_cat[col] = X_test_cat[col].astype(str)
for col in feature_cols:
    if col not in cat_cols:
        X_cat[col] = X_cat[col].astype(np.float32)
        X_test_cat[col] = X_test_cat[col].astype(np.float32)

# Define diverse CatBoost configurations
CONFIGS = {
    'cat_d6': dict(
        iterations=1000, learning_rate=0.05, depth=6,
        l2_leaf_reg=5.0, min_data_in_leaf=15,
        subsample=0.8, colsample_bylevel=0.7,
        one_hot_max_size=10,
    ),
    'cat_d5': dict(
        iterations=1000, learning_rate=0.03, depth=5,
        l2_leaf_reg=7.0, min_data_in_leaf=20,
        subsample=0.75, colsample_bylevel=0.65,
        one_hot_max_size=10, random_strength=0.5,
    ),
    'cat_d7': dict(
        iterations=800, learning_rate=0.05, depth=7,
        l2_leaf_reg=3.0, min_data_in_leaf=10,
        subsample=0.85, colsample_bylevel=0.75,
        one_hot_max_size=10,
    ),
    'cat_d4': dict(
        iterations=1200, learning_rate=0.03, depth=4,
        l2_leaf_reg=10.0, min_data_in_leaf=30,
        subsample=0.7, colsample_bylevel=0.6,
        one_hot_max_size=10, random_strength=1.0,
    ),
}

n_train = len(y)
n_test = len(X_test_cat)
seeds = [42, 123, 456, 789, 2024, 7, 314]
N_SEEDS = len(seeds)
N_FOLDS = 5

config_names = list(CONFIGS.keys())
N_CONFIGS = len(config_names)


def train_all_configs(X_train_df, X_test_df, y_train, seeds, n_folds=5,
                      label="", X_val_df=None, y_val=None, augmented=False):
    """Train all CatBoost configs and return OOF + test predictions."""
    if augmented and X_val_df is not None:
        n_oof = len(y_val)
    else:
        n_oof = len(y_train)

    oof = {c: np.zeros(n_oof) for c in config_names}
    test_preds = {c: np.zeros(n_test) for c in config_names}

    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        seed_oof = {c: np.zeros(n_oof) for c in config_names}
        seed_test = {c: np.zeros(n_test) for c in config_names}

        if augmented:
            split_X = X_val_df
            split_y = y_val
        else:
            split_X = X_train_df
            split_y = y_train

        for fold, (tr_idx, va_idx) in enumerate(skf.split(split_X, split_y)):
            if augmented:
                n_orig = len(y_val)
                aug_indices = np.arange(n_orig, len(y_train))
                train_indices = np.concatenate([tr_idx, aug_indices])
                X_tr = X_train_df.iloc[train_indices]
                y_tr = y_train[train_indices]
                X_va = X_val_df.iloc[va_idx]
                y_va = split_y[va_idx]
            else:
                X_tr = X_train_df.iloc[tr_idx]
                y_tr = y_train[tr_idx]
                X_va = X_train_df.iloc[va_idx]
                y_va = y_train[va_idx]

            train_pool = Pool(X_tr, y_tr, cat_features=cat_cols)
            val_pool = Pool(X_va, y_va, cat_features=cat_cols)
            test_pool = Pool(X_test_df, cat_features=cat_cols)

            for cfg_name in config_names:
                params = CONFIGS[cfg_name].copy()
                cat_m = CatBoostClassifier(
                    **params,
                    eval_metric='Accuracy', random_seed=seed, verbose=0,
                    early_stopping_rounds=60,
                )
                cat_m.fit(train_pool, eval_set=val_pool, verbose=0)
                seed_oof[cfg_name][va_idx] = cat_m.predict_proba(val_pool)[:, 1]
                seed_test[cfg_name] += cat_m.predict_proba(test_pool)[:, 1] / n_folds

        for c in config_names:
            oof[c] += seed_oof[c] / len(seeds)
            test_preds[c] += seed_test[c] / len(seeds)

        accs = " ".join(
            f"{c}={accuracy_score(split_y, (seed_oof[c] > 0.5).astype(int)):.4f}"
            for c in config_names
        )
        print(f"  {label} Seed {seed}: {accs}")

    return oof, test_preds


# ── Round 1 ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ROUND 1: Original training data")
print("=" * 60)

oof_r1, test_r1 = train_all_configs(
    X_cat, X_test_cat, y.values, seeds, label="R1"
)

print("\nR1 individual OOF:")
for c in config_names:
    cv = accuracy_score(y, (oof_r1[c] > 0.5).astype(int))
    print(f"  {c}: {cv:.4f}")

# Find optimal weights
best_cv_r1 = 0
best_w = {c: 1.0 / N_CONFIGS for c in config_names}
step = 0.05
for w0 in np.arange(0.1, 0.6, step):
    for w1 in np.arange(0.05, 0.5, step):
        for w2 in np.arange(0.05, 0.5, step):
            w3 = 1 - w0 - w1 - w2
            if w3 < 0.05 or w3 > 0.5:
                continue
            weights = [w0, w1, w2, w3]
            blend = sum(w * oof_r1[c] for w, c in zip(weights, config_names))
            cv = accuracy_score(y, (blend > 0.5).astype(int))
            if cv > best_cv_r1:
                best_cv_r1 = cv
                best_w = dict(zip(config_names, weights))

print(f"\nBest weights: {best_w}")
print(f"Best weighted avg CV: {best_cv_r1:.4f}")

blend_test_r1 = sum(best_w[c] * test_r1[c] for c in config_names)
blend_oof_r1 = sum(best_w[c] * oof_r1[c] for c in config_names)

# Simple average for comparison
avg_test_r1 = sum(test_r1[c] for c in config_names) / N_CONFIGS
avg_oof_r1 = sum(oof_r1[c] for c in config_names) / N_CONFIGS
avg_cv_r1 = accuracy_score(y, (avg_oof_r1 > 0.5).astype(int))
print(f"Simple average CV: {avg_cv_r1:.4f}")

# ── Pseudo-labeling Round 1 ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PSEUDO-LABELING ROUND 1")
print("=" * 60)

# Use simple average for pseudo-labels (less overfit to OOF)
CONFIDENCE_THR_1 = 0.92
confident_mask_1 = (avg_test_r1 > CONFIDENCE_THR_1) | (avg_test_r1 < (1 - CONFIDENCE_THR_1))
pseudo_labels_1 = (avg_test_r1 > 0.5).astype(int)
n_pseudo_1 = confident_mask_1.sum()
print(f"Confident (thr={CONFIDENCE_THR_1}): {n_pseudo_1}/{n_test} ({100*n_pseudo_1/n_test:.1f}%)")

X_cat_aug1 = pd.concat([X_cat, X_test_cat[confident_mask_1]], ignore_index=True)
y_aug1 = np.concatenate([y.values, pseudo_labels_1[confident_mask_1]])
print(f"Augmented: {len(X_cat_aug1)} ({n_train} + {n_pseudo_1})")

# ── Round 2 ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ROUND 2: With pseudo-labels (high confidence)")
print("=" * 60)

oof_r2, test_r2 = train_all_configs(
    X_cat_aug1, X_test_cat, y_aug1, seeds, label="R2",
    X_val_df=X_cat, y_val=y.values, augmented=True
)

print("\nR2 individual OOF:")
for c in config_names:
    cv = accuracy_score(y, (oof_r2[c] > 0.5).astype(int))
    print(f"  {c}: {cv:.4f}")

blend_oof_r2 = sum(best_w[c] * oof_r2[c] for c in config_names)
blend_test_r2 = sum(best_w[c] * test_r2[c] for c in config_names)
cv_r2 = accuracy_score(y, (blend_oof_r2 > 0.5).astype(int))
print(f"R2 Weighted avg CV: {cv_r2:.4f}")

avg_test_r2 = sum(test_r2[c] for c in config_names) / N_CONFIGS
avg_oof_r2 = sum(oof_r2[c] for c in config_names) / N_CONFIGS
avg_cv_r2 = accuracy_score(y, (avg_oof_r2 > 0.5).astype(int))
print(f"R2 Simple average CV: {avg_cv_r2:.4f}")

# ── Blend R1 + R2 ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BLEND R1 + R2")
print("=" * 60)

# Try blending weighted averages
best_blend_cv = 0
best_alpha = 0.5
for alpha in np.arange(0.0, 1.05, 0.05):
    blend = alpha * blend_oof_r1 + (1 - alpha) * blend_oof_r2
    cv = accuracy_score(y, (blend > 0.5).astype(int))
    if cv > best_blend_cv:
        best_blend_cv = cv
        best_alpha = alpha
print(f"Weighted blend: alpha={best_alpha:.2f}, CV={best_blend_cv:.4f}")

# Try blending simple averages
best_avg_blend_cv = 0
best_avg_alpha = 0.5
for alpha in np.arange(0.0, 1.05, 0.05):
    blend = alpha * avg_oof_r1 + (1 - alpha) * avg_oof_r2
    cv = accuracy_score(y, (blend > 0.5).astype(int))
    if cv > best_avg_blend_cv:
        best_avg_blend_cv = cv
        best_avg_alpha = alpha
print(f"Simple avg blend: alpha={best_avg_alpha:.2f}, CV={best_avg_blend_cv:.4f}")

# Use whichever blend is better
if best_blend_cv >= best_avg_blend_cv:
    final_test = best_alpha * blend_test_r1 + (1 - best_alpha) * blend_test_r2
    final_oof = best_alpha * blend_oof_r1 + (1 - best_alpha) * blend_oof_r2
    final_cv = best_blend_cv
    blend_type = "weighted"
else:
    final_test = best_avg_alpha * avg_test_r1 + (1 - best_avg_alpha) * avg_test_r2
    final_oof = best_avg_alpha * avg_oof_r1 + (1 - best_avg_alpha) * avg_oof_r2
    final_cv = best_avg_blend_cv
    blend_type = "simple_avg"

print(f"\nUsing {blend_type} blend, CV={final_cv:.4f}")

# ── Also compute majority vote across all configs + rounds ───────────────────
print("\n" + "=" * 60)
print("MAJORITY VOTE")
print("=" * 60)

# Each config x each round = 8 binary predictions
all_preds = []
all_oof_preds = []
for c in config_names:
    all_preds.append((test_r1[c] > 0.5).astype(int))
    all_preds.append((test_r2[c] > 0.5).astype(int))
    all_oof_preds.append((oof_r1[c] > 0.5).astype(int))
    all_oof_preds.append((oof_r2[c] > 0.5).astype(int))

vote_test = np.mean(all_preds, axis=0)
vote_oof = np.mean(all_oof_preds, axis=0)
vote_cv = accuracy_score(y, (vote_oof > 0.5).astype(int))
print(f"Majority vote CV (8 voters): {vote_cv:.4f}")

# ── Save submissions ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUBMISSIONS")
print("=" * 60)

# 1. Best probability blend
sub_prob = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Transported': (final_test > 0.5)
})
sub_prob.to_csv('submission_multi_cat_v6_prob.csv', index=False)
print(f"submission_multi_cat_v6_prob.csv ({blend_type}, CV={final_cv:.4f}):")
print(sub_prob['Transported'].value_counts())

# 2. Majority vote
sub_vote = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Transported': (vote_test > 0.5)
})
sub_vote.to_csv('submission_multi_cat_v6_vote.csv', index=False)
print(f"\nsubmission_multi_cat_v6_vote.csv (majority vote, CV={vote_cv:.4f}):")
print(sub_vote['Transported'].value_counts())

# 3. Simple average of all config test predictions (R1+R2 combined)
all_avg_test = sum(test_r1[c] + test_r2[c] for c in config_names) / (2 * N_CONFIGS)
sub_avg = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Transported': (all_avg_test > 0.5)
})
sub_avg.to_csv('submission_multi_cat_v6_avg.csv', index=False)
all_avg_oof = sum(oof_r1[c] + oof_r2[c] for c in config_names) / (2 * N_CONFIGS)
avg_cv_all = accuracy_score(y, (all_avg_oof > 0.5).astype(int))
print(f"\nsubmission_multi_cat_v6_avg.csv (all avg, CV={avg_cv_all:.4f}):")
print(sub_avg['Transported'].value_counts())

# Pick best CV submission as main
options = [
    ('prob', final_cv, sub_prob),
    ('vote', vote_cv, sub_vote),
    ('avg', avg_cv_all, sub_avg),
]
best_name, best_cv_val, best_sub = max(options, key=lambda x: x[1])
best_sub.to_csv('submission.csv', index=False)
print(f"\nSaved submission.csv (multi_cat_v6_{best_name}, CV={best_cv_val:.4f})")

# Count disagreements between prob and vote
disagree = (sub_prob['Transported'] != sub_vote['Transported']).sum()
print(f"\nDisagreements prob vs vote: {disagree}/{n_test}")

print("\nDone!")
