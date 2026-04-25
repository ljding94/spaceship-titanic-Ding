"""
Spaceship Titanic — XGBoost Baseline
CV: 0.802 ± 0.008 | LB: 0.799
Simple features: median fill + label encode, no complex engineering.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 50)
print("SPACESHIP TITANIC — XGBoost Baseline")
print("=" * 50)

# Load
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(f"Train: {train.shape}, Test: {test.shape}")
print(f"Target balance: {train['Transported'].mean():.3f}")

# ── Minimal Feature Engineering ───────────────────────────────────────────────
def baseline_features(df):
    df = df.copy()

    # Cabin → deck / num / side
    df['Deck'] = df['Cabin'].str.split('/').str[0]
    df['CabinNum'] = df['Cabin'].str.split('/').str[1].astype(float)
    df['Side'] = df['Cabin'].str.split('/').str[2]

    # Group size from PassengerId
    df['GroupId'] = df['PassengerId'].str.split('_').str[0].astype(int)
    df['PersonNum'] = df['PassengerId'].str.split('_').str[1].astype(int)
    group_sizes = df.groupby('GroupId')['PassengerId'].transform('count')
    df['GroupSize'] = group_sizes
    df['IsSolo'] = (df['GroupSize'] == 1).astype(int)

    # Total spend
    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['TotalSpend'] = df[spend_cols].sum(axis=1)

    # Age bins
    df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 18, 25, 35, 50, 65, 100],
                          labels=[0, 1, 2, 3, 4, 5, 6]).astype(float)

    # CryoSleep / VIP encoding
    df['CryoSleep'] = df['CryoSleep'].map({True: 1, False: 0, 'True': 1, 'False': 0})
    df['VIP'] = df['VIP'].map({True: 1, False: 0, 'True': 1, 'False': 0})

    # Label encode categoricals
    for col in ['HomePlanet', 'Destination', 'Deck', 'Side']:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])

    return df

# Combine for consistent encoding
train['is_train'] = 1
test['is_train'] = 0
test['Transported'] = np.nan
combined = pd.concat([train, test], axis=0, ignore_index=True)

spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for col in spend_cols:
    combined[col] = combined[col].fillna(combined[col].median())
combined['Age'] = combined['Age'].fillna(combined['Age'].median())
combined['HomePlanet'] = combined['HomePlanet'].fillna(combined['HomePlanet'].mode()[0])
combined['Destination'] = combined['Destination'].fillna(combined['Destination'].mode()[0])
combined['CryoSleep'] = combined['CryoSleep'].fillna(combined['CryoSleep'].mode()[0])
combined['VIP'] = combined['VIP'].fillna(False)

combined = baseline_features(combined)

train_df = combined[combined['is_train'] == 1].copy()
test_df = combined[combined['is_train'] == 0].copy()

y = train_df['Transported'].astype(int)
drop_cols = ['PassengerId', 'Cabin', 'Name', 'Transported', 'is_train', 'GroupId']
feature_cols = [c for c in train_df.columns if c not in drop_cols]
print(f"Features ({len(feature_cols)}): {feature_cols}")

X = train_df[feature_cols].values
X_test = test_df[feature_cols].values

# ── 5-Fold CV with XGBoost ───────────────────────────────────────────────────
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'error',
    'learning_rate': 0.05,
    'max_depth': 5,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'n_estimators': 500,
    'early_stopping_rounds': 50,
    'verbosity': 0,
    'random_state': 42,
    'tree_method': 'hist',
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []
test_preds = np.zeros(X_test.shape[0])

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y.values[train_idx], y.values[val_idx]

    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    val_pred = model.predict(X_val)
    acc = accuracy_score(y_val, val_pred)
    scores.append(acc)
    test_preds += model.predict_proba(X_test)[:, 1] / 5
    print(f"  Fold {fold+1}: {acc:.4f}")

cv_mean = np.mean(scores)
cv_std = np.std(scores)
print(f"\nXGBoost CV: {cv_mean:.4f} ± {cv_std:.4f}")

# Submission
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Transported': (test_preds > 0.5)})
submission.to_csv('submission_baseline.csv', index=False)
print(f"\nSubmission saved: {submission.shape}")
print(submission['Transported'].value_counts())
print(f"\nDone! CV={cv_mean:.4f}")
