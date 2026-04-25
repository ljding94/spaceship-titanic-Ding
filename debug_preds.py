import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(f'Train: {train.shape}, Test: {test.shape}')

def engineer_features(df):
    df = df.copy()
    df['Deck'] = df['Cabin'].str.split('/').str[0]
    df['CabinNum'] = df['Cabin'].str.split('/').str[1].astype(float)
    df['Side'] = df['Cabin'].str.split('/').str[2]
    df['GroupId'] = df['PassengerId'].str.split('_').str[0].astype(int)
    df['PersonNum'] = df['PassengerId'].str.split('_').str[1].astype(int)
    group_sizes = df.groupby('GroupId')['PassengerId'].transform('count')
    df['GroupSize'] = group_sizes
    df['IsSolo'] = (df['GroupSize'] == 1).astype(int)
    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['TotalSpend'] = df[spend_cols].sum(axis=1)
    df['HasSpend'] = (df['TotalSpend'] > 0).astype(int)
    df['NumSpendCategories'] = (df[spend_cols] > 0).sum(axis=1)
    for col in spend_cols + ['TotalSpend']:
        df[f'{col}_log'] = np.log1p(df[col])
    df['RoomServiceRatio'] = df['RoomService'] / (df['TotalSpend'] + 1)
    df['FoodCourtRatio'] = df['FoodCourt'] / (df['TotalSpend'] + 1)
    df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 18, 25, 35, 50, 65, 100], labels=[0, 1, 2, 3, 4, 5, 6]).astype(float)
    df['IsChild'] = (df['Age'] < 13).astype(float)
    df['IsTeenager'] = ((df['Age'] >= 13) & (df['Age'] < 18)).astype(float)
    df['CryoSleep'] = df['CryoSleep'].map({True: 1, False: 0, 'True': 1, 'False': 0})
    df['VIP'] = df['VIP'].map({True: 1, False: 0, 'True': 1, 'False': 0})
    for col in spend_cols:
        mask = (df['CryoSleep'] == 1) & (df[col].isnull())
        df.loc[mask, col] = 0.0
    df.loc[(df['TotalSpend'] == 0) & (df['CryoSleep'].isnull()), 'CryoSleep'] = 1.0
    for col in ['HomePlanet', 'Destination', 'Deck', 'Side']:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
    return df

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
combined = engineer_features(combined)
train_df = combined[combined['is_train'] == 1].copy()
test_df = combined[combined['is_train'] == 0].copy()

y = train_df['Transported'].astype(int)
drop_cols = ['PassengerId', 'Cabin', 'Name', 'Transported', 'is_train', 'GroupId']
feature_cols = [c for c in train_df.columns if c not in drop_cols]
X = train_df[feature_cols].values
X_test = test_df[feature_cols].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lgb_test_preds = np.zeros(X_test.shape[0])
lgb_oof = np.zeros(len(X))
xgb_test_preds = np.zeros(X_test.shape[0])
xgb_oof = np.zeros(len(X))

lgb_params = {'objective': 'binary', 'metric': 'binary_error', 'boosting_type': 'gbdt', 'learning_rate': 0.03, 'num_leaves': 31, 'max_depth': 6, 'min_child_samples': 20, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'n_estimators': 500, 'verbose': -1, 'random_state': 42}
xgb_params = {'objective': 'binary:logistic', 'eval_metric': 'error', 'learning_rate': 0.03, 'max_depth': 6, 'min_child_weight': 3, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'n_estimators': 500, 'early_stopping_rounds': 50, 'verbosity': 0, 'random_state': 42, 'tree_method': 'hist'}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y.values[train_idx], y.values[val_idx]
    
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    lgb_oof[val_idx] = lgb_model.predict_proba(X_val)[:, 1]
    lgb_test_preds += lgb_model.predict_proba(X_test)[:, 1] / 5
    
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    xgb_oof[val_idx] = xgb_model.predict_proba(X_val)[:, 1]
    xgb_test_preds += xgb_model.predict_proba(X_test)[:, 1] / 5

ensemble_oof = (lgb_oof + xgb_oof) / 2
ensemble_test = (lgb_test_preds + xgb_test_preds) / 2

print(f'LGB OOF: min={lgb_oof.min():.4f}, max={lgb_oof.max():.4f}, mean={lgb_oof.mean():.4f}')
print(f'XGB OOF: min={xgb_oof.min():.4f}, max={xgb_oof.max():.4f}, mean={xgb_oof.mean():.4f}')
print(f'Ensemble OOF: min={ensemble_oof.min():.4f}, max={ensemble_oof.max():.4f}, mean={ensemble_oof.mean():.4f}')
print(f'LGB CV acc: {accuracy_score(y, (lgb_oof > 0.5).astype(int)):.4f}')
print(f'XGB CV acc: {accuracy_score(y, (xgb_oof > 0.5).astype(int)):.4f}')
print(f'Ensemble CV acc: {accuracy_score(y, (ensemble_oof > 0.5).astype(int)):.4f}')
print(f'Ensemble test: min={ensemble_test.min():.4f}, max={ensemble_test.max():.4f}, mean={ensemble_test.mean():.4f}')
print(f'Ensemble test > 0.5: {(ensemble_test > 0.5).sum()}/{len(ensemble_test)}')

# Save submission
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Transported': (ensemble_test > 0.5)})
submission.to_csv('submission.csv', index=False)
print(f'Submission:\n{submission["Transported"].value_counts()}')
