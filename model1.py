import pandas as pd
from xgboost import XGBRegressor
from catboost import CatBoostRegressor, Pool
from sklearn.multioutput import MultiOutputRegressor

# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample_submission = pd.read_csv("sample_submission.csv")

# Features (first 55 columns), Targets (last 10 columns)
X = train.iloc[:, :55]
y = train.iloc[:, 55:]

# ------ XGBOOST MODEL ------
xgb_model = MultiOutputRegressor(XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    tree_method="hist"  # Faster on CPUs
))
xgb_model.fit(X, y)
xgb_preds = xgb_model.predict(test)

# ------ CATBOOST MODEL ------
catboost_model = MultiOutputRegressor(CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    verbose=0,
    random_seed=42
))
catboost_model.fit(X, y)
catboost_preds = catboost_model.predict(test)

# ------ AVERAGING ENSEMBLE ------
final_preds = (xgb_preds + catboost_preds) / 2

# Prepare submission
submission = sample_submission.copy()
submission.iloc[:, 1:] = final_preds
submission.to_csv("final_submission.csv", index=False)
