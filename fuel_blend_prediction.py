# Enhanced Stacking Model with Additional Base Learners

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample_submission = pd.read_csv("sample_solution.csv")

X = train.iloc[:, :55]
y = train.iloc[:, 55:]
test_ids = test["ID"]
test = test.drop(columns=["ID"])

# Feature engineering
for p in range(1, 11):
    prop_cols = [f"Component{i}_Property{p}" for i in range(1, 6)]
    fractions = [f"Component{i}_fraction" for i in range(1, 6)]

    X[f"weighted_Property{p}"] = sum(X[fractions[i]] * X[prop_cols[i]] for i in range(5))
    test[f"weighted_Property{p}"] = sum(test[fractions[i]] * test[prop_cols[i]] for i in range(5))

    X[f"Property{p}_mean"] = X[prop_cols].mean(axis=1)
    X[f"Property{p}_std"] = X[prop_cols].std(axis=1)
    test[f"Property{p}_mean"] = test[prop_cols].mean(axis=1)
    test[f"Property{p}_std"] = test[prop_cols].std(axis=1)

    safe_vals = X[prop_cols].replace(0, 1e-6)
    test_safe_vals = test[prop_cols].replace(0, 1e-6)
    X[f"Property{p}_gmean"] = np.exp(np.log(safe_vals).mean(axis=1))
    X[f"Property{p}_hmean"] = 5 / (1 / safe_vals).sum(axis=1)
    test[f"Property{p}_gmean"] = np.exp(np.log(test_safe_vals).mean(axis=1))
    test[f"Property{p}_hmean"] = 5 / (1 / test_safe_vals).sum(axis=1)

    for i in range(1, 6):
        X[f"interaction_C{i}_P{p}"] = X[f"Component{i}_fraction"] * X[f"Component{i}_Property{p}"]
        test[f"interaction_C{i}_P{p}"] = test[f"Component{i}_fraction"] * test[f"Component{i}_Property{p}"]

X = X.fillna(0)
test = test.fillna(0)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test)

# Autoencoder
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=32):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, embedding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def get_embedding(self, x):
        return self.encoder(x)

# Train autoencoder
print("Training autoencoder...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
ae_model = AutoEncoder(input_dim=X.shape[1]).to(device)
optimizer = torch.optim.Adam(ae_model.parameters(), lr=0.001)
criterion = nn.MSELoss()
dataloader = DataLoader(TensorDataset(X_tensor, X_tensor), batch_size=64, shuffle=True)

for epoch in range(30):
    ae_model.train()
    for xb, yb in dataloader:
        optimizer.zero_grad()
        loss = criterion(ae_model(xb), yb)
        loss.backward()
        optimizer.step()

ae_model.eval()
with torch.no_grad():
    X_embed = ae_model.get_embedding(X_tensor).cpu().numpy()
    test_embed = ae_model.get_embedding(torch.tensor(test_scaled, dtype=torch.float32).to(device)).cpu().numpy()

# PCA
pca = PCA(n_components=16)
X_pca = pca.fit_transform(X_scaled)
test_pca = pca.transform(test_scaled)

# Combine features
X_all = np.hstack([X_scaled, X_embed, X_pca])
test_all = np.hstack([test_scaled, test_embed, test_pca])

# Model training with enhanced stacking
final_preds = np.zeros((test.shape[0], y.shape[1]))
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for t in range(y.shape[1]):
    print(f"\nTarget {t+1}/10")
    oof = np.zeros(X_all.shape[0])
    preds = np.zeros(test.shape[0])

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_all)):
        X_tr, X_val = X_all[tr_idx], X_all[val_idx]
        y_tr, y_val = y.iloc[tr_idx, t], y.iloc[val_idx, t]

        base_models = [
            CatBoostRegressor(iterations=1000, learning_rate=0.03, depth=6, verbose=0, random_seed=42),
            LGBMRegressor(n_estimators=1000, learning_rate=0.03, num_leaves=31, random_state=42),
            Ridge(alpha=1.0),
            MLPRegressor(hidden_layer_sizes=(64, 32), early_stopping=True, random_state=42),
            RandomForestRegressor(n_estimators=500, max_depth=10, random_state=42),
            ExtraTreesRegressor(n_estimators=500, random_state=42),
            HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, random_state=42)
        ]

        meta_features_val = []
        meta_features_test = []

        for model in base_models:
            model.fit(X_tr, y_tr)
            meta_features_val.append(model.predict(X_val))
            meta_features_test.append(model.predict(test_all))

        meta_val = np.vstack(meta_features_val).T
        meta_test = np.vstack(meta_features_test).T

        meta_model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, random_state=42)
        meta_model.fit(meta_val, y_val)
        oof[val_idx] = meta_model.predict(meta_val)
        preds += meta_model.predict(meta_test) / kf.n_splits

    rmse = np.sqrt(mean_squared_error(y.iloc[:, t], oof))
    print(f"✅ RMSE Target {t+1}: {rmse:.4f}")
    final_preds[:, t] = preds

# Submission
submission = sample_submission.copy()
submission.iloc[:, 1:] = np.clip(final_preds, sample_submission.iloc[:, 1:].min().values, sample_submission.iloc[:, 1:].max().values)
submission["ID"] = test_ids
submission.to_csv("final_submission.csv", index=False)
print("\n🎯 Final submission saved as final_submission.csv")
