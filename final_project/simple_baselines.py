import pandas as pd # Pandas implementation
from sklearn.model_selection import train_test_split # Train-Test-Split
from sklearn.preprocessing import StandardScaler # Standard Scaler
from sklearn.ensemble import RandomForestRegressor # Random Forest Regressor
from sklearn.linear_model import Ridge # Ridge Regression
from sklearn.metrics import mean_squared_error # MSE for simple accuracy
import numpy as np

df = pd.read_csv("hfrp_dataset.csv")

# Keep only rows where target and numeric features are all present
num_cols = ["revenue", "net_income", "operating_income", "eps", "total_assets"]
df = df.dropna(subset=["future_vol"] + num_cols)
print("Final baseline dataset shape:", df.shape)


X_num = df[num_cols].values # a matrix of shape (n_samples, 7); these are my input features
y = df["future_vol"].values # my risk target, future volatility; regression to predict this variable

X_train, X_test, y_train, y_test = train_test_split( # The classic function to split the data into train, test, and split
    X_num, y, test_size=0.2, random_state=42
)

scaler = StandardScaler() # Scaler to standardize features by removing the mean and scaling to unit variance
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# This is one of my baseline models; it is ridge regression; we use a parameter alpha=1.0 for regularization
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
pred_ridge = ridge.predict(X_test_scaled)
rmse_ridge = np.sqrt(mean_squared_error(y_test, pred_ridge))

# Another baseline model; this is random forest regression; we use 200 trees here
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train_scaled, y_train)
pred_rf = rf.predict(X_test_scaled)
rmse_rf = np.sqrt(mean_squared_error(y_test, pred_rf))

print("RMSE Ridge:", rmse_ridge) # simple metrics for accuracy; these will be comapred to the hyrbid model
print("RMSE RF   :", rmse_rf)
