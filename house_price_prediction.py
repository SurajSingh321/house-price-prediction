# ============================================================
#   HOUSE PRICE PREDICTION
#   Model: Linear Regression
#   Dataset: Housing Price Prediction (Kaggle)
# ============================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score

# ── 1. LOAD DATASET ─────────────────────────────────────────
df = pd.read_csv("project_3/Housing.csv")

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn names:")
print(df.columns.to_list())

# ── 2. BASIC CLEANING ───────────────────────────────────────
# Drop rows with missing values
df = df.dropna()
print(f"\nAfter dropping nulls:{df.shape}")

# ── 3. HANDLE CATEGORICAL COLUMNS ───────────────────────────
# If any column is text (yes/no etc.), convert to numbers
# get_dummies converts categorical → 0/1 columns automatically
df = pd.get_dummies(df,drop_first=True)
print(f"\nAfter encoding categoricals: {df.shape}")

# ── 4. SPLIT FEATURES AND LABEL ─────────────────────────────
# Last column is usually price — adjust if needed
target_col ="price" # change this if your dataset has different column name

X = df.drop(target_col,axis=1)
y = df[target_col]

print(f"\nFeatures: {X.columns.tolist()}")

# ── 5. TRAIN-TEST SPLIT ──────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X,y , test_size=0.2 , random_state=42
)

print(f"\nTraining samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")

# ── 6. FEATURE SCALING ──────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── 7. TRAIN MODEL ──────────────────────────────────────────
model = LinearRegression()
model.fit(X_train_scaled,y_train)

# ── 8. EVALUATE ─────────────────────────────────────────────
y_pred = model.predict(X_test_scaled)
 
mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print(f"\nMAE  (Mean Absolute Error) : {mae:.2f}")
print(f"RMSE (Root Mean Sq Error)  : {rmse:.2f}")
print(f"R2 Score                   : {r2:.4f}")
print("\nR2 Score :")
print("  1.0 = perfect prediction")
print("  0.7+ = good model")
print("  <0.5 = weak model")

# ── 9. FEATURE IMPORTANCE ───────────────────────────────────
print("\nFeature Weights (importance):")
for name, coef in sorted(zip(X.columns, model.coef_), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {name:30s} : {coef:.4f}")






