import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os

# --- 1. Load Data ---
ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT, '..', 'data')
TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')

print("Loading data...")
df_train = pd.read_csv(TRAIN_PATH)

# --- 2. Remove "Id" column if present ---
if "Id" in df_train.columns:
    df_train = df_train.drop("Id", axis=1)

# --- 3. Split into X and y ---
y = df_train["SalePrice"]
X = df_train.drop("SalePrice", axis=1)

# --- 4. Detect numeric + categorical features ---
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_features = X.select_dtypes(include=["object"]).columns.tolist()

# --- 5. Remove columns with >50% missing data ---
missing_pct = X.isnull().mean() * 100
high_missing_cols = missing_pct[missing_pct > 50].index.tolist()

for col in high_missing_cols:
    if col in numeric_features:
        numeric_features.remove(col)
    if col in cat_features:
        cat_features.remove(col)

print("Numeric features used:", numeric_features)
print("Categorical features used:", cat_features)

# --- 6. Preprocessing ---

# Numerics → Median Imputation + Scaling
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical → Most Frequent + OneHotEncoder
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, cat_features)
    ],
    remainder="drop"
)

# --- 7. Full Model Pipeline ---
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

# --- 8. Train-Test Split ---
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")
model.fit(X_train, y_train)

# --- 9. Validation ---
preds = model.predict(X_valid)
rmse = np.sqrt(mean_squared_error(y_valid, preds))
print("Validation RMSE:", rmse)

# --- 10. Save Model + Preprocessing ---
MODEL_PATH = os.path.join(ROOT, "model.pkl")
PREPROCESS_PATH = os.path.join(ROOT, "preprocess.pkl")

joblib.dump(model, MODEL_PATH)
print("Saved trained model to:", MODEL_PATH)

meta = {
    "numeric_features": numeric_features,
    "cat_features": cat_features
}
joblib.dump(meta, PREPROCESS_PATH)
print("Saved metadata to:", PREPROCESS_PATH)
