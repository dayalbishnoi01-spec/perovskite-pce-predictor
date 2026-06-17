import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

df = pd.read_csv("perovskite_15.csv")
print(f"Shape: {df.shape}")
print(df.head())
# Drop leakage columns
df = df.drop(columns=["Voc", "Jsc", "FF"])

# Drop missing PCE
df = df.dropna(subset=["PCE"])
df = df[(df["PCE"] > 0) & (df["PCE"] <= 28)]

# Fill missing categorical with Unknown
cat_cols = ["B_Ion", "A_Ion", "C_Ion", "HTL", "ETL", "Backcontact",
            "Solvents", "AntiSolvent", "Synth_Atm", "Dep_Procedure"]
for col in cat_cols:
    df[col] = df[col].fillna("Unknown")

# Fill missing numerical with median
num_cols = ["Bandgap", "Cell_Area", "Anneal_Temp"]
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

print(f"Clean rows: {len(df)}")
print(df.dtypes)
# Label encode all categorical columns
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

# Features and target
FEATURE_COLS = ["B_Ion", "A_Ion", "C_Ion", "Bandgap", "Cell_Area",
                "HTL", "ETL", "Backcontact", "Solvents", "AntiSolvent",
                "Synth_Atm", "Dep_Procedure", "Anneal_Temp", "Is_Module", "Solv_Anneal"]

X = df[FEATURE_COLS].values
y = df["PCE"].values
print(f"X: {X.shape}  y mean: {y.mean():.2f}  std: {y.std():.2f}")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print(f"Train: {len(X_train)}  Test: {len(X_test)}")

rf = RandomForestRegressor(n_estimators=300, max_depth=15,
                           min_samples_leaf=5, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
print("Training done!")

y_pred = rf.predict(X_test)
print(f"R²   : {r2_score(y_test, y_pred):.4f}")
print(f"MAE  : {mean_absolute_error(y_test, y_pred):.4f}")
print(f"RMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
import matplotlib.pyplot as plt

# 1. Feature Importance
feat_imp = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)
plt.figure(figsize=(10, 7))
feat_imp.plot(kind="barh", color="teal", alpha=0.85)
plt.title("Feature Importance — Random Forest (PCE Prediction)", fontsize=14, fontweight="bold")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()
print("Most important feature:", feat_imp.idxmax())

# 2. Actual vs Predicted
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred, alpha=0.3, s=10, color="#1565C0")
plt.plot([0, 28], [0, 28], "r--", linewidth=2)
plt.xlabel("Actual PCE (%)", fontsize=12)
plt.ylabel("Predicted PCE (%)", fontsize=12)
plt.title(f"Actual vs Predicted PCE  (R² = {r2_score(y_test, y_pred):.4f})", fontweight="bold")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.show()

# 3. PCE Distribution
plt.figure(figsize=(8, 5))
plt.hist(y, bins=60, color="#E65100", edgecolor="white", alpha=0.85)
plt.axvline(y.mean(), color="black", linestyle="--", linewidth=2, label=f"Mean = {y.mean():.2f}%")
plt.xlabel("PCE (%)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.title("PCE Distribution in Dataset", fontweight="bold")
plt.legend()
plt.tight_layout()
plt.savefig("pce_distribution.png", dpi=150)
plt.show()

# 4. PCE by B_Ion (most common perovskite metal)
plt.figure(figsize=(8, 5))
df.groupby("B_Ion")["PCE"].mean().sort_values().plot(kind="barh", color="#4CAF50", alpha=0.85)
plt.xlabel("Mean PCE (%)")
plt.title("Average PCE by B-Ion", fontweight="bold")
plt.tight_layout()
plt.savefig("pce_by_bion.png", dpi=150)
plt.show()

# 5. PCE by A_Ion
plt.figure(figsize=(8, 5))
df.groupby("A_Ion")["PCE"].mean().sort_values().plot(kind="barh", color="#9C27B0", alpha=0.85)
plt.xlabel("Mean PCE (%)")
plt.title("Average PCE by A-Ion", fontweight="bold")
plt.tight_layout()
plt.savefig("pce_by_aion.png", dpi=150)
plt.show()

print("\n=== PPT SUMMARY ===")
print(f"Total clean samples   : {len(df)}")
print(f"R² Score              : {r2_score(y_test, y_pred):.4f}")
print(f"MAE                   : {mean_absolute_error(y_test, y_pred):.4f} %")
print(f"RMSE                  : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f} %")
print(f"Most important feature: {feat_imp.idxmax()}")
print(f"Mean PCE in dataset   : {y.mean():.2f}%")
print(f"Best B_Ion            : {df.groupby('B_Ion')['PCE'].mean().idxmax()}")
print(f"Best A_Ion            : {df.groupby('A_Ion')['PCE'].mean().idxmax()}")

joblib.dump(rf, "model.joblib")
joblib.dump(le_dict, "label_encoders.joblib")
joblib.dump(FEATURE_COLS, "feature_cols.joblib")
print("Model saved!")
