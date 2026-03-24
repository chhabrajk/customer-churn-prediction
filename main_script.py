# ============================================================
# Customer Churn Prediction — End-to-End ML Pipeline
# Author: Jagdish Chhabra | github.com/jagdish-chhabra
# Stack: Python, Pandas, Scikit-learn, XGBoost, Matplotlib
# Business Goal: Predict which customers will churn next month
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, ConfusionMatrixDisplay)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
print("=" * 60)
print("  CUSTOMER CHURN PREDICTION PIPELINE")
print("  Author: Jagdish Chhabra")
print("=" * 60)

os.makedirs("screenshots", exist_ok=True)
os.makedirs("output", exist_ok=True)

# ── 1. GENERATE DATASET ──────────────────────────────────────
n = 5000

df = pd.DataFrame({
    "customer_id"        : range(1, n + 1),
    "tenure_months"      : np.random.randint(1, 72, n),
    "monthly_charges"    : np.round(np.random.uniform(20, 120, n), 2),
    "total_charges"      : None,                                      # derived
    "num_products"       : np.random.randint(1, 6, n),
    "support_calls"      : np.random.poisson(2, n),
    "payment_delay_days" : np.random.exponential(3, n).astype(int),
    "contract_type"      : np.random.choice(
                               ["Month-to-Month", "One Year", "Two Year"],
                               n, p=[0.55, 0.25, 0.20]),
    "internet_service"   : np.random.choice(["Fiber", "DSL", "None"], n, p=[0.45, 0.40, 0.15]),
    "has_tech_support"   : np.random.choice([0, 1], n),
    "is_senior_citizen"  : np.random.choice([0, 1], n, p=[0.84, 0.16]),
    "gender"             : np.random.choice(["Male", "Female"], n),
})

df["total_charges"] = (df["tenure_months"] * df["monthly_charges"]
                       * np.random.uniform(0.85, 1.05, n)).round(2)

# Engineered churn signal (realistic rules)
churn_prob = (
    0.35 * (df["contract_type"] == "Month-to-Month").astype(int)
    + 0.20 * (df["tenure_months"] < 12).astype(int)
    + 0.15 * (df["support_calls"] > 3).astype(int)
    + 0.10 * (df["payment_delay_days"] > 7).astype(int)
    - 0.10 * df["has_tech_support"]
    - 0.05 * (df["num_products"] > 3).astype(int)
    + np.random.uniform(0, 0.2, n)
)
df["churn"] = (churn_prob > 0.45).astype(int)

print(f"\n📊 Dataset: {len(df):,} customers")
print(f"   Churn Rate: {df['churn'].mean():.1%}")
print(f"   Features  : {df.shape[1] - 2} predictors\n")

# ── 2. PREPROCESSING ─────────────────────────────────────────
le = LabelEncoder()
df["contract_enc"]  = le.fit_transform(df["contract_type"])
df["internet_enc"]  = le.fit_transform(df["internet_service"])
df["gender_enc"]    = le.fit_transform(df["gender"])

FEATURES = ["tenure_months", "monthly_charges", "total_charges",
            "num_products", "support_calls", "payment_delay_days",
            "has_tech_support", "is_senior_citizen",
            "contract_enc", "internet_enc", "gender_enc"]

X = df[FEATURES]
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

# ── 3. MODEL TRAINING ────────────────────────────────────────
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, random_state=42))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=150, max_depth=8, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42),
}

results = {}
print("\n🔄 Training models...\n")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)
    cv_auc  = cross_val_score(model, X_train, y_train,
                               cv=5, scoring="roc_auc").mean()
    results[name] = {"model": model, "preds": y_pred, "proba": y_proba,
                     "auc": auc, "cv_auc": cv_auc}
    print(f"  {name:<25} AUC={auc:.4f}  CV-AUC={cv_auc:.4f}")

best_name = max(results, key=lambda k: results[k]["auc"])
best      = results[best_name]
print(f"\n✅ Best Model: {best_name} (AUC = {best['auc']:.4f})")

# ── 4. DETAILED REPORT ───────────────────────────────────────
print(f"\n📋 Classification Report — {best_name}")
print(classification_report(y_test, best["preds"],
                             target_names=["Retained", "Churned"]))

# ── 5. VISUALIZATIONS ────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.suptitle("Customer Churn Prediction Dashboard\nJagdish Chhabra — Analytics Portfolio",
             fontsize=14, fontweight="bold", y=1.01)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# 5a — ROC Curves (all models)
ax1 = fig.add_subplot(gs[0, 0])
colors = ["#4361ee", "#3f8600", "#d62828"]
for (name, res), col in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, res["proba"])
    ax1.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})", color=col, lw=2)
ax1.plot([0, 1], [0, 1], "k--", lw=1)
ax1.set(title="ROC Curves — All Models", xlabel="False Positive Rate",
        ylabel="True Positive Rate")
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

# 5b — Confusion Matrix
ax2 = fig.add_subplot(gs[0, 1])
ConfusionMatrixDisplay(
    confusion_matrix(y_test, best["preds"]),
    display_labels=["Retained", "Churned"]
).plot(ax=ax2, colorbar=False, cmap="Blues")
ax2.set_title(f"Confusion Matrix — {best_name}")

# 5c — Feature Importance
ax3 = fig.add_subplot(gs[0, 2])
if hasattr(results[best_name]["model"], "feature_importances_"):
    importances = results[best_name]["model"].feature_importances_
else:
    importances = abs(results[best_name]["model"].named_steps["clf"].coef_[0])
feat_imp = pd.Series(importances, index=FEATURES).sort_values()
feat_imp.plot(kind="barh", ax=ax3, color="#4361ee", edgecolor="white")
ax3.set_title("Feature Importances")
ax3.set_xlabel("Importance Score")
ax3.grid(axis="x", alpha=0.3)

# 5d — Churn distribution
ax4 = fig.add_subplot(gs[1, 0])
churn_counts = df["churn"].value_counts()
ax4.bar(["Retained", "Churned"], churn_counts.values,
        color=["#3f8600", "#d62828"], edgecolor="white", width=0.5)
ax4.set_title("Class Distribution")
ax4.set_ylabel("Customers")
for i, v in enumerate(churn_counts.values):
    ax4.text(i, v + 30, f"{v:,}", ha="center", fontsize=10, fontweight="bold")
ax4.grid(axis="y", alpha=0.3)

# 5e — Churn by Contract Type
ax5 = fig.add_subplot(gs[1, 1])
churn_contract = df.groupby("contract_type")["churn"].mean().sort_values(ascending=False)
churn_contract.plot(kind="bar", ax=ax5, color=["#d62828", "#f4a261", "#3f8600"],
                    edgecolor="white", rot=15)
ax5.set_title("Churn Rate by Contract Type")
ax5.set_ylabel("Churn Rate")
ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax5.grid(axis="y", alpha=0.3)

# 5f — Tenure vs Churn
ax6 = fig.add_subplot(gs[1, 2])
df["tenure_bucket"] = pd.cut(df["tenure_months"],
                               bins=[0, 12, 24, 36, 48, 72],
                               labels=["0-12", "13-24", "25-36", "37-48", "49-72"])
churn_tenure = df.groupby("tenure_bucket")["churn"].mean()
churn_tenure.plot(kind="line", ax=ax6, marker="o", color="#7209b7", lw=2.5, ms=8)
ax6.set_title("Churn Rate by Tenure (Months)")
ax6.set_xlabel("Tenure Bucket")
ax6.set_ylabel("Churn Rate")
ax6.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax6.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("screenshots/churn_dashboard.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n💾 Saved: screenshots/churn_dashboard.png")

# ── 6. BUSINESS INSIGHT SUMMARY ──────────────────────────────
retained = df[df["churn"] == 0]["monthly_charges"].mean()
churned  = df[df["churn"] == 1]["monthly_charges"].mean()
top_feat = pd.Series(importances, index=FEATURES).sort_values(ascending=False)

print("\n" + "=" * 60)
print("  BUSINESS INSIGHTS")
print("=" * 60)
print(f"  🔴 Avg monthly charge — Churned  : ${churned:.2f}")
print(f"  🟢 Avg monthly charge — Retained : ${retained:.2f}")
print(f"  📌 Top churn driver: {top_feat.index[0].replace('_', ' ').title()}")
print(f"  📌 2nd driver      : {top_feat.index[1].replace('_', ' ').title()}")
print(f"\n  💡 Recommendation:")
print(f"     Customers on Month-to-Month contracts with < 12 months")
print(f"     tenure are {df[(df['contract_type']=='Month-to-Month') & (df['tenure_months']<12)]['churn'].mean():.0%} likely to churn.")
print(f"     Priority retention campaigns should target this segment.")
print("=" * 60)

# ── HOW TO RUN ───────────────────────────────────────────────
# pip install pandas numpy matplotlib scikit-learn
# python churn_prediction.py