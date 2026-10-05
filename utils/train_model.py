"""
train_model.py  (One Health Edition)
--------------------------------------
Trains a Random Forest classifier to predict MDR phenotype from
One Health isolate data (human + animal + environment).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (classification_report, roc_auc_score,
                              confusion_matrix, roc_curve)
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.data_generator import generate_dataset, ALL_ANTIBIOTICS

DATA_PATH  = ROOT / "data" / "amr_synthetic_dataset.csv"
MODELS_DIR = ROOT / "models"
TARGET     = "is_mdr"

CATEGORICAL_FEATURES = [
    "host_species", "host_type", "species", "gram_stain", "family",
    "country", "host_sex", "zoonotic_risk",
]
NUMERIC_FEATURES = [
    "host_age", "prior_antibiotic_exposure", "intensive_care_or_farming",
    "zoonotic_score", "esbl", "carbapenemase", "mrsa", "vre", "year",
]


def load_or_generate(path: Path) -> pd.DataFrame:
    if path.exists():
        print(f"[→] Loading data from {path}")
        return pd.read_csv(path)
    print("[→] Generating One Health dataset …")
    return generate_dataset(n=3000, save_path=str(path))


def build_preprocessor(cat_cols, num_cols):
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    num_pipe = Pipeline([
        ("imp",    SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer([
        ("cat", cat_pipe, cat_cols),
        ("num", num_pipe, num_cols),
    ], remainder="drop")


def train(use_antibiogram: bool = True):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_or_generate(DATA_PATH)

    feature_cols = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    if use_antibiogram:
        feature_cols += [ab for ab in ALL_ANTIBIOTICS if ab in df.columns]

    num_features_actual = NUMERIC_FEATURES + ([ab for ab in ALL_ANTIBIOTICS if ab in df.columns]
                                               if use_antibiogram else [])

    X = df[feature_cols]
    y = df[TARGET]

    print(f"[i] Dataset: {len(df):,} | MDR rate: {y.mean():.1%} | Features: {len(feature_cols)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(CATEGORICAL_FEATURES, num_features_actual)

    pipeline = Pipeline([
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=4,
            class_weight="balanced", random_state=42, n_jobs=1,
        )),
    ])

    print("[→] Training …")
    pipeline.fit(X_train, y_train)

    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    report  = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_proba)
    cv_scores = cross_val_score(pipeline, X, y, cv=StratifiedKFold(5), scoring="roc_auc")

    print("\n── Classification Report ──────────────────────────")
    print(classification_report(y_test, y_pred, target_names=["Non-MDR","MDR"]))
    print(f"ROC-AUC : {roc_auc:.4f}")
    print(f"5-fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Feature importance
    ohe_features = (pipeline.named_steps["pre"]
                    .named_transformers_["cat"]
                    .named_steps["ohe"]
                    .get_feature_names_out(CATEGORICAL_FEATURES).tolist())
    all_feat_names = ohe_features + num_features_actual
    importances    = pipeline.named_steps["clf"].feature_importances_
    feat_imp = pd.Series(importances, index=all_feat_names).sort_values(ascending=False)
    print("\n── Top 15 Features ────────────────────────────────")
    print(feat_imp.head(15).to_string())

    fpr, tpr, _ = roc_curve(y_test, y_proba)

    metrics = {
        "classification_report": report,
        "roc_auc":               roc_auc,
        "cv_auc_mean":           cv_scores.mean(),
        "cv_auc_std":            cv_scores.std(),
        "feature_importance":    feat_imp.to_dict(),
        "feature_cols":          feature_cols,
        "confusion_matrix":      confusion_matrix(y_test, y_pred).tolist(),
        "y_test":                y_test.tolist(),
        "y_proba":               y_proba.tolist(),
        "roc_fpr":               fpr.tolist(),
        "roc_tpr":               tpr.tolist(),
    }

    joblib.dump(pipeline,     MODELS_DIR / "amr_rf_model.joblib")
    joblib.dump(feature_cols, MODELS_DIR / "feature_names.joblib")
    joblib.dump(metrics,      MODELS_DIR / "model_metrics.joblib")
    print(f"\n[✓] Artefacts saved → {MODELS_DIR}/")
    return pipeline, metrics


if __name__ == "__main__":
    train(use_antibiogram=True)
