"""
setup.py — First-run bootstrap script
======================================
Generates the synthetic dataset and trains the ML model so the
Streamlit app is ready to launch immediately.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils.data_generator import generate_dataset
from utils.train_model import train

if __name__ == "__main__":
    print("=" * 55)
    print("  AMR-Predictor — First-run Setup")
    print("=" * 55)

    print("\n[1/2] Generating synthetic dataset …")
    df = generate_dataset(n=2000, save_path=str(ROOT / "data" / "amr_synthetic_dataset.csv"))
    print(f"      ✓ {len(df):,} isolates | MDR rate: {df['is_mdr'].mean():.1%}")

    print("\n[2/2] Training Random Forest model …")
    model, metrics = train(use_antibiogram=True)
    print(f"      ✓ ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"      ✓ CV AUC: {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")

    print("\n" + "=" * 55)
    print("  Setup complete!  Run: streamlit run app.py")
    print("=" * 55)
