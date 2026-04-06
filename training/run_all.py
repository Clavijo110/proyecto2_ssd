"""Ejecuta prepare + los 6 entrenamientos en orden (requiere red para ucimlrepo si no hay CSV)."""
import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "prepare.py",
    "train_dt.py",
    "train_knn.py",
    "train_gbm.py",
    "train_lr.py",
    "train_xgb.py",
    "train_tabpfn.py",
]


def main():
    root = Path(__file__).resolve().parent
    for name in SCRIPTS:
        path = root / name
        print(f"\n=== {name} ===")
        subprocess.check_call([sys.executable, str(path)], cwd=str(root))
    print("\nListo. Artefactos en ../ai-service/models/")


if __name__ == "__main__":
    main()
