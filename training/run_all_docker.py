"""Pipeline entrenamiento para Docker: prepare + 5 modelos + stub TabPFN (sin torch)."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    "prepare.py",
    "train_dt.py",
    "train_knn.py",
    "train_gbm.py",
    "train_lr.py",
    "train_xgb.py",
    "train_tabpfn_stub.py",
]


def main():
    for name in SCRIPTS:
        path = ROOT / name
        print(f"\n=== {name} ===", flush=True)
        subprocess.check_call([sys.executable, str(path)], cwd=str(ROOT))
    print("\nArtefactos listos en MODELS_DIR.", flush=True)


if __name__ == "__main__":
    main()
