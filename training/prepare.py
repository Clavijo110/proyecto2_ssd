"""Genera preprocessor, manifest de comparación y data_bundle (ejecutar una vez antes de entrenar)."""
from common import prepare_train_test

if __name__ == "__main__":
    prepare_train_test()
    print("Listo: ai-service/models/preprocessor.pkl, data_bundle.joblib, compare_manifest.json")
