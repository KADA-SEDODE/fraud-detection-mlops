from pathlib import Path

# --- Racine du projet ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- Dossiers principaux ---
DATA_DIR = PROJECT_ROOT / "data"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
MODELS_DIR = PROJECT_ROOT / "models"
MLFLOW_DIR = PROJECT_ROOT / "mlruns"

# --- Fichiers spécifiques ---
CREDITCARD_PATH = DATA_DIR / "creditcard.csv"
