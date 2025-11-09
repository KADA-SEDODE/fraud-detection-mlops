import joblib
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from src.logging_utils import get_logger
from src.utils.config import MODELS_DIR

logger = get_logger(__name__)


#  Chargement du modèle et du scaler

def load_model_and_scaler(model_name: str = "xgbclassifier_fraud"):
    """
    Charge le modèle et le scaler sauvegardés après l'entraînement.
    """
    model_path = MODELS_DIR / f"{model_name}.joblib"
    scaler_path = MODELS_DIR / "scaler_time_amount.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Modèle non trouvé : {model_path}")

    logger.info(f"Chargement du modèle depuis {model_path}")
    model = joblib.load(model_path)

    scaler = None
    if scaler_path.exists():
        logger.info(f"Chargement du scaler depuis {scaler_path}")
        scaler = joblib.load(scaler_path)
    else:
        logger.warning("Aucun scaler trouvé — les données ne seront pas standardisées.")

    return model, scaler



#  Prétraitement des nouvelles données

def preprocess_input(df: pd.DataFrame, scaler: StandardScaler | None):
    """
    Applique le même prétraitement que pendant l'entraînement :
    - Standardisation des colonnes 'Time' et 'Amount'
    - Réordonne les colonnes dans le même ordre que le modèle
    """
    expected_cols = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")

    df_processed = df[expected_cols].copy()

    if scaler is not None:
        cols_to_scale = ["Time", "Amount"]
        df_processed[cols_to_scale] = scaler.transform(df_processed[cols_to_scale])

    return df_processed



#  Prédiction

def predict_fraud(df: pd.DataFrame, model, scaler=None, threshold: float = 0.5):
    """
    Retourne les probabilités et prédictions binaires de fraude.
    Force les colonnes à être dans le même ordre que lors de l'entraînement.
    """
    ordered_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


    for col in ordered_cols:
        if col not in df.columns:
            raise ValueError(f"Colonne manquante : {col}")

    df = df[ordered_cols].copy()

    if scaler is not None:
        df[["Time", "Amount"]] = scaler.transform(df[["Time", "Amount"]])

    proba = model.predict_proba(df)[:, 1]
    preds = (proba >= threshold).astype(int)

    results = pd.DataFrame({
        "fraud_probability": proba,
        "fraud_prediction": preds
    })

    logger.info(f"Prédictions terminées pour {len(df)} transactions.")
    return results



#  Point d’entrée principal

if __name__ == "__main__":
    new_data_path = Path("data/new_transactions.csv")
    if not new_data_path.exists():
        raise FileNotFoundError(f"Fichier non trouvé : {new_data_path}")

    df_new = pd.read_csv(new_data_path)
    logger.info(f"Nouvelles transactions chargées : {df_new.shape[0]} lignes")

    model, scaler = load_model_and_scaler("xgbclassifier_fraud")

    results = predict_fraud(df_new, model, scaler, threshold=0.5)
    print(results)

    output_path = Path("outputs/predictions.csv")
    output_path.parent.mkdir(exist_ok=True)
    results.to_csv(output_path, index=False)
    logger.info(f"Résultats sauvegardés dans {output_path}")
