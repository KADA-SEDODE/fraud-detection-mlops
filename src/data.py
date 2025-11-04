import  pandas as  pd
from pathlib import Path
from .logging_utils import get_logger

logger = get_logger(__name__)

def load_creditcard_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    logger.info(f"chargement des données depuis {path}")
    df = pd.read_csv(path)
    logger.info(f"Dataset chargé :{df.shape[0]} lignes, {df.shape[1]} colonnes.")
    return df