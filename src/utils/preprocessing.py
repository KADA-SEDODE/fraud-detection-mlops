import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.logging_utils import get_logger

logger = get_logger(__name__)

FEATURE_COLS = None  
TARGET_COL = "Class"

def train_test_split_creditcard(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 90,
    scale_time_amount: bool = True,
):
    logger.info("Préparation des features et de la cible")

    if FEATURE_COLS is None:
        feature_cols = [c for c in df.columns if c != TARGET_COL]
    else:
        feature_cols = FEATURE_COLS

    X = df[feature_cols].copy()
    y = df[TARGET_COL].astype(int)

    if scale_time_amount:
        logger.info("Standardisation des colonnes Time et Amount")
        cols_to_scale = [c for c in ["Time", "Amount"] if c in X.columns]
        scaler = StandardScaler()
        X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])
    else:
        scaler = None

    logger.info("Découpage train/test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, scaler


if __name__ =="__main__":
    from src.data import load_creditcard_data  
    from src.utils.config import CREDITCARD_PATH
    
    df = load_creditcard_data(CREDITCARD_PATH)
    
    X_train, X_test, y_train, y_test, scaler = train_test_split_creditcard(df)
    
    print("Taille du train :", X_train.shape, y_train.shape)
    print("Taille du test  :", X_test.shape, y_test.shape)
    print("Scaler utilisé :", type(scaler).__name__)
    
