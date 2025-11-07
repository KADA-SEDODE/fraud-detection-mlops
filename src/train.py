import yaml
import joblib
from pathlib import Path
import mlflow
import mlflow.sklearn

from src.logging_utils import get_logger
from src.data import load_creditcard_data
from src.utils.config import CREDITCARD_PATH, MODELS_DIR, MLFLOW_DIR
from src.utils.preprocessing import train_test_split_creditcard
from src.metrics import evaluate_model

logger = get_logger(__name__)



#  Chargement de la configuration YAML

def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)



#  Entraînement générique de modèles ML

def train_model(model_name: str, params: dict, X_train, y_train):
    logger.info(f"Initialisation du modèle {model_name} avec paramètres : {params}")

    if model_name == "RandomForestClassifier":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(**params)
    elif model_name == "XGBClassifier":
        from xgboost import XGBClassifier
        model = XGBClassifier(**params)
    else:
        raise ValueError(f"Modèle non reconnu : {model_name}")

    model.fit(X_train, y_train)
    logger.info("Entraînement terminé")
    return model



#  Sauvegarde du modèle et du scaler

def save_artifacts(model, scaler, model_name: str):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{model_name.lower()}_fraud.joblib"
    joblib.dump(model, model_path)
    logger.info(f"Modèle sauvegardé dans {model_path}")

    scaler_path = None
    if scaler is not None:
        scaler_path = MODELS_DIR / "scaler_time_amount.joblib"
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler sauvegardé dans {scaler_path}")

    return model_path, scaler_path



#  Main pipeline avec MLflow

def main():
    logger.info("=== DÉBUT DU PIPELINE D'ENTRAÎNEMENT FRAUDFLOW ===")

    #  Chargement de la config
    config_path = Path("config/train_config.yaml")
    config = load_config(config_path)
    model_name = config["model"]["name"]
    model_params = config["model"]["params"]
    train_params = config["training"]
    eval_params = config["evaluation"]

    #  Chargement du dataset
    df = load_creditcard_data(CREDITCARD_PATH)

    #  Split et scaling
    X_train, X_test, y_train, y_test, scaler = train_test_split_creditcard(
        df,
        test_size=train_params["test_size"],
        random_state=train_params["random_state"],
        scale_time_amount=train_params["scale_time_amount"],
    )

    #  Ajustement dynamique du poids de classe
    if model_name == "XGBClassifier" and "scale_pos_weight" in model_params:
        ratio = int((y_train == 0).sum() / (y_train == 1).sum())
        model_params["scale_pos_weight"] = ratio
        logger.info(f"scale_pos_weight ajusté automatiquement à {ratio}")

    #  Configuration MLflow
    mlflow.set_tracking_uri(MLFLOW_DIR.as_uri())
    mlflow.set_experiment("fraudflow_experiment")

    with mlflow.start_run(run_name=f"{model_name}_training"):
        mlflow.log_params(model_params)
        mlflow.log_params(train_params)
        mlflow.log_params(eval_params)

        #  Entraînement
        model = train_model(model_name, model_params, X_train, y_train)

        #  Évaluation (seuil fixé à 0.5)
        metrics = evaluate_model(
            model, X_test, y_test,
            threshold=0.5,
            plot=True,
            save_dir="models/plots"
        )

        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})

        #  Sauvegarde du modèle et artefacts
        model_path, scaler_path = save_artifacts(model, scaler, model_name)
        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_artifact("config/train_config.yaml")
        mlflow.log_artifacts("models/plots")

        if scaler_path:
            mlflow.log_artifact(str(scaler_path))

        logger.info("Run MLflow complété et artefacts sauvegardés")

    logger.info("=== FIN DU PIPELINE D'ENTRAÎNEMENT FRAUDFLOW ===")


if __name__ == "__main__":
    main()
