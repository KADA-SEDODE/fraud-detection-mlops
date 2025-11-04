import  logging
import sys

def get_logger(name : str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger # pour éviter d'ajouter plusieurs handlers
    
    logger.setLevel(logging.INFO)  # Définit le niveau minimal des logs à afficher (INFO ou plus grave).
    
    handler = logging.StreamHandler(sys.stdout) # Crée un handler qui affiche les logs dans la console.
    fmt = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    
    logger.addHandler(handler)
    return logger   
