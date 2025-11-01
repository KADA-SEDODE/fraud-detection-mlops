# ==========================================================
# 📦 IMPORTS GLOBAUX POUR LE PROJET FRAUDFLOW-MLOPS
# ==========================================================

# --- Système & gestion des fichiers ---
import os
import sys
import json
import time
import shutil
import logging
import joblib
import warnings
from pathlib import Path

# --- Manipulation et analyse des données ---
import pandas as pd
import numpy as np

# --- Visualisation & EDA ---
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# --- Prétraitement ---
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline

# --- Modélisation (classiques + avancés) ---
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# --- Déséquilibre de classes (imblearn) ---
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import EasyEnsembleClassifier, BalancedBaggingClassifier

# --- Évaluation & interprétation ---
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score,
    roc_curve, roc_auc_score,
    accuracy_score, f1_score, precision_score, recall_score
)

# --- Explicabilité ---
import shap

# --- Sauvegarde et traçabilité ---
import mlflow
import mlflow.sklearn

# --- API & déploiement ---
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# # --- Orchestration & automatisation ---
# # (optionnel, pour Airflow)
# try:
#     from airflow import DAG
#     from airflow.operators.python import PythonOperator
# except ImportError:
#     pass  # pas nécessaire si tu n’exécutes pas encore Airflow en local

# --- Divers / Utilitaires ---
import datetime as dt
from tqdm import tqdm
from pprint import pprint

# --- Config d'affichage pandas & warnings ---
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')



# --- Confirmation ---
print(" Toutes les librairies principales importées avec succès (Data | ML | MLOps | API | Orchestration).")
