"""
Utilidades comunes para el proyecto de Clasificación de Violencia de Género - WomenHelp-MX.
Autor: Proyecto NLP - UNSAAC
"""
import re
import random
import numpy as np
import pandas as pd

SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)

DATA_DIR = "/home/claude/proyecto/data"

LABEL_NAMES_MULTICLASS = {0: "Mild", 1: "Medium", 2: "High", 3: "Severe"}

# Mapeo INFERIDO y validado por evidencia textual en la Fase 1 (EDA).
# L6 es puro en el 100% de los casos (167/167), consistente con "N/A".
LABEL_NAMES_MULTILABEL = {
    "L0": "Economic",
    "L1": "Physical",
    "L2": "Property-related",
    "L3": "Psychological",
    "L4": "Sexual",
    "L5": "Vicarious",
    "L6": "N/A",
}
MULTILABEL_COLS = list(LABEL_NAMES_MULTILABEL.keys())


def load_multiclass():
    train = pd.read_csv(f"{DATA_DIR}/Multiclass/train.csv")
    devel = pd.read_csv(f"{DATA_DIR}/Multiclass/devel.csv")
    test = pd.read_csv(f"{DATA_DIR}/Multiclass/test.csv")
    for df in (train, devel, test):
        df.columns = [c.upper() for c in df.columns]
        df.rename(columns={"TEXT": "TEXT", "CLASS": "CLASS"}, inplace=True)
    return train, devel, test


def load_multilabel():
    train = pd.read_csv(f"{DATA_DIR}/Multilabel/train.csv")
    devel = pd.read_csv(f"{DATA_DIR}/Multilabel/devel.csv")
    test = pd.read_csv(f"{DATA_DIR}/Multilabel/test.csv")
    return train, devel, test


def clean_text_traditional(text: str) -> str:
    """Limpieza para modelos tradicionales (TF-IDF).
    Decisiones documentadas en el informe (Sección Metodología):
    - minúsculas: reduce dispersión del vocabulario sin perder semántica.
    - normalización de espacios/saltos de línea.
    - se preservan tildes y ñ (relevante en español).
    - se preservan signos de exclamación/interrogación como señal de intensidad emocional.
    - URLs y menciones (no se encontraron en el corpus, pero se incluye por robustez).
    - NO se eliminan stopwords por defecto: se evalúa como hiperparámetro (ver GridSearch).
    """
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^a-záéíóúñü0-9¿?¡!.,;:\-\"'\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def clean_text_transformer(text: str) -> str:
    """Limpieza mínima para Transformers (BETO).
    Solo se normalizan espacios; se preserva mayúsculas/minúsculas, puntuación y
    tildes ya que el tokenizer de BETO (WordPiece) maneja subpalabras y el
    modelo fue preentrenado con texto crudo en español.
    """
    if not isinstance(text, str):
        return ""
    t = re.sub(r"\s+", " ", text).strip()
    return t
