"""Configuration centralisée et gestion des chemins pour le projet StageL3-RAG."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Chargement du fichier .env s'il existe
load_dotenv()

# Racine du projet (2 niveaux au-dessus de src/common/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Répertoires de données par défaut
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
CHUNKS_DIR = DATA_DIR / "chunks"
BENCHMARKS_DIR = DATA_DIR / "benchmarks"
PAPERS_DIR = DATA_DIR / "papers"

# Paramètres API LLM
DEFAULT_API_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "mistral-nemo:latest"

def get_api_url() -> str:
    """Retourne l'URL de l'API LLM configurée."""
    return os.getenv("LLM_API_URL", DEFAULT_API_URL)

def get_api_key() -> str:
    """Retourne la clé d'API LLM configurée."""
    return os.getenv("LLM_API_KEY", "")

def get_model_name() -> str:
    """Retourne le nom du modèle LLM configuré."""
    return os.getenv("LLM_MODEL", DEFAULT_MODEL)
