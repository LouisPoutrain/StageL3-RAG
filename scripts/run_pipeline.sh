#!/bin/bash
# Pipeline de bout en bout : Prétraitement TEI -> Découpage Chunks -> Inférence RAG
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT_DIR"

echo "=========================================================="
echo "Lancement du pipeline complet StageL3-RAG"
echo "=========================================================="

# Utiliser le venv s'il existe
PYTHON="python3"
if [ -f "$ROOT_DIR/.venv/bin/python" ]; then
  PYTHON="$ROOT_DIR/.venv/bin/python"
fi

# 1. Vérification des dossiers
mkdir -p data/chunks/standard data/chunks/invasive_detection data/input data/output

# 2. Exécution du RAG
echo "[Etape] Execution du pipeline RAG..."
"$ROOT_DIR/run_rag.sh" "$@"

echo "=========================================================="
echo "Pipeline termine avec succes."
echo "Resultats disponibles dans data/output/Protocoles.csv"
echo "=========================================================="
