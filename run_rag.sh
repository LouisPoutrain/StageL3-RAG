#!/bin/bash
# Script de lancement de production du pipeline RAG
# Usage: ./run_rag.sh [--api_url URL] [--json_file FILE] [autres options]
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Détection de l'interpréteur Python
if [ -f "$PROJ_DIR/.venv/bin/python" ]; then
    PYTHON="$PROJ_DIR/.venv/bin/python"
elif [ -f "$PROJ_DIR/venv/bin/python" ]; then
    PYTHON="$PROJ_DIR/venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
else
    echo "Erreur: Aucun interpreteur Python disponible."
    exit 1
fi

echo "=========================================================="
echo "Demarrage du pipeline StageL3-RAG"
echo "Repertoire racine : $PROJ_DIR"
echo "Interpreteur      : $PYTHON"
echo "=========================================================="

# Paramètres par défaut
API_URL="${API_URL:-http://localhost:11434/api/chat}"
INPUT_DIR="${INPUT_DIR:-$PROJ_DIR/data/input}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJ_DIR/data/output}"

# Exécuter le script
"$PYTHON" "$PROJ_DIR/src/rag/mainRag.py" \
    --api_url "$API_URL" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    "$@"
