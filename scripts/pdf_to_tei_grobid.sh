#!/bin/bash
# Conversion de documents PDF en XML TEI via serveur GROBID Docker
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

PDF_DIR="${1:-$ROOT_DIR/data/papers}"
OUT_DIR="${2:-$ROOT_DIR/data/tei_xml}"
GROBID_URL="${GROBID_URL:-http://localhost:8070}"
GROBID_IMAGE="${GROBID_IMAGE:-lfoppiano/grobid:0.8.0}"
GROBID_CONTAINER="${GROBID_CONTAINER:-grobid-server}"

if ! command -v docker >/dev/null 2>&1; then
  echo "Erreur: docker est requis mais introuvable dans le PATH."
  exit 1
fi

if [ ! -d "$PDF_DIR" ]; then
  echo "Erreur: dossier PDF introuvable : $PDF_DIR"
  exit 1
fi

mkdir -p "$OUT_DIR"

if ! docker ps --format '{{.Names}}' | grep -qx "$GROBID_CONTAINER"; then
  if docker ps -a --format '{{.Names}}' | grep -qx "$GROBID_CONTAINER"; then
    docker start "$GROBID_CONTAINER" >/dev/null
  else
    docker run -d --name "$GROBID_CONTAINER" -p 8070:8070 "$GROBID_IMAGE" >/dev/null
  fi
fi

echo "Attente de reponse du serveur GROBID sur $GROBID_URL ..."
for _ in $(seq 1 60); do
  if curl -fsS "$GROBID_URL/api/isalive" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

if ! curl -fsS "$GROBID_URL/api/isalive" >/dev/null 2>&1; then
  echo "Erreur: GROBID ne repond pas sur $GROBID_URL"
  exit 1
fi

shopt -s nullglob
pdf_files=("$PDF_DIR"/*.pdf)

if [ ${#pdf_files[@]} -eq 0 ]; then
  echo "Aucun fichier PDF trouve dans $PDF_DIR"
  exit 0
fi

for pdf in "${pdf_files[@]}"; do
  base_name="$(basename "$pdf" .pdf)"
  out_file="$OUT_DIR/$base_name.grobid.tei.xml"
  echo "Conversion : $(basename "$pdf") -> $(basename "$out_file")"

  curl -fsS "$GROBID_URL/api/processFulltextDocument" \
    -F "input=@$pdf" \
    -F "consolidateHeader=1" \
    -F "consolidateCitations=1" \
    -F "includeRawCitations=1" \
    -F "segmentSentences=1" \
    > "$out_file"
done

echo "Conversion GROBID terminee avec succes. Fichiers dans : $OUT_DIR"
