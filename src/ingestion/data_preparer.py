"""Module de préparation des données JSON prêtes pour l'indexation RAG."""

import os
import re
import json
from typing import List, Dict, Any


def extract_chunks_from_txt(file_path: str) -> List[Dict[str, str]]:
    """Extrait les chunks structurés depuis un fichier texte de sections."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = re.compile(r"\[SECTION\](.*?)\n(.*?)(?=\n\[SECTION\]|$)", re.DOTALL)
    matches = pattern.findall(content)

    chunks: List[Dict[str, str]] = []
    for title, body in matches:
        title_clean = title.strip()
        body_clean = body.strip()
        if body_clean:
            chunks.append({
                "section": title_clean,
                "text": body_clean
            })

    return chunks


def save_chunk_metadata(chunks: List[Dict[str, str]], path: str) -> None:
    """Enregistre les chunks au format JSON structuré."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def process_txt_file(file_path: str, output_dir: str) -> bool:
    """Traite un fichier texte unique et génère son équivalent JSON."""
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    chunks = extract_chunks_from_txt(file_path)
    if not chunks:
        return False

    out_json = os.path.join(output_dir, f"{base_name}.json")
    save_chunk_metadata(chunks, out_json)
    return True


def prepare_json_chunks(input_dir: str, output_dir: str) -> int:
    """Traite un répertoire complet de fichiers .txt pour générer les JSON d'indexation."""
    if not os.path.isdir(input_dir):
        print(f"Dossier source introuvable : {input_dir}")
        return 0

    os.makedirs(output_dir, exist_ok=True)
    processed = 0

    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_dir, filename)
            if process_txt_file(file_path, output_dir):
                processed += 1

    print(f"Preparation JSON terminee : {processed} documents indexes dans {output_dir}")
    return processed
