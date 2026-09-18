"""Module de division des chunks globaux en fichiers unitaires par document."""

import os
import re
from typing import List, Tuple


def extract_chunks_from_txt(content: str) -> List[Tuple[str, str]]:
    """Extrait les couples (titre_section, contenu) depuis une chaîne de sections."""
    sections = re.split(r"\[SECTION\]", content)
    chunks: List[Tuple[str, str]] = []

    for section in sections:
        section = section.strip()
        if not section:
            continue
        lines = section.splitlines()
        if lines:
            title = lines[0].strip()
            text = "\n".join(line.strip() for line in lines[1:] if line.strip())
            if text:
                chunks.append((title, text))
    return chunks


def save_chunks_to_txt_file(chunks: List[Tuple[str, str]], output_path: str) -> None:
    """Écrit les chunks formatés dans un fichier texte."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for title, text in chunks:
            f.write(f"[SECTION] {title}\n{text}\n\n")


def divide_chunks(input_txt_path: str, output_dir: str) -> int:
    """Découpe un fichier concaténé par bloc TEI et génère un fichier .txt par document."""
    if not os.path.isfile(input_txt_path):
        print(f"Fichier introuvable : {input_txt_path}")
        return 0

    os.makedirs(output_dir, exist_ok=True)
    with open(input_txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    file_blocks = re.split(r"={5,}\s*FILE: (.+?)\.grobid\.tei\.xml\s*={5,}", content)
    count = 0

    for i in range(1, len(file_blocks), 2):
        filename = file_blocks[i].strip()
        file_content = file_blocks[i + 1].strip()

        chunks = extract_chunks_from_txt(file_content)
        output_filename = os.path.join(output_dir, f"{filename}.txt")
        save_chunks_to_txt_file(chunks, output_filename)
        count += 1

    print(f"Decoupage termine : {count} fichiers generes dans {output_dir}")
    return count
