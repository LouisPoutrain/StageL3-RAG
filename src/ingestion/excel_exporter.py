"""Module d'export de résumés d'articles vers un tableau Excel."""

import os
from typing import List, Dict
from openpyxl import Workbook


def parse_chunks(file_path: str) -> List[Dict[str, str]]:
    """Parse le fichier texte contenant les métadonnées et résumés des articles."""
    if not os.path.isfile(file_path):
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    chunks = content.strip().split("=" * 80)
    rows: List[Dict[str, str]] = []

    for chunk in chunks:
        lines = chunk.strip().splitlines()
        data = {"Fichier": "", "Titre": "", "Date": "", "Auteurs": "", "Résumé": ""}
        current_field = None

        for line in lines:
            if line.startswith("FILE:"):
                data["Fichier"] = line.replace("FILE:", "").strip()
            elif line.startswith("TITLE:"):
                data["Titre"] = line.replace("TITLE:", "").strip()
            elif line.startswith("DATE:"):
                data["Date"] = line.replace("DATE:", "").strip()
            elif line.startswith("AUTHORS:"):
                data["Auteurs"] = line.replace("AUTHORS:", "").strip()
            elif line.startswith("ABSTRACT:"):
                current_field = "Résumé"
                data["Résumé"] = ""
            elif current_field == "Résumé":
                data["Résumé"] += line.strip() + " "

        if data["Fichier"] or data["Titre"]:
            rows.append(data)
    return rows


def write_to_excel(rows: List[Dict[str, str]], output_file: str) -> None:
    """Écrit la liste des articles dans un classeur Excel."""
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.title = "Résumé articles"

    headers = ["Fichier", "Titre", "Date", "Auteurs", "Résumé"]
    ws.append(headers)

    for row in rows:
        ws.append([row.get(h, "") for h in headers])

    wb.save(output_file)
    print(f"Classeur Excel enregistre sous : {output_file}")
