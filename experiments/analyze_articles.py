#!/usr/bin/env python3
"""
Script pour analyser les données de vérité terrain et sélectionner les articles pour les tests.

Usage:
    python experiments/analyze_articles.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Résolution portable du chemin du fichier Excel de référence
BASE_DIR = Path(__file__).resolve().parent.parent
EXCEL_CANDIDATES = [
    BASE_DIR / "data" / "benchmarks" / "MergedRawData.xlsx",
    BASE_DIR / "data" / "benchmarks" / "FinalRawData.xlsx",
    BASE_DIR / "Résultats" / "MergedRawData.xlsx",
]
excel_file = next((p for p in EXCEL_CANDIDATES if p.exists()), EXCEL_CANDIDATES[0])

if not excel_file.exists():
    print(f"Erreur : fichier de référence introuvable parmi les candidats.")
    sys.exit(1)

df = pd.read_excel(excel_file, sheet_name=0)

print("=" * 120)
print("ANALYSE DU CORPUS POUR LA SELECTION D'ARTICLES DE TEST")
print("=" * 120)

# Statistiques générales
col_filename = 'Filename_TEI' if 'Filename_TEI' in df.columns else df.columns[0]
print(f"\nSTATISTIQUES GENERALES:")
print(f"  - Total de lignes (protocoles) : {len(df)}")
print(f"  - Articles uniques             : {df[col_filename].nunique()}")

# 1. Contradiction: titre dit non-invasif mais évaluation dit invasif
print(f"\n\nCRITERE 1: CONTRADICTION (Titre vs Evaluation)")
print("-" * 120)

if 'Non invasive in the title' in df.columns and "Évaluation d'invasivité" in df.columns:
    contradictions = df[
        (df['Non invasive in the title'].astype(str).str.lower().str.contains('oui|yes')) & 
        (df["Évaluation d'invasivité"].astype(str).str.lower().str.contains('invasif|invasive'))
    ].copy()

    print(f"Articles avec contradiction (titre=Non-invasif + evaluation=Invasif): {len(contradictions)}")
    if len(contradictions) > 0:
        for idx, row in contradictions.head(5).iterrows():
            title_val = row.get('Title', 'Sans titre')
            fname_val = row.get('Filename_Protocoles', 'Inconnu')
            sample_val = row.get('Sampling', 'N/A')
            excerpt_val = row.get('Excerpt', 'N/A')
            print(f"\n  Ligne {idx}: {str(title_val)[:80]}...")
            print(f"      Fichier : {fname_val}")
            print(f"      Protocole : {sample_val}")
            print(f"      Extrait   : {str(excerpt_val)[:100]}...")
else:
    print("Colonnes requises absentes pour l'analyse de contradiction directe.")

print("\n" + "=" * 120)
print("Analyse terminee avec succes.")
