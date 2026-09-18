#!/usr/bin/env python3
"""
Script de lancement des tests sur les 5 articles recommandés.

Usage:
    python experiments/run_tests_on_articles.py --article 1 --test definition
    python experiments/run_tests_on_articles.py --article all --test temperature
    python experiments/run_tests_on_articles.py --article 5 --test all
"""

import sys
import json
import argparse
from pathlib import Path
import pandas as pd

# Articles recommandés avec leurs métadonnées
RECOMMENDED_ARTICLES = {
    1: {
        "filename": "012017-jfwm-007",
        "title": "Identification of Southeastern Bat Species Using Noninvasive Genetic Sampling",
        "year": 2017,
        "test_focus": ["CONTRADICTION", "DEFINITION", "AMBIGUITY", "SINS"],
        "protocols": [
            "Échantillonnage d'ADN à partir de guano et de tissu",
            "Échantillonnage d'ADN à partir de guano"
        ],
        "expected_evaluation": "Invasif",
        "title_says": "Noninvasive",
        "sins": [1, 3]
    },
    2: {
        "filename": "1-s2.0-S0006320713002772-main",
        "title": "Combining camera-trapping and noninvasive genetic data in spatial capture-recapture",
        "year": 2013,
        "test_focus": ["TEMPERATURE", "VARIANCE", "CONFIDENCE"],
        "protocols": [
            "Utilisation de pièges photographiques"
        ],
        "expected_evaluation": "Invasif",
        "expected_confidence": "75%",
        "confidence_reason": "Manque d'informations sur impact des pièges"
    },
    4: {
        "filename": "012017-jfwm-007",
        "title": "Identification of Southeastern Bat Species",
        "year": 2017,
        "test_focus": ["SINS", "RULES"],
        "note": "Même que Article #1, teste le mécanisme des 7 péchés",
        "sins": [1, 3],
        "sin_details": {
            1: "Mauvaise classification des fèces (la capture implique invasif)",
            3: "Échantillonnage systématique"
        }
    },
    5: {
        "filename": "Genetic structure and population history of wintering Asian Great Bustard",
        "title": "Genetic structure of Asian Great Bustard - multi-species sampling",
        "year": None,
        "test_focus": ["VARIANCE_INTRA", "CONSISTENCY", "CHUNK_ORDER"],
        "protocols": [
            "Fecal sampling of Great Bustard",
            "Tissue sampling of Leopard Cat",
            "Stomach sampling of Leopard Cat",
            "Scat sampling of Leopard Cat"
        ],
        "expected_evaluations": {
            "Great Bustard": "Non invasif",
            "Leopard Cat (tissue)": "Invasif",
            "Leopard Cat (stomach)": "Invasif",
            "Leopard Cat (scat)": "Invasif - Territory marking"
        }
    }
}


def load_excel_data() -> pd.DataFrame:
    """Charge les données du fichier de vérité terrain."""
    base_dir = Path(__file__).resolve().parent.parent
    candidates = [
        base_dir / "data" / "benchmarks" / "MergedRawData.xlsx",
        base_dir / "data" / "benchmarks" / "FinalRawData.xlsx",
        base_dir / "Résultats" / "MergedRawData.xlsx",
    ]
    excel_file = next((p for p in candidates if p.exists()), candidates[0])
    if not excel_file.exists():
        print(f"Attention : classeur de référence introuvable.")
        return pd.DataFrame()
    return pd.read_excel(excel_file, sheet_name=0)


def get_article_data(article_num: int, df: pd.DataFrame) -> dict:
    """Récupère les données pour un article spécifique."""
    article = RECOMMENDED_ARTICLES[article_num]
    filename = article["filename"]

    if df.empty or 'Filename_Protocoles' not in df.columns:
        return {"metadata": article, "rows": pd.DataFrame(), "num_protocols": 0}

    article_data = df[df['Filename_Protocoles'] == filename].copy()
    return {
        "metadata": article,
        "rows": article_data,
        "num_protocols": len(article_data)
    }


def print_article_summary(article_num: int) -> None:
    """Affiche un résumé structuré de l'article de test."""
    article = RECOMMENDED_ARTICLES[article_num]
    print("\n" + "=" * 80)
    print(f"ARTICLE #{article_num}: {article['title']}")
    print("=" * 80)
    print(f"Fichier  : {article['filename']}")
    print(f"Objectif : {', '.join(article['test_focus'])}")
    print(f"Protocoles attendus :")
    for p in article.get('protocols', []):
        print(f"  - {p}")


def generate_test_commands(article_num: int, test_type: str) -> None:
    """Affiche les commandes CLI recommandées pour tester un article."""
    article = RECOMMENDED_ARTICLES[article_num]
    fname = article['filename']
    print(f"\n[Commandes recommandees pour {test_type}]")
    print(f"  python experiments/test_parameter_influence.py --article {fname} --test {test_type}")


def main():
    parser = argparse.ArgumentParser(description="Gestion des tests sur les articles recommandés")
    parser.add_argument("--article", type=str, choices=["1", "2", "4", "5", "all"], default="all")
    parser.add_argument("--test", type=str, choices=["definition", "temperature", "sins", "variance", "all"], default="all")
    parser.add_argument("--summary", action="store_true", help="Afficher seulement le résumé")

    args = parser.parse_args()
    df = load_excel_data()

    articles_to_test = [int(args.article)] if args.article != "all" else [1, 2, 4, 5]

    for article_num in articles_to_test:
        if article_num in RECOMMENDED_ARTICLES:
            print_article_summary(article_num)

            if not args.summary:
                if args.test == "all":
                    for test_type in ["definition", "temperature", "sins", "variance"]:
                        generate_test_commands(article_num, test_type)
                else:
                    generate_test_commands(article_num, args.test)

            article_data = get_article_data(article_num, df)
            if len(article_data['rows']) > 0:
                print(f"\nDonnees de reference : {len(article_data['rows'])} protocole(s)")
                for idx, (_, row) in enumerate(article_data['rows'].iterrows(), 1):
                    sampling_val = row.get('Sampling', 'N/A')
                    eval_val = row.get("Évaluation d'invasivité", 'N/A')
                    conf_val = row.get('Taux de confiance', 'N/A')
                    print(f"  Protocole #{idx}:")
                    print(f"    - Echantillon : {sampling_val}")
                    print(f"    - Evaluation  : {eval_val}")
                    print(f"    - Confiance   : {conf_val}")

    print("\n" + "=" * 80)
    print("Plan de test pret a l'execution.")
    print("=" * 80)


if __name__ == "__main__":
    main()
