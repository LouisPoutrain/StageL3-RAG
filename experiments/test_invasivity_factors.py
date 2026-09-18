#!/usr/bin/env python3
"""
Script de test pour analyser l'impact de chaque paramètre sur la conclusion d'invasivité.

Usage:
    python experiments/test_invasivity_factors.py --test temperature --protocol "fecal sampling"
    python experiments/test_invasivity_factors.py --test definition --top-k-values 4 8 15
    python experiments/test_invasivity_factors.py --test all
"""

import json
import sys
import argparse
from typing import List, Dict, Any
import os
from pathlib import Path

# Configuration pour tests
BASE_DIR = Path(__file__).resolve().parent.parent
TEST_RESULTS_DIR = str(BASE_DIR / "data" / "output" / "test_results_invasivity")


class InvasivityFactorTester:
    """Classe pour tester l'impact de différents facteurs sur l'invasivité."""

    def __init__(self, api_key: str, api_url: str):
        self.api_key = api_key
        self.api_url = api_url
        self.results: List[Dict[str, Any]] = []
        os.makedirs(TEST_RESULTS_DIR, exist_ok=True)

    @staticmethod
    def get_test_definitions() -> Dict[str, Dict[str, Any]]:
        """Retourne différentes définitions pour tester la sensibilité."""
        return {
            "taberlet_strict": {
                "name": "Taberlet Strict (Original)",
                "definition": "Selon Taberlet et al. (1999), l'échantillonnage d'ADN non invasif désigne toute méthode permettant d'obtenir du matériel génétique sans avoir à capturer, blesser, ni perturber significativement l'animal. Cela inclut, par exemple, l'analyse d'échantillons laissés dans l'environnement comme les poils, les plumes, les fèces, l'urine, ou encore la salive.",
                "strictness": 7
            },
            "taberlet_permissive": {
                "name": "Taberlet Permissive",
                "definition": "L'échantillonnage d'ADN non invasif désigne toute méthode ne causant pas de blessures permanentes ou de dommages permanents à l'animal, y compris la capture temporaire si elle ne crée pas de stress durable.",
                "strictness": 3
            },
            "medical_non_invasive": {
                "name": "Définition Médicale (Piège)",
                "definition": "Non-invasif signifie qui ne pénètre pas la barrière de la peau. Cela inclut la capture et la manipulation s'il n'y a pas de perforation cutanée.",
                "strictness": 2
            },
            "broad_definition": {
                "name": "Définition Très Large",
                "definition": "Toute collecte d'échantillon biologique d'animaux sauvages dans la nature.",
                "strictness": 1
            }
        }

    def test_definition_sensitivity(self, protocol: str, chunks: List[str]) -> List[Dict[str, Any]]:
        """Teste comment différentes définitions changent la conclusion."""
        print(f"\n[Test de sensibilite a la definition]")
        print(f"Protocole : {protocol}")

        results = []
        for def_key, def_data in self.get_test_definitions().items():
            print(f"  - Test avec {def_data['name']} (strictesse: {def_data['strictness']}/10)")
            res = {
                "definition_key": def_key,
                "definition_name": def_data["name"],
                "strictness": def_data["strictness"],
                "protocol": protocol,
                "status": "configured"
            }
            results.append(res)
        return results

    def test_temperature_sensitivity(
        self,
        protocol: str,
        chunks: List[str],
        definition: str,
        temperatures: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """Teste la stabilité en faisant varier la température."""
        temperatures = temperatures or [0.1, 0.3, 0.5, 0.7, 0.9]
        print(f"\n[Test de sensibilite a la temperature: {temperatures}]")
        results = []
        for temp in temperatures:
            print(f"  - Test temp={temp}")
            results.append({"temperature": temp, "status": "configured"})
        return results

    def run_all_tests(self, protocol: str, test_chunks: List[str]) -> None:
        """Exécute l'ensemble des scénarios de test."""
        self.test_definition_sensitivity(protocol, test_chunks)
        self.test_temperature_sensitivity(
            protocol, test_chunks, self.get_test_definitions()["taberlet_strict"]["definition"]
        )


def main():
    parser = argparse.ArgumentParser(description="Test des facteurs d'influence de l'invasivite")
    parser.add_argument("--test", choices=["all", "definition", "temperature"], default="all")
    parser.add_argument("--protocol", default="Capture et prelevement de sang", help="Protocole a tester")
    parser.add_argument("--api-key", default=os.getenv("LLM_API_KEY", ""), help="Cle API LLM")
    parser.add_argument("--api-url", default=os.getenv("LLM_API_URL", "http://localhost:11434/api/chat"), help="URL API LLM")

    args = parser.parse_args()
    test_chunks = [
        "Les feces ont ete collectees sur le terrain sans capturer les animaux.",
        "L'echantillonnage a ete effectue sans derangement de la faune."
    ]

    tester = InvasivityFactorTester(api_key=args.api_key, api_url=args.api_url)
    if args.test == "all":
        tester.run_all_tests(args.protocol, test_chunks)
    elif args.test == "definition":
        tester.test_definition_sensitivity(args.protocol, test_chunks)
    elif args.test == "temperature":
        tester.test_temperature_sensitivity(
            args.protocol, test_chunks, tester.get_test_definitions()["taberlet_strict"]["definition"]
        )


if __name__ == "__main__":
    main()
