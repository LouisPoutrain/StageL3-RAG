#!/usr/bin/env python3
"""
Script pour tester l'influence des paramètres sur la conclusion de la pipeline RAG.
Exécute la pipeline avec différents paramètres (définition, température, top-k, n_generations).

Usage:
    python experiments/test_parameter_influence.py --article 012017-jfwm-007 --test definition
    python experiments/test_parameter_influence.py --article all --test temperature
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Résolution portable des chemins
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

try:
    from rag.rag_system import RAGSystem, RefineRAGSystem
    from rag.pipelines import create_hyde_pipeline
    from rag.UniversityLLMAdapter import UniversityLLMAdapter
except ImportError as e:
    print(f"Erreur d'import des modules RAG: {e}")
    print(f"sys.path: {sys.path[:3]}")
    sys.exit(1)

# Configuration sécurisée via variables d'environnement
TEST_RESULTS_DIR = str(BASE_DIR / "data" / "output" / "test_results_comparative")
API_KEY = os.getenv("LLM_API_KEY", "")
API_URL = os.getenv("LLM_API_URL", "http://localhost:11434/api/chat")

# Articles sélectionnés
SELECTED_ARTICLES = {
    "012017-jfwm-007": {
        "json_file": str(BASE_DIR / "data" / "input" / "012017-jfwm-007.json"),
        "title": "Identification of Southeastern Bat Species Using Noninvasive Genetic Sampling"
    },
    "1-s2.0-S0006320713002772-main": {
        "json_file": str(BASE_DIR / "data" / "input" / "1-s2.0-S0006320713002772-main.json"),
        "title": "Combining camera-trapping and noninvasive genetic data"
    },
    "23.JournalofNaturalHistory": {
        "json_file": str(BASE_DIR / "data" / "input" / "23.JournalofNaturalHistory.json"),
        "title": "Non-invasive genetic study and population monitoring of the brown bear"
    }
}

DEFINITIONS_TO_TEST = {
    "taberlet_original": "Selon Taberlet et al. (1999), l'échantillonnage d'ADN non invasif désigne toute méthode permettant d'obtenir du matériel génétique sans avoir à capturer, blesser, ni perturber significativement l'animal. Cela inclut, par exemple, l'analyse d'échantillons laissés dans l'environnement comme les poils, les plumes, les fèces, l'urine, ou encore la salive.",
    "taberlet_strict": "L'échantillonnage d'ADN non invasif désigne exclusivement toute méthode n'impliquant AUCUN contact avec l'animal vivant, AUCUNE capture, AUCUNE manipulation, et AUCUNE perturbation observable de son comportement ou territoire. Seule la collecte passive d'échantillons environnementaux sans interaction avec l'animal est acceptable.",
    "taberlet_permissive": "L'échantillonnage d'ADN non invasif désigne toute méthode ne causant pas de blessures permanentes ou de dommages irréversibles à l'animal. La capture temporaire et la manipulation brève sont acceptables si elles ne causent pas de stress durable ou de modifications comportementales à long terme.",
    "medical_definition": "L'échantillonnage d'ADN non invasif désigne toute procédure qui ne pénètre pas la barrière cutanée de l'animal. La capture, la manipulation et le confinement temporaire sont considérés comme non invasifs tant qu'il n'y a pas de perforation de la peau.",
    "minimal_definition": "L'échantillonnage d'ADN non invasif désigne tout prélèvement d'ADN qui ne nécessite pas de procédure chirurgicale."
}


class ParameterInfluenceTester:
    """Teste l'influence des paramètres sur les conclusions de la pipeline RAG."""

    def __init__(self, api_key: str, api_url: str):
        self.api_key = api_key
        self.api_url = api_url
        self.results: List[Dict[str, Any]] = []
        os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
        print("Testeur initialise")
        print(f"Repertoire resultats : {TEST_RESULTS_DIR}/")

    def run_rag_analysis(
        self,
        article_file: str,
        definition: str,
        temperature: float = 0.7,
        top_k: int = 8,
        n_generations: int = 3,
        test_id: str = ""
    ) -> Dict[str, Any]:
        """Exécute la pipeline RAG avec les paramètres spécifiés."""
        print(f"\nExecution RAG : test_id={test_id}")
        print(f"  temperature={temperature}, top_k={top_k}, n_generations={n_generations}")

        try:
            rag_system = RefineRAGSystem(api_key=self.api_key, api_url=self.api_url)
            rag_system.llm_adapter.temperature = temperature

            with open(article_file, 'r', encoding='utf-8') as f:
                article_data = json.load(f)

            if isinstance(article_data, list):
                article_data = article_data[0] if article_data else {}

            title = article_data.get('title', '') if isinstance(article_data, dict) else ''
            question = "Dans l'article, peux-tu me donner tous les protocoles d'échantillonnage d'ADN et s'ils sont considérés comme invasifs ou non selon la définition de Taberlet ?"

            result = rag_system.refine_analysis(
                question=question,
                definition=definition,
                top_k=top_k,
                title=title
            )

            protocols = self._parse_rag_output(result)

            return {
                "test_id": test_id,
                "success": True,
                "protocols": protocols,
                "raw_output": result,
                "parameters": {
                    "definition_key": test_id.split('_')[0] if '_' in test_id else "unknown",
                    "temperature": temperature,
                    "top_k": top_k,
                    "n_generations": n_generations
                }
            }

        except Exception as e:
            print(f"Erreur d'execution : {str(e)}")
            return {
                "test_id": test_id,
                "success": False,
                "error": str(e),
                "parameters": {
                    "temperature": temperature,
                    "top_k": top_k,
                    "n_generations": n_generations
                }
            }

    def _parse_rag_output(self, output: str) -> List[Dict]:
        """Parse la sortie JSON de la pipeline RAG."""
        try:
            if '```json' in output:
                json_start = output.find('```json') + 7
                json_end = output.find('```', json_start)
                json_str = output[json_start:json_end].strip()
            elif '[' in output:
                json_start = output.find('[')
                json_end = output.rfind(']') + 1
                json_str = output[json_start:json_end]
            else:
                json_str = output

            return json.loads(json_str)
        except Exception as e:
            print(f"Avertissement parsing JSON : {e}")
            return []

    def test_definition_influence(self, article_file: str, article_name: str) -> List[Dict]:
        """Test 1: Influence de la DEFINITION."""
        print(f"\nTEST 1: INFLUENCE DE LA DEFINITION - {article_name}")
        results = []
        for def_key, definition in DEFINITIONS_TO_TEST.items():
            print(f"Test definition : {def_key}")
            result = self.run_rag_analysis(
                article_file=article_file,
                definition=definition,
                temperature=0.7,
                top_k=8,
                n_generations=3,
                test_id=f"{def_key}_temp0.7_k8"
            )
            result['definition_key'] = def_key
            result['definition_text'] = definition[:200]
            results.append(result)

        output_file = os.path.join(TEST_RESULTS_DIR, f"definition_test_{article_name}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        self._analyze_definition_results(results, article_name)
        return results

    def test_temperature_influence(
        self, article_file: str, article_name: str, definition: str, runs_per_temp: int = 3
    ) -> List[Dict]:
        """Test 2: Influence de la TEMPERATURE."""
        print(f"\nTEST 2: INFLUENCE DE LA TEMPERATURE - {article_name}")
        temperatures = [0.1, 0.3, 0.5, 0.7, 0.9]
        results = []

        for temp in temperatures:
            print(f"Test temperature : {temp}")
            for run in range(runs_per_temp):
                result = self.run_rag_analysis(
                    article_file=article_file,
                    definition=definition,
                    temperature=temp,
                    top_k=8,
                    n_generations=3,
                    test_id=f"temp{temp}_run{run+1}"
                )
                result['temperature'] = temp
                result['run_number'] = run + 1
                results.append(result)

        output_file = os.path.join(TEST_RESULTS_DIR, f"temperature_test_{article_name}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        self._analyze_temperature_results(results, article_name)
        return results

    def test_topk_influence(self, article_file: str, article_name: str, definition: str) -> List[Dict]:
        """Test 3: Influence du TOP-K."""
        print(f"\nTEST 3: INFLUENCE DU TOP-K - {article_name}")
        top_k_values = [2, 4, 6, 8, 10, 15]
        results = []

        for top_k in top_k_values:
            result = self.run_rag_analysis(
                article_file=article_file,
                definition=definition,
                temperature=0.7,
                top_k=top_k,
                n_generations=3,
                test_id=f"topk{top_k}"
            )
            result['top_k_value'] = top_k
            results.append(result)

        output_file = os.path.join(TEST_RESULTS_DIR, f"topk_test_{article_name}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return results

    def _analyze_definition_results(self, results: List[Dict], article_name: str) -> None:
        """Analyse comparative des résultats selon la définition."""
        comparison = []
        for result in results:
            if result.get('success'):
                protocols = result.get('protocols', [])
                if not isinstance(protocols, list):
                    continue

                invasif_count = 0
                non_invasif_count = 0
                protocols_detail = []

                for p in protocols:
                    if not isinstance(p, dict):
                        continue
                    eval_invasivite = p.get('evaluation_invasivite', '') or p.get("Évaluation d'invasivité", '')
                    eval_lower = str(eval_invasivite).lower()

                    if 'invasif' in eval_lower and 'non' not in eval_lower:
                        invasif_count += 1
                    elif 'non invasif' in eval_lower:
                        non_invasif_count += 1

                    protocols_detail.append({
                        'nom': p.get('protocole', p.get('Protocole', 'Unknown')),
                        'evaluation': eval_invasivite,
                        'confiance': p.get('taux_de_confiance', p.get('Taux de confiance', 'Unknown'))
                    })

                comparison.append({
                    'definition': result['definition_key'],
                    'num_protocols': len(protocols),
                    'invasif_count': invasif_count,
                    'non_invasif_count': non_invasif_count,
                    'protocols_detail': protocols_detail
                })

        analysis_file = os.path.join(TEST_RESULTS_DIR, f"definition_analysis_{article_name}.json")
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)

    def _analyze_temperature_results(self, results: List[Dict], article_name: str) -> None:
        """Analyse de la variance selon la température."""
        by_temp: Dict[float, List[Dict]] = {}
        for result in results:
            if result.get('success'):
                temp = result.get('temperature', 0.7)
                by_temp.setdefault(temp, []).append(result)

        variance_data = []
        for temp in sorted(by_temp.keys()):
            runs = by_temp[temp]
            invasif_counts = []
            for run in runs:
                protocols = run.get('protocols', [])
                count = sum(
                    1 for p in protocols
                    if isinstance(p, dict) and 'invasif' in str(p.get('evaluation_invasivite', '')).lower()
                    and 'non' not in str(p.get('evaluation_invasivite', '')).lower()
                )
                invasif_counts.append(count)

            variance_data.append({
                'temperature': temp,
                'num_runs': len(runs),
                'invasif_counts': invasif_counts,
                'variance': max(invasif_counts) - min(invasif_counts) if invasif_counts else 0
            })

        analysis_file = os.path.join(TEST_RESULTS_DIR, f"temperature_analysis_{article_name}.json")
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(variance_data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Test de l'influence des parametres sur la pipeline RAG")
    parser.add_argument("--article", choices=list(SELECTED_ARTICLES.keys()) + ["all"], default="012017-jfwm-007")
    parser.add_argument("--test", choices=["definition", "temperature", "topk", "all"], default="all")
    parser.add_argument("--api-key", default=API_KEY, help="Cle d'API LLM")
    parser.add_argument("--api-url", default=API_URL, help="URL d'API LLM")

    args = parser.parse_args()
    tester = ParameterInfluenceTester(args.api_key, args.api_url)

    articles_to_test = [args.article] if args.article != "all" else list(SELECTED_ARTICLES.keys())
    default_def = DEFINITIONS_TO_TEST["taberlet_original"]

    for art_key in articles_to_test:
        art_info = SELECTED_ARTICLES.get(art_key)
        if not art_info or not os.path.exists(art_info["json_file"]):
            continue

        fpath = art_info["json_file"]
        if args.test in ["all", "definition"]:
            tester.test_definition_influence(fpath, art_key)
        if args.test in ["all", "temperature"]:
            tester.test_temperature_influence(fpath, art_key, default_def)
        if args.test in ["all", "topk"]:
            tester.test_topk_influence(fpath, art_key, default_def)

    print("Tests acheves avec succes.")


if __name__ == "__main__":
    main()
