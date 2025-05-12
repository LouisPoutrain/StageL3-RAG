import argparse
import sys
import os
import pandas as pd
from rag_system import RAGSystem, parse_json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def process_file(api_key, api_url, json_path, definition, question, method, top_k, output_dir):
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    log_path = os.path.join(output_dir, f"{base_name}.txt")



    df_result = pd.DataFrame(columns=[
        'Filename', 'Protocole', 'echantillon','Impacts potentiels' ,'Extrait pertinent', 
        'Évaluation d\'invasivité', 'Péchés identifiés', 'Nouveaux péchés'
    ])

    with open(log_path, "w", encoding="utf-8") as f:
        original_stdout = sys.stdout
        sys.stdout = f

        try:
            print(f"Traitement du fichier : {json_path}")

            rag_system = RAGSystem(api_key=api_key, api_url=api_url)
            rag_system.index_from_json(json_path)

            hyde_answer = ""
            if method in ["hyde", "both"]:
                print("\n--- Méthode avec HyDE ---")
                hyde_answer = rag_system.answer_question(question, definition, use_hyde=True, top_k=top_k)
                print("\nRéponse avec HyDE:")
                print(hyde_answer)

            standard_answer = ""
            if method in ["standard", "both"]:
                print("\n--- Méthode standard ---")
                standard_answer = rag_system.answer_question(question, definition, use_hyde=False)
                print("\nRéponse standard:")
                print(standard_answer)

            print("Données parsées:")
            resultats = parse_json(hyde_answer, base_name)

            expected_columns = ["Filename", "Protocole", "echantillon", 'Impacts potentiels' ,"Extrait pertinent", 
                                "Évaluation d'invasivité", "Péchés identifiés", "Nouveaux péchés"]

            if not all(col in resultats.columns for col in expected_columns):
                print(f" Erreur : colonnes manquantes. Colonnes présentes : {resultats.columns}")
            else:
                print(resultats[expected_columns])
                df_result = pd.concat([df_result, resultats], ignore_index=True)

        finally:
            sys.stdout = original_stdout

    return df_result


def process_file_wrapper(args):
    return process_file(*args)


def main():
    parser = argparse.ArgumentParser(description="Système RAG pour l'analyse de documents scientifiques")
    parser.add_argument("--api_key", help="Clé API pour le LLM")
    parser.add_argument("--api_url", help="URL de l'API LLM")
    parser.add_argument("--json_file", default="output/s11356-015-5754-2.json")
    parser.add_argument("--definition", default="Selon Taberlet et al. (1999), l'échantillonnage d'ADN non invasif désigne toute méthode permettant d'obtenir du matériel génétique sans avoir à capturer, blesser, ni perturber significativement l'animal. Cela inclut, par exemple, l'analyse d'échantillons laissés dans l'environnement comme les poils, les plumes, les fèces, l'urine, ou encore la salive.")
    parser.add_argument("--question", default="Dans l'article, peux-tu me donner tous les protocoles d'échantillonnage d'ADN et s’ils sont considérés comme invasifs ou non selon la définition de Taberlet ?")
    parser.add_argument("--method", choices=["hyde", "standard", "both"], default="hyde")
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--output_dir", default="Résultats")
    parser.add_argument("--input_dir", default="output")

    args = parser.parse_args()

    api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6ImRjZjZkYWE2LTRjYzMtNDYyOS05MjJiLTkyYzM1NGQzODYwMCJ9.04oTM2nVW8iQTCr8qrs4MknI13dGgqBp85Wq7t2jAeQ"
    api_url = args.api_url or "http://gpu1.pedagogie.sandbox.univ-tours.fr:32800/api/chat/completions"


    os.makedirs(args.output_dir, exist_ok=True)

    json_files = [f for f in os.listdir(args.input_dir) if f.endswith(".json")]

    tasks = []
    for filename in json_files:
        json_path = os.path.join(args.input_dir, filename)
        tasks.append((
            api_key,
            api_url,
            json_path,
            args.definition,
            args.question,
            args.method,
            args.top_k,
            args.output_dir
        ))

    print(f"Type de la clé dans tasks : {type(tasks[0][0])}")

    df_total = pd.DataFrame(columns=[
        'Filename', 'Protocole', 'echantillon','Impacts potentiels' , 'Extrait pertinent',
        'Évaluation d\'invasivité', 'Péchés identifiés', 'Nouveaux péchés'
    ])

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_file_wrapper, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Traitement des fichiers JSON"):
            result_df = future.result()
            df_total = pd.concat([df_total, result_df], ignore_index=True)

    df_total.to_csv(os.path.join(args.output_dir, "Test_protocoles.csv"), index=False)
    print("Traitement terminé. Résultats sauvegardés dans 'Test_protocoles.csv'.")


if __name__ == "__main__":
    main()


 
                            

            
