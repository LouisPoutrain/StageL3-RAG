import argparse
import sys
import os
import pandas as pd
from rag_system import RAGSystem, parse_json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import logging
import tqdm
from time import time

def process_file(api_key, api_url, json_path, definition, question, method, top_k, output_dir):
    """Traite un fichier JSON unique"""
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    log_path = os.path.join(output_dir, f"{base_name}.txt")

    df_result = pd.DataFrame(columns=[
        'Filename', 'Protocole', 'echantillon', 'Impacts potentiels', 'Extrait pertinent', 
        'Évaluation d\'invasivité', 'Péchés identifiés', 'Nouveaux péchés'
    ])

    with open(log_path, "w", encoding="utf-8") as f:
        original_stdout = sys.stdout
        sys.stdout = f

        try:
            print(f"Traitement du fichier : {json_path}")
            start_time = time()
            rag_system = RAGSystem(api_key=api_key, api_url=api_url)
            rag_system.index_from_json(json_path)

            hyde_answer = ""
            if method in ["hyde", "both"]:
                print("\n--- Méthode avec HyDE ---")
                #hyde_answer = rag_system.answer_question(question, definition, use_hyde=True, top_k=top_k)
                hyde_answer = rag_system.analyse_par_chunk(question, definition, top_k=top_k)

                print("\nRéponse avec HyDE:")
                print(hyde_answer)
                print("--------------------------------")
            standard_answer = ""
            if method in ["standard", "both"]:
                print("\n--- Méthode standard ---")
                standard_answer = rag_system.answer_question(question, definition, use_hyde=False, top_k=top_k)
                print("\nRéponse standard:")
                print(standard_answer)

            answer_to_parse = hyde_answer if hyde_answer else standard_answer
            json_final = rag_system.fusionne_analyses(hyde_answer)
            print(json_final)
            resultats = parse_json(json_final, base_name)
            
            if not resultats.empty:
                print("\nDonnées parsées:")
                print(resultats)
                df_result = pd.concat([df_result, resultats], ignore_index=True)
                print(f"\nTraitement terminé avec succès pour le fichier : {json_path}")
            else:
                print(f"\nAucun résultat généré pour le fichier : {json_path}")

        except Exception as e:
            print(f"Erreur lors du traitement du fichier {json_path}: {str(e)}")
            import traceback
            print(traceback.format_exc())
        finally:
            sys.stdout = original_stdout
            print(f"Traitement terminé avec succès pour : {os.path.basename(json_path)} en {time.time() - start_time:.2f} secondes")

    return df_result

def process_file_wrapper(args):
    return process_file(*args)

def main():

    
    parser = argparse.ArgumentParser(description="Système RAG pour l'analyse de documents scientifiques")
    parser.add_argument("--api_key", help="Clé API pour le LLM")
    parser.add_argument("--api_url", help="URL de l'API LLM")
    parser.add_argument("--json_file", default="output/A novel molecular method for noninvasive sex identification of order Carnivora.json")
    parser.add_argument("--definition", default="Selon Taberlet et al. (1999), l'échantillonnage d'ADN non invasif désigne toute méthode permettant d'obtenir du matériel génétique sans avoir à capturer, blesser, ni perturber significativement l'animal. Cela inclut, par exemple, l'analyse d'échantillons laissés dans l'environnement comme les poils, les plumes, les fèces, l'urine, ou encore la salive.")
    parser.add_argument("--question", default="Dans l'article, peux-tu me donner tous les protocoles d'échantillonnage d'ADN et s'ils sont considérés comme invasifs ou non selon la définition de Taberlet ?")
    parser.add_argument("--method", choices=["hyde", "standard", "both"], default="hyde")
    parser.add_argument("--top_k", type=int, default=6)
    parser.add_argument("--output_dir", default="Résultats")
    parser.add_argument("--input_dir", default="output")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_workers", type=int, default=4)
    
    args = parser.parse_args()

    api_key = args.api_key or "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6ImRjZjZkYWE2LTRjYzMtNDYyOS05MjJiLTkyYzM1NGQzODYwMCJ9.04oTM2nVW8iQTCr8qrs4MknI13dGgqBp85Wq7t2jAeQ"
    api_url = args.api_url or "http://gpu1.pedagogie.sandbox.univ-tours.fr:32800/api/chat/completions"

    os.makedirs(args.output_dir, exist_ok=True)
    json_files = [f for f in os.listdir(args.input_dir) if f.endswith(".json")]
    
    df_total = pd.DataFrame(columns=[
        'Filename', 'Protocole', 'echantillon', 'Impacts potentiels', 'Extrait pertinent',
        'Évaluation d\'invasivité', 'Péchés identifiés', 'Nouveaux péchés'
    ])

    # Création des tâches par lots
    tasks = []
    for i in range(0, len(json_files), args.batch_size):
        batch = json_files[i:i + args.batch_size]
        for filename in batch:
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

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        for i in range(0, len(json_files), args.batch_size):
            batch = json_files[i:i + args.batch_size]
            for json_file in batch:
                path = os.path.join(args.input_dir, json_file)
                futures.append(executor.submit(process_file, api_key, api_url, path, args.definition, args.question, args.method, args.top_k, args.output_dir))

        # Barre de progression globale
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fichiers traités"):
            try:
                df_result = future.result()
                df_total = pd.concat([df_total, df_result], ignore_index=True)
                df_total.to_csv(os.path.join(args.output_dir, "Protocoles_intermediaire.csv"), index=False)

            except Exception as e:
                print(f"Erreur dans un traitement parallèle : {e}")

    # Sauvegarde finale
    if not df_total.empty:
        df_total.to_csv(os.path.join(args.output_dir, "Protocoles.csv"), index=False)
        print("Traitement terminé. Résultats sauvegardés dans 'Protocoles.csv'.")
    else:
        print("Attention: Aucun résultat n'a été généré.")

if __name__ == "__main__":
    main()


 
                            

            
