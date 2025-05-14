import argparse
import sys
import os
import pandas as pd
from rag_system import RAGSystem, parse_json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
import gc


def setup_environment():
    """Configure l'environnement pour optimiser l'utilisation de la mémoire"""
    # Import local pour éviter les erreurs de portée
    import torch
    import os
    
    # Limiter l'utilisation de la mémoire GPU si disponible
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Pour les systèmes mac
        # Désactiver complètement la limite mémoire haute pour éviter l'erreur OOM
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        print("MPS est disponible - limite mémoire haute désactivée (ratio 0.0)")
    
    # Nettoyage de la mémoire
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        try:
            # Tentative de libération de la mémoire MPS
            torch.mps.empty_cache()
            print("Cache MPS vidé avec succès")
        except Exception as e:
            print(f"Note: Impossible de vider le cache MPS: {e}")
    
    # Désactiver le parallélisme dans certaines bibliothèques
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    

def process_file(api_key, api_url, json_path, definition, question, method, top_k, output_dir):
    # Imports locaux pour éviter les problèmes de portée entre processus
    import gc
    import torch
    
    # Configuration de l'environnement pour chaque processus
    setup_environment()
    
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    log_path = os.path.join(output_dir, f"{base_name}.txt")

    df_result = pd.DataFrame(columns=[
        'Filename', 'Protocole', 'echantillon','Impacts potentiels' ,'Extrait pertinent', 
        'Évaluation d\'invasivité', 'Péchés identifiés'
    ])

    with open(log_path, "w", encoding="utf-8") as f:
        original_stdout = sys.stdout
        sys.stdout = f

        try:
            print(f"Traitement du fichier : {json_path}")

            try:
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
                    standard_answer = rag_system.answer_question(question, definition, use_hyde=False, top_k=top_k)
                    print("\nRéponse standard:")
                    print(standard_answer)

                answer_to_parse = hyde_answer if hyde_answer else standard_answer
                
                print("Données parsées:")
                resultats = parse_json(answer_to_parse, base_name)

                expected_columns = ["Filename", "Protocole", "echantillon", 'Impacts potentiels' ,"Extrait pertinent", 
                                    "Évaluation d'invasivité", "Péchés identifiés"]

                if not all(col in resultats.columns for col in expected_columns):
                    print(f"Erreur : colonnes manquantes. Colonnes présentes : {resultats.columns}")
                else:
                    print(resultats[expected_columns])
                    df_result = pd.concat([df_result, resultats], ignore_index=True)
            except Exception as e:
                print(f"Erreur spécifique lors du traitement du fichier {json_path}: {str(e)}")
                import traceback
                print(traceback.format_exc())

        finally:
            sys.stdout = original_stdout
    
    # Nettoyage de la mémoire à la fin du traitement
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception as e:
        print(f"Erreur lors du nettoyage mémoire: {e}")

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
    parser.add_argument("--output_dir", default="ResultatsTest")
    parser.add_argument("--input_dir", default="outputTest")
    parser.add_argument("--batch_size", type=int, default=5, help="Nombre de fichiers à traiter en parallèle")
    parser.add_argument("--max_workers", type=int, default=2, help="Nombre de processus en parallèle")
    
    args = parser.parse_args()

    api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6ImRjZjZkYWE2LTRjYzMtNDYyOS05MjJiLTkyYzM1NGQzODYwMCJ9.04oTM2nVW8iQTCr8qrs4MknI13dGgqBp85Wq7t2jAeQ"
    api_url = args.api_url or "http://gpu1.pedagogie.sandbox.univ-tours.fr:32800/api/chat/completions"


    # Désactivation de la limite pour macOS
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    
    # Importer torch pour configurer le backend global
    import torch
    
    setup_environment()
    
    # Affichage les infos sur l'environnement
    print(f"Configuration environnement:")
    print(f"- MPS ratio: {os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO', 'Non défini')}")
    print(f"- OMP threads: {os.environ.get('OMP_NUM_THREADS', 'Non défini')}")
    print(f"- Workers: {args.max_workers}")
    print(f"- Batch size: {args.batch_size}")

    os.makedirs(args.output_dir, exist_ok=True)

    json_files = [f for f in os.listdir(args.input_dir) if f.endswith(".json")]
    # On crée une liste de tâches pour chaque fichier JSON
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
        'Évaluation d\'invasivité', 'Péchés identifiés'
    ])

    # Réduction du nombre de workers (coeurs logiques utilisés) + on limite le nombre de fichiers traités en parallèle
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        batch_size = args.batch_size
        for i in range(0, len(tasks), batch_size): # On parcourt les fichiers par pas de batch_size
            batch = tasks[i:i + batch_size] 
            futures = [executor.submit(process_file_wrapper, task) for task in batch]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Traitement des fichiers {i+1}-{min(i+batch_size, len(tasks))}"):
                try:
                    result_df = future.result()
                    df_total = pd.concat([df_total, result_df], ignore_index=True)
                except Exception as e:
                    print(f"Erreur lors du traitement d'un fichier: {str(e)}")
                    continue
            
            # Nettoyage de la mémoire entre les batchs
            print("Nettoyage de la mémoire entre les batchs...")
            gc.collect()
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except Exception as e:
                print(f"Erreur lors du nettoyage mémoire entre les batchs: {e}")
            
            # Sauvegarde intermédiaire
            df_total.to_csv(os.path.join(args.output_dir, "Protocoles_intermediaire.csv"), index=False)
            print(f"Progression sauvegardée : {i + len(batch)}/{len(tasks)} fichiers traités")

    df_total.to_csv(os.path.join(args.output_dir, "Protocoles.csv"), index=False)
    print("Traitement terminé. Résultats sauvegardés dans 'Test_protocoles.csv'.")


if __name__ == "__main__":
    main()


 
                            

            
