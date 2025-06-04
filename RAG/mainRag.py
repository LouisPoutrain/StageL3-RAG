import argparse
import sys
import os
import pandas as pd
from rag_system import RAGSystem, RefineRAGSystem, parse_json, RAGNonInvasiveDetection
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import logging
from time import time
import json

def process_file(api_key, api_url, json_path, definition, question, method, top_k, output_dir):
    """Traite un fichier JSON unique"""
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    log_path = os.path.join(output_dir, f"{base_name}.txt")

    df_result = pd.DataFrame(columns=[
        'Filename', 'Protocole', 'echantillon', 'Impacts potentiels', 'Extrait pertinent', 
        'Évaluation d\'invasivité', 'Justification d\'invasivité', 'Péchés identifiés', 'Taux de confiance', 
        'annonce_invasivite', 'justification_annonce_invasivite'
    ])

    with open(log_path, "w", encoding="utf-8") as f:
        original_stdout = sys.stdout
        sys.stdout = f

        try:
            print(f"Traitement du fichier : {json_path}")
            start_time = time()
            
            # Utilisation de RefineRAGSystem au lieu de RAGSystem
            rag_system = RefineRAGSystem(api_key=api_key, api_url=api_url)
            rag_system.index_from_json(json_path)
            rag_system_non_invasive_detection = RAGNonInvasiveDetection(api_key=api_key, api_url=api_url)

            if method in ["refine", "both"]:
                print("\n--- Méthode avec Refine ---")
                refine_answer = rag_system.refine_analysis(question, definition, top_k=top_k, title=base_name)
                
                json_final = refine_answer
                resultats = parse_json(json_final, base_name)
                
                if not resultats.empty:
                    print("\n=== Résultats de l'analyse ===")
                    for idx, row in resultats.iterrows():
                        print(f"\nProtocole {idx + 1}:")
                        print(f"Nom: {row['Protocole']}")
                        print(f"Échantillon: {row['echantillon']}")
                        print(f"Évaluation: {row['Évaluation d\'invasivité']}")
                        print(f"Justification: {row['Justification d\'invasivité']}")
                        print(f"Taux de confiance: {row['Taux de confiance']}%")
                        
                        print("\nImpacts potentiels:")
                        for impact in row['Impacts potentiels']:
                            print(f"- {impact}")
                        print("\nPéchés identifiés:")
                        for peche in row['Péchés identifiés']:
                            print(f"- Péché #{peche}")
                        print("\nExtraits pertinents:")
                        for extrait in row['Extrait pertinent']:
                            print(f"- {extrait}")
                        print("-" * 50)
                    
                    print("\n--- Méthode avec Refine et detect_non_invasive_level ---")
                    # Pour chaque protocole dans les résultats
                    for idx, row in resultats.iterrows():
                        protocole = {
                            "nom": row["Protocole"],
                            "description": row["Extrait pertinent"],
                            "evaluation": row["Évaluation d'invasivité"]
                        }
                        if protocole['evaluation'] == 'Invasif':
                            try:
                                invasive_json_path = os.path.join("output_invasive_detection", f"{base_name}.json")
                                # On indexe le fichier
                                rag_system_non_invasive_detection.index_from_json(invasive_json_path)
                                # récupère le titre de l'article dans le fichier json si il y'a une section title
                                title = base_name
                                with open(invasive_json_path, "r", encoding="utf-8") as f:
                                    invasive_data = json.load(f)
                                for section in invasive_data:
                                    if section["section"] == "title":
                                        title = section["text"]
                                        break
                                
                                question = f"{protocole['nom'].capitalize()} constitue-t-il une méthode non invasive ou minimalement invasive pour l'obtention d'échantillons génétiques chez les animaux étudiés ?"
                                non_invasive_level = rag_system_non_invasive_detection.answer_question(
                                    question=question,
                                    protocole=str(protocole),
                                    top_k=8,
                                    title=title
                                )
                                
                                print(f"\nAnalyse du niveau non invasif pour le protocole {protocole['nom']}:")
                                print(non_invasive_level)
                                non_invasive_level = parse_json(non_invasive_level, base_name, invasive_detection=True)
                                print(f"non_invasive_level: {non_invasive_level}")

                                # Insérer les données dans le DataFrame
                                resultats.at[idx, 'annonce_invasivite'] = non_invasive_level[0]['annonce_invasivite']
                                resultats.at[idx, 'justification_annonce_invasivite'] = non_invasive_level[0]['justification_annonce_invasivite']
                            except Exception as e:
                                print(f"Erreur lors de l'analyse du niveau non invasif pour le protocole {protocole['nom']}: {str(e)}")
                                resultats.at[idx, 'annonce_invasivite'] = f"Erreur: {str(e)}"
                                resultats.at[idx, 'justification_annonce_invasivite'] = "Erreur lors de l'analyse"
                    print("--------------------------------")
                    
                    # Vérification et nettoyage des DataFrames avant la concaténation
                    if df_result.empty:
                        df_result = resultats.copy()
                    else:
                        # S'assurer que les types de colonnes sont compatibles
                        for col in df_result.columns:
                            if col in resultats.columns:
                                df_result[col] = df_result[col].astype(resultats[col].dtype)
                        df_result = pd.concat([df_result, resultats], ignore_index=True)
                    print(f"\nTraitement terminé avec succès pour le fichier : {json_path}")
                else:
                    print(f"\nAucun résultat généré pour le fichier : {json_path}")
            
            if method in ["standard", "both"]:
                print("\n--- Méthode standard ---")
                standard_answer = rag_system.answer_question(question, definition, use_hyde=False, top_k=top_k)
                print("\nRéponse standard:")
                print(standard_answer)
                
                json_final = rag_system.fusionne_analyses([standard_answer])
                resultats = parse_json(json_final, base_name)
                
                if not resultats.empty:
                    print("\n=== Résultats de l'analyse standard ===")
                    for idx, row in resultats.iterrows():
                        print(f"\nProtocole {idx + 1}:")
                        print(f"Nom: {row['Protocole']}")
                        print(f"Échantillon: {row['echantillon']}")
                        print(f"Évaluation: {row['Évaluation d\'invasivité']}")
                        print(f"Justification: {row['Justification d\'invasivité']}")
                        print(f"Taux de confiance: {row['Taux de confiance']}%")
                        
                        print("\nImpacts potentiels:")
                        for impact in row['Impacts potentiels']:
                            print(f"- {impact}")
                        print("\nPéchés identifiés:")
                        for peche in row['Péchés identifiés']:
                            print(f"- Péché #{peche}")
                        print("\nExtraits pertinents:")
                        for extrait in row['Extrait pertinent']:
                            print(f"- {extrait}")
                        print("-" * 50)
                    
                    # Vérification et nettoyage des DataFrames avant la concaténation
                    if df_result.empty:
                        df_result = resultats.copy()
                    else:
                        # S'assurer que les types de colonnes sont compatibles
                        for col in df_result.columns:
                            if col in resultats.columns:
                                df_result[col] = df_result[col].astype(resultats[col].dtype)
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
            print(f"Traitement terminé avec succès pour : {os.path.basename(json_path)} en {time() - start_time:.2f} secondes")

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
    parser.add_argument("--method", choices=["refine", "standard", "both", "detect_non_invasive_level"], default="refine")
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--output_dir", default="Résultats")
    parser.add_argument("--input_dir", default="output")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_workers", type=int, default=4)
    
    args = parser.parse_args()

    api_key = args.api_key or "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjE0NWYwMDBhLTVhMDYtNDAxMS1hZTkzLWYwYTk5MzMzNWYzZCJ9.xVQJVIgjlzUyZmhp8yIblh4WlcF6Ya88sLUSGtF_Jt4"
    api_url = args.api_url or "http://gpu1.pedagogie.sandbox.univ-tours.fr:32800/api/chat/completions"

    os.makedirs(args.output_dir, exist_ok=True)
    json_files = [f for f in os.listdir(args.input_dir) if f.endswith(".json")]
    
    df_total = pd.DataFrame(columns=[
        'Filename', 'Protocole', 'echantillon', 'Impacts potentiels', 'Extrait pertinent',
        'Évaluation d\'invasivité', 'Justification d\'invasivité', 'Péchés identifiés', 'Taux de confiance', 
        'annonce_invasivite', 'justification_annonce_invasivite'
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
        futures = [executor.submit(process_file_wrapper, task) for task in tasks]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Fichiers traités"):
            try:
                df_result = future.result()
                # Vérification et nettoyage des DataFrames avant la concaténation
                if df_total.empty:
                    df_total = df_result.copy()
                else:
                    # S'assurer que les types de colonnes sont compatibles
                    for col in df_total.columns:
                        if col in df_result.columns:
                            df_total[col] = df_total[col].astype(df_result[col].dtype)
                    df_total = pd.concat([df_total, df_result], ignore_index=True)

                # Sauvegarde intermédiaire après chaque fichier traité
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


 
                            

            
