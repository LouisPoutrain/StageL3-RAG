import argparse
import sys
import os
import pandas as pd

# Ajouter le répertoire src au path pour les imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag.rag_system import RAGSystem, RefineRAGSystem, parse_json, RAGNonInvasiveDetection
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import time
import json
from dotenv import load_dotenv
import nltk

DATAFRAME_COLUMNS = [
    'Filename', 'Protocole', 'echantillon', 'Impacts potentiels', 'Extrait pertinent', 
    'Évaluation d\'invasivité', 'Justification d\'invasivité', 'Péchés identifiés', 'Taux de confiance', 
    'annonce_invasivite', 'justification_annonce_invasivite'
]

def process_file(api_key, api_url, json_path, definition, question, method, top_k, output_dir, invasive_input_dir):
    """
    Traite un unique fichier JSON pour en extraire les protocoles, les évaluer
    et analyser leur présentation.

    Args:
        api_key (str): Clé d'API pour le service LLM.
        api_url (str): URL de l'API LLM.
        json_path (str): Chemin vers le fichier JSON à traiter.
        definition (str): Définition de l'invasivité à utiliser.
        question (str): Question à poser au RAG pour l'extraction de protocole.
        method (str): Méthode d'analyse à utiliser ('refine', 'both', ...).  (Obselete)
        top_k (int): Nombre de documents pertinents à récupérer.
        output_dir (str): Répertoire où sauvegarder les logs.
        invasive_input_dir (str): Répertoire contenant les fichiers JSON pour l'analyse d'annonce d'invasivité.

    Returns:
        tuple: Un tuple contenant:
            - df_result (pd.DataFrame): DataFrame avec les résultats de l'analyse.
            - impacts_energy (list): Liste des impacts énergétiques de l'analyse principale.
            - impacts_co2 (list): Liste des impacts CO2 de l'analyse principale.
            - impacts_energy_non_invasive (list): Liste des impacts énergétiques de l'analyse d'invasivité.
            - impacts_co2_non_invasive (list): Liste des impacts CO2 de l'analyse d'invasivité.
    """
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    log_path = os.path.join(output_dir, f"{base_name}.txt")

    df_result = pd.DataFrame(columns=DATAFRAME_COLUMNS)

    with open(log_path, "w", encoding="utf-8") as f:
        # Redirection de la sortie standard (print) vers un fichier de log propre à chaque fichier traité.
        original_stdout = sys.stdout
        sys.stdout = f

        try:
            print(f"Traitement du fichier : {json_path}")
            start_time = time()
            
            rag_system = RefineRAGSystem(api_key=api_key, api_url=api_url) 
            rag_system.index_from_json(json_path) # Indexation des documents
            rag_system_non_invasive_detection = RAGNonInvasiveDetection(api_key=api_key, api_url=api_url) # Pour l'analyse sur l'annonce de l'invasivité

            if method in ["refine", "both"]:
                print("\n--- Méthode avec Refine ---")
                refine_answer = rag_system.refine_analysis(question, definition, top_k=top_k, title=base_name) # Lancement de l'analyse via la méthode Refine
                
                json_final = refine_answer
                resultats = parse_json(json_final, base_name)
                
                if not resultats.empty:
                    
                    print("\n--- Méthode avec Refine et detect_non_invasive_level ---")
                    # Pour chaque protocole dans les résultats, on récupère les infos utiles pour la suite
                    for idx, row in resultats.iterrows():
                        protocole = {
                            "nom": row["Protocole"],
                            "description": row["Extrait pertinent"],
                            "evaluation": row["Évaluation d'invasivité"]
                        }

                        # Partie pour l'analyse sur l'anonce de l'invasivité lorsqu'un protocole est detecté comme invasif
                        if protocole['evaluation'] == 'Invasif' or protocole['evaluation'] == 'Invasif - Territory marking':
                            try:
                                # Pour les protocoles jugés invasifs, une seconde analyse est lancée pour vérifier
                                # si l'article présente lui-même le protocole comme "non invasif".
                                invasive_json_path = os.path.join(invasive_input_dir, f"{base_name}.json")
                                # On indexe le fichier
                                rag_system_non_invasive_detection.index_from_json(invasive_json_path)
                                # Le titre de l'article est important pour la technique HyDE (Hypothetical Document Embeddings)
                                title = base_name
                                # Charge le fichier JSON contenant les chunks pour l'annonce de l'invasivité (pas le même que pour la détéctions des protocoles car contient plus de sections (Intro, abstract...))
                                with open(invasive_json_path, "r", encoding="utf-8") as f:
                                    invasive_data = json.load(f)
                                for section in invasive_data:
                                    # récupère le titre dans la section titre du JSON (pour le fournir à HyDE)
                                    if section["section"] == "title":
                                        title = section["text"]
                                        break
                                
                                # Une question spécifique est formulée pour évaluer comment le protocole est présenté dans l'article.
                                question = f"{protocole['nom'].capitalize()} constitue-t-il une méthode non invasive ou minimalement invasive pour l'obtention d'échantillons génétiques chez les animaux étudiés ?"
                                non_invasive_level = rag_system_non_invasive_detection.answer_question(
                                    question=question,
                                    protocole=str(protocole),
                                    top_k=8,
                                    title=title
                                )
                                
                                print(f"\nAnalyse du niveau non invasif pour le protocole {protocole['nom']}:")
                                print(non_invasive_level)
                                non_invasive_level = parse_json(non_invasive_level, base_name, invasive_detection=True) # Récupère le JSON extrait de la réponse du LLM
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

        except Exception as e:
            print(f"Erreur lors du traitement du fichier {json_path}: {str(e)}")
            import traceback
            print(traceback.format_exc())
        finally:
            # Retour à la sortie original
            sys.stdout = original_stdout
            print(f"Traitement terminé avec succès pour : {os.path.basename(json_path)} en {time() - start_time:.2f} secondes")

    return df_result, rag_system.impacts_energy, rag_system.impacts_co2, rag_system_non_invasive_detection.impacts_energy, rag_system_non_invasive_detection.impacts_co2

# Wrapper pour passer plusieurs arguments à process_file via ProcessPoolExecutor.
def process_file_wrapper(args):
    """
    Wrapper pour la fonction `process_file`.

    Cette fonction est nécessaire pour utiliser `ProcessPoolExecutor.submit` qui
    ne peut passer qu'un seul argument à la fonction cible. Elle prend un tuple
    d'arguments et le dépaquette pour appeler `process_file`.

    Args:
        args (tuple): Un tuple contenant tous les arguments pour `process_file`.

    Returns:
        Le résultat de l'appel à `process_file`.
    """
    return process_file(*args)

def main():
    """
    Fonction principale du script.

    Parse les arguments de la ligne de commande, prépare les tâches pour chaque
    fichier JSON à analyser, et lance le traitement en parallèle.
    Agrège ensuite les résultats et les sauvegarde dans un fichier CSV.
    """
    # Charge les variables d'environnement à partir d'un fichier .env.
    load_dotenv()
    
    # S'assure que les packages NLTK nécessaires sont téléchargés une seule fois.
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')
    
    parser = argparse.ArgumentParser(description="Système RAG pour l'analyse de documents scientifiques")
    parser.add_argument("--api_key", help="Clé API pour le LLM")
    parser.add_argument("--api_url", help="URL de l'API LLM")
    parser.add_argument("--json_file", default="", help="Traiter uniquement ce fichier JSON (optionnel)")
    parser.add_argument("--definition", default="Selon Taberlet et al. (1999), l'échantillonnage d'ADN non invasif désigne toute méthode permettant d'obtenir du matériel génétique sans avoir à capturer, blesser, ni perturber significativement l'animal. Cela inclut, par exemple, l'analyse d'échantillons laissés dans l'environnement comme les poils, les plumes, les fèces, l'urine, ou encore la salive.")
    parser.add_argument("--question", default="Dans l'article, peux-tu me donner tous les protocoles d'échantillonnage d'ADN et s'ils sont considérés comme invasifs ou non selon la définition de Taberlet ?")
    parser.add_argument("--method", choices=["refine", "standard", "both", "detect_non_invasive_level"], default="refine")
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--output_dir", default="Résultats") # Dossier pour retrouver les résultats
    parser.add_argument("--input_dir", default="output") # Dossier où les fichiers JSON sont stockés
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--invasive_input_dir", default="", help="Dossier contenant les JSON pour l'analyse d'annonce d'invasivité")
    
    args = parser.parse_args()

    # Récupération de la clé API et de l'URL depuis les arguments ou les variables d'environnement.
    api_key = args.api_key or os.getenv("LLM_API_KEY")
    api_url = args.api_url or os.getenv("LLM_API_URL", "http://gpu1.pedagogie.sandbox.univ-tours.fr:32800/api/chat/completions")

    def is_local_url(url: str) -> bool:
        if not url:
            return False
        url_lower = url.lower()
        return (
            "localhost" in url_lower
            or "127.0.0.1" in url_lower
            or "0.0.0.0" in url_lower
            or "ollama" in url_lower
        )

    if not api_key and not is_local_url(api_url):
        # Si la clé API n'est pas trouvée et que l'URL n'est pas locale, le programme s'arrête.
        print("Erreur : La clé API n'est pas fournie.", file=sys.stderr)
        print("Veuillez la passer avec --api_key ou définir la variable d'environnement LLM_API_KEY.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    normalized_input_dir = os.path.normpath(args.input_dir)
    base_input_dir = os.path.dirname(normalized_input_dir)
    if base_input_dir == "":
        base_input_dir = "."
    invasive_input_dir = args.invasive_input_dir or os.path.join(base_input_dir, "output_invasive_detection")
    if not os.path.isdir(invasive_input_dir):
        print(f"Attention : le dossier d'annonce d'invasivité {invasive_input_dir} est introuvable. Les analyses associées échoueront.")
    json_files = [f for f in os.listdir(args.input_dir) if f.endswith(".json")]

    if args.json_file:
        candidate_path = args.json_file
        if not os.path.isfile(candidate_path):
            candidate_path = os.path.join(args.input_dir, os.path.basename(args.json_file))

        if os.path.isfile(candidate_path):
            json_files = [os.path.basename(candidate_path)]
            print(f"Traitement limité au fichier : {json_files[0]}")
        else:
            print(f"Erreur : le fichier spécifié {args.json_file} est introuvable.")
            sys.exit(1)
    
    df_total = pd.DataFrame(columns=DATAFRAME_COLUMNS)

    # Variables pour suivre les impacts totaux
    total_energy = []
    total_co2 = []
    total_energy_non_invasive = []
    total_co2_non_invasive = []

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
                args.output_dir,
                invasive_input_dir
            ))

    print(f"Type de la clé dans tasks : {type(tasks[0][0])}")

    results_list = []
    # Utilisation de ProcessPoolExecutor pour traiter les fichiers en parallèle.
    # Chaque fichier est traité dans un processus distinct
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(process_file_wrapper, task) for task in tasks]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Fichiers traités"):
            try:
                df_result, impacts_energy, impacts_co2, impacts_energy_non_invasive, impacts_co2_non_invasive = future.result()
                if not df_result.empty:
                    results_list.append(df_result)

                # Ajouter les impacts aux totaux
                total_energy.extend(impacts_energy)
                total_co2.extend(impacts_co2)
                total_energy_non_invasive.extend(impacts_energy_non_invasive)
                total_co2_non_invasive.extend(impacts_co2_non_invasive)

                # Sauvegarde intermédiaire des résultats cumulés.
                if results_list:
                    pd.concat(results_list, ignore_index=True).to_csv(os.path.join(args.output_dir, "Protocoles_intermediaire.csv"), index=False)

            except Exception as e:
                print(f"Erreur dans un traitement parallèle : {e}")

    if results_list:
        df_total = pd.concat(results_list, ignore_index=True)

    # Sauvegarde finale
    if not df_total.empty:
        df_total.to_csv(os.path.join(args.output_dir, "Protocoles.csv"), index=False)
        print("Traitement terminé. Résultats sauvegardés dans 'Protocoles.csv'.")
        
        # Afficher les impacts totaux
        total_energy_all = sum(total_energy) + sum(total_energy_non_invasive)
        total_co2_all = sum(total_co2) + sum(total_co2_non_invasive)
        print(f"\n🌱 Énergie totale consommée : {total_energy_all:.4f} kWh")
        print(f"🌍 CO₂ total émis : {total_co2_all:.2f} g")
        
        return df_total
    else:
        print("Attention: Aucun résultat n'a été généré.")
        return None

if __name__ == "__main__":
    main()



 
                            

            
