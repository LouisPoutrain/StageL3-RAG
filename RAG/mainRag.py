import argparse
import sys
import os
import pandas as pd
from rag_system import RAGSystem, parse_json

def process_file(api_key, api_url, json_path, definition, question, method, top_k, output_dir, df):
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    log_path = os.path.join(output_dir, f"{base_name}.txt")

    with open(log_path, "w", encoding="utf-8") as f:
        original_stdout = sys.stdout  # Sauvegarder la sortie standard originale
        sys.stdout = f

        try:
            print(f"Traitement du fichier : {json_path}")

            # Nouvelle instance de RAGSystem pour garantir isolation
            rag_system = RAGSystem(
                api_key=api_key,
                api_url=api_url
            )

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

            # Vérifier le contenu du DataFrame resultats
            print("Contenu du DataFrame resultats:")
            print(resultats)

            # Vérifier si les colonnes attendues sont présentes
            expected_columns = ["Filename", "Protocole", "Extrait pertinent", "Évaluation d'invasivité", "Péchés identifiés", "Nouveaux péchés"]
            if not all(col in resultats.columns for col in expected_columns):
                print(f"Erreur : Les colonnes attendues {expected_columns} ne sont pas présentes dans le DataFrame.")
                print(f"Colonnes présentes : {resultats.columns}")
            else:
                print(resultats[expected_columns])
                df = pd.concat([df, resultats], ignore_index=True)
        finally:
            sys.stdout = original_stdout

    return df

        

def main():
    parser = argparse.ArgumentParser(description="Système RAG pour l'analyse de documents scientifiques")
    parser.add_argument("--api_key", help="Clé API pour le LLM")
    parser.add_argument("--api_url", help="URL de l'API LLM")
    parser.add_argument("--json_file", default="output/s11356-015-5754-2.json", help="Fichier JSON de données à indexer")
    parser.add_argument("--definition", default="Selon Taberlet et al. (1999), l'échantillonnage d'ADN non invasif désigne toute méthode permettant d'obtenir du matériel génétique sans avoir à capturer, blesser, ni perturber significativement l'animal. Cela inclut, par exemple, l'analyse d'échantillons laissés dans l'environnement comme les poils, les plumes, les fèces, l'urine, ou encore la salive.",
                        help="Définition de l'échantillonnage non-invasif")
    parser.add_argument("--question", default="Dans l'article, peux-tu me donner tous les protocoles d'échantillonnage d'ADN et si ils sont considérés comme invasif ou non-invasif selon la définition de Taberlet ?",
                        help="Question à poser au système")
    parser.add_argument("--method", choices=["hyde", "standard", "both"], default="hyde",
                        help="Méthode de récupération à utiliser (hyde, standard, both)")
    parser.add_argument("--top_k", type=int, default=4,
                        help="Nombre de documents à récupérer")
    parser.add_argument("--output_dir", default="Résultats", help="Dossier de sortie pour les fichiers de log")
    parser.add_argument("--input_dir", default="output", help="Dossier contenant les fichiers JSON à traiter")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialiser le DataFrame avec les colonnes souhaitées
    df = pd.DataFrame(columns=[
        'Filename', 'Protocole', 'Extrait pertinent', 'Évaluation d\'invasivité', 
        'Péchés identifiés', 'Nouveaux péchés'
    ])

    for filename in os.listdir(args.input_dir):
        if filename.endswith(".json"):
            json_path = os.path.join(args.input_dir, filename)
            print(f"Traitement du fichier : {json_path}") 
            df = process_file(
                api_key=args.api_key or "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6ImRjZjZkYWE2LTRjYzMtNDYyOS05MjJiLTkyYzM1NGQzODYwMCJ9.04oTM2nVW8iQTCr8qrs4MknI13dGgqBp85Wq7t2jAeQ",
                api_url=args.api_url or "http://gpu1.pedagogie.sandbox.univ-tours.fr:32800/api/chat/completions",
                json_path=json_path,
                definition=args.definition,
                question=args.question,
                method=args.method,
                top_k=args.top_k,
                output_dir=args.output_dir,
                df=df
            )

    output_csv_path = os.path.join(args.output_dir, "tous_protocoles.csv")
    df.to_csv(output_csv_path, index=False)
    print(f"Résultats consolidés sauvegardés dans : {output_csv_path}")
    

if __name__ == "__main__":
    main()
