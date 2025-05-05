import argparse
import sys
import os
import pandas as pd
from rag_system import RAGSystem, print_markdown_table

def process_file(api_key, api_url, json_path, definition, question, method, top_k, output_dir, df):
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    log_path = os.path.join(output_dir, f"{base_name}.txt")

    with open(log_path, "w", encoding="utf-8") as f:
        original_stdout = sys.stdout  # Sauvegarder la sortie standard originale
        sys.stdout = f

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

        # Utilise la réponse HyDE si disponible, sinon la réponse standard
        response_to_parse = hyde_answer if hyde_answer else standard_answer
        parsed_data = rag_system.parse_response(response_to_parse)

        print("Parsed Data:", parsed_data)

        # Supprimer les entrées précédentes de ce fichier pour éviter les doublons
        df = df[df['Filename'] != base_name]

        # Conserver les protocoles déjà traités pour éviter les doublons
        processed_protocols = set()

        # Traiter d'abord les protocoles qui sont plus complets
        protocols = parsed_data.get("protocols", [])
        for protocol in protocols:
            print(f"\nProtocole: {protocol['nom']}")
            print(f"Status: {protocol['invasivite']}")
            print(f"Péchés identifiés: {', '.join(protocol['peches']) if isinstance(protocol['peches'], list) else protocol['peches']}")
            print(f"Nouveaux péchés: {protocol.get('nouveaux_peches', '')}")
            print(f"Extrait: {protocol['extrait']}")
            print(f"---")

            # Vérifier que le nom du protocole n'est pas juste une ligne de séparation
            if protocol['nom'] and not protocol['nom'].strip().startswith('---'):

                processed_protocols.add(protocol['nom'])

                # Convertir la liste de péchés en chaîne si nécessaire
                peches_str = ', '.join(protocol['peches']) if isinstance(protocol['peches'], list) else protocol['peches']

                new_row = pd.DataFrame({
                    'Filename': [base_name],
                    'Protocol': [protocol['nom']],
                    'Extrait': [protocol['extrait']],
                    'Statut': [protocol['invasivite']],
                    'Peches': [peches_str],
                    'Nouveaux_peches': [protocol.get('nouveaux_peches', '')]
                })
                df = pd.concat([df, new_row], ignore_index=True)

        tableau = parsed_data.get("tableau", [])
        for entry in tableau:
            # Vérifier si ce protocole n'a pas déjà été traité
            if entry['protocole'] not in processed_protocols and entry['protocole'] and not entry['protocole'].strip().startswith('---'):
                new_row = pd.DataFrame({
                    'Filename': [base_name],
                    'Protocol': [entry['protocole']],
                    'Extrait': [entry['extrait']],
                    'Statut': [entry['statut']],
                    'Peches': [entry['peches']],
                    'Nouveaux_peches': [entry.get('nouveaux_peches', '')]
                })
                df = pd.concat([df, new_row], ignore_index=True)


        sys.stdout = original_stdout  # Restaurer la sortie standard originale

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
    parser.add_argument("--method", choices=["hyde", "standard", "both"], default="both",
                        help="Méthode de récupération à utiliser (hyde, standard, both)")
    parser.add_argument("--top_k", type=int, default=4,
                        help="Nombre de documents à récupérer")
    parser.add_argument("--output_dir", default="Résultats", help="Dossier de sortie pour les fichiers de log")
    parser.add_argument("--input_dir", default="output", help="Dossier contenant les fichiers JSON à traiter")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialiser le DataFrame avec les colonnes souhaitées
    df = pd.DataFrame(columns=['Filename', 'Protocol', 'Extrait', 'Statut', 'Peches', 'Nouveaux_peches'])

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

    # Éliminer les lignes où Protocol ne contient que des caractères de séparation
    df = df[~df['Protocol'].str.match(r'^[-\s]*$')]

    # Éliminer les doublons potentiels
    df = df.drop_duplicates(subset=['Filename', 'Protocol'])

    # Réinitialiser les index
    df = df.reset_index(drop=True)

    # Vérifier le contenu du DataFrame avant de l'enregistrer
    print("Contenu du DataFrame avant enregistrement :")
    print(df)

    # Enregistrer le DataFrame dans un fichier Excel
    df.to_excel(os.path.join(args.output_dir, 'parsed_data.xlsx'), index=False)

    print("Données parsées et enregistrées dans 'parsed_data.xlsx'")

if __name__ == "__main__":
    main()
