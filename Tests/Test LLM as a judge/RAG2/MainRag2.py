import argparse
import sys
import os
import pandas as pd
from rag_system2 import RAGSystem

def process_file(api_key, api_url, json_path, definition, question, method, top_k, output_dir, df):
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    log_path = os.path.join(output_dir, f"{base_name}.txt")

    with open(log_path, "w", encoding="utf-8") as f:
        original_stdout = sys.stdout
        sys.stdout = f

        print(f"Traitement du fichier : {json_path}")

        rag_system = RAGSystem(api_key=api_key, api_url=api_url)
        rag_system.index_from_json(json_path)

        hyde_answer = ""
        if method in ["hyde", "both"]:
            print("\n--- Méthode avec HyDE ---")
            hyde_answer = rag_system.answer_question(question, definition, use_hyde=True, top_k=top_k)
            print("\nRéponse avec HyDE:")
            print(hyde_answer)

        print("Données parsées:")

        response_to_parse = hyde_answer
        parsed_data = rag_system.parse_animal_metadata_response(response_to_parse)

        print("Parsed Data:", parsed_data)

        df = df[df['Filename'] != base_name]

        for entry in parsed_data:
            if entry:
                new_row = pd.DataFrame({
                    'Filename': [base_name],
                    'Animal': [entry.get('animal', '')],
                    'Famille': [entry.get('famille', '')],
                    'Invert/Vert' : [entry.get('Invert/Vert','')],
                    'Pays': [entry.get('pays', '')],
                    'Extrait': [entry.get('source', '')]
                })
                df = pd.concat([df, new_row], ignore_index=True)

        sys.stdout = original_stdout

    return df


def main():
    parser = argparse.ArgumentParser(description="Extraction d'informations sur les animaux dans des articles scientifiques.")
    parser.add_argument("--api_key", help="Clé API pour le LLM")
    parser.add_argument("--api_url", help="URL de l'API LLM")
    parser.add_argument("--json_file", default="output/s11356-015-5754-2.json", help="Fichier JSON de données à indexer")
    parser.add_argument("--definition", default="", help="Définition à utiliser pour guider l'interprétation (optionnel)")
    parser.add_argument("--question", default="Quels animaux et pays sont mentionnés dans ces textes ?", help="Question à poser au système")
    parser.add_argument("--method", choices=["hyde", "standard", "both"], default="both", help="Méthode de récupération à utiliser (hyde, standard, both)")
    parser.add_argument("--top_k", type=int, default=4, help="Nombre de documents à récupérer")
    parser.add_argument("--output_dir", default="Résultats", help="Dossier de sortie pour les fichiers de log")
    parser.add_argument("--input_dir", default="output2", help="Dossier contenant les fichiers JSON à traiter")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.DataFrame(columns=['Filename', 'Animal', 'Famille', 'Invert/Vert', 'Pays', 'Extrait'])

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

    df = df.drop_duplicates(subset=['Filename', 'Animal', 'Famille', 'Invert/Vert', 'Pays', 'Extrait'])
    df = df.reset_index(drop=True)

    print("Contenu du DataFrame avant enregistrement :")
    print(df)

    df.to_excel(os.path.join(args.output_dir, 'parsed_animals.xlsx'), index=False)

    print("Données parsées et enregistrées dans 'parsed_animals.xlsx'")

if __name__ == "__main__":
    main()
