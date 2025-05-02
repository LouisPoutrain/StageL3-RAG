"""
Point d'entrée principal pour le système RAG
"""

import argparse
from rag_system import RAGSystem


def main():
    parser = argparse.ArgumentParser(description="Système RAG pour l'analyse de documents scientifiques")
    parser.add_argument("--api_key", help="Clé API pour le LLM")
    parser.add_argument("--api_url", help="URL de l'API LLM")
    parser.add_argument("--json_file", default="output/Molecular-Ecology.json", help="Fichier JSON de données à indexer")
    parser.add_argument("--definition", default="Selon Taberlet et al. (1999), l'échantillonnage d'ADN non invasif désigne toute méthode permettant d'obtenir du matériel génétique sans avoir à capturer, blesser, ni perturber significativement l'animal. Cela inclut, par exemple, l'analyse d'échantillons laissés dans l'environnement comme les poils, les plumes, les fèces, l'urine, ou encore la salive.", 
                        help="Définition de l'échantillonnage non-invasif")
    parser.add_argument("--question", default="Dans l'article, peux-tu me donner tous les protocoles d'échantillonnage d'ADN et si ils sont considérés comme invasif ou non-invasif selon la définition de Taberlet ?", 
                        help="Question à poser au système")
    parser.add_argument("--method", choices=["hyde", "standard", "both"], default="both", 
                        help="Méthode de récupération à utiliser (hyde, standard, both)")
    parser.add_argument("--top_k", type=int, default=4, 
                        help="Nombre de documents à récupérer")
    
    args = parser.parse_args()
    
    # Initialisation du système RAG
    rag_system = RAGSystem(
        api_key=args.api_key or "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjE0NWYwMDBhLTVhMDYtNDAxMS1hZTkzLWYwYTk5MzMzNWYzZCJ9.omzpNc36f2QVJk2OAErLnBU2_kQbqPHfrE0vt4f8ruQ",
        api_url=args.api_url or "http://gpu1.pedagogie.sandbox.univ-tours.fr:32800/api/chat/completions"
    )
    
    # Indexation des documents
    rag_system.index_from_json(args.json_file)
    
    # Traitement selon la méthode choisie
    if args.method in ["hyde", "both"]:
        print("\n--- Méthode avec HyDE ---")
        hyde_answer = rag_system.answer_question(args.question, args.definition, use_hyde=True, top_k=args.top_k)
        print("\nRéponse avec HyDE:")
        print(hyde_answer)
    print("\n--- Méthode standard ---")
    standard_answer = rag_system.answer_question(args.question, args.definition, use_hyde=False)
    print("\nRéponse standard:")
    print(standard_answer)


main()
