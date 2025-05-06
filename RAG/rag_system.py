"""
Système RAG principal
"""

import os
import json
from typing import List, Dict, Any, Optional
import pandas as pd
import re

from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.components.preprocessors import DocumentSplitter

from UniversityLLMAdapter import UniversityLLMAdapter
from pipelines import create_hyde_pipeline, create_rag_pipeline, create_indexing_pipeline


class RAGSystem:
    """
    Système RAG complet avec pipelines HyDE et standard
    """
    
    def __init__(self, api_key=None, api_url=None):
        """
        Initialise le système RAG avec tous ses composants
        
        Args:
            api_key: Clé API pour le LLM (optionnel si défini dans l'environnement)
            api_url: URL de l'API LLM (optionnel si défini dans l'environnement)
        """
        self.api_key = api_key or os.environ.get("UNIVERSITY_LLM_API_KEY")
        self.api_url = api_url or os.environ.get("UNIVERSITY_LLM_API_URL")
        
        if not self.api_key or not self.api_url:
            raise ValueError("API key and URL must be provided or set as environment variables")
            
        # Initialisation du store de documents
        self.document_store = InMemoryDocumentStore()
        
        # Initialisation des embedders 
        self.embedder_model = "sentence-transformers/allenai-specter"
        self.text_embedder = SentenceTransformersTextEmbedder(model=self.embedder_model)
        self.text_embedder.warm_up()
        
        # Initialisation du retriever et du writer
        self.retriever = InMemoryEmbeddingRetriever(document_store=self.document_store)
        self.writer = DocumentWriter(document_store=self.document_store)
        
        # Initialisation du reranker
        self.reranker = TransformersSimilarityRanker(model="cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.reranker.warm_up()

        # Initialisation du splitter
        self.splitter = DocumentSplitter(split_by="word", split_length=500, split_overlap=50)

        # Initialisation des pipelines
        self.hyde_pipeline = create_hyde_pipeline(self.api_key, self.api_url, self.embedder_model)
        self.rag_pipeline = create_rag_pipeline(self.text_embedder, self.retriever, self.splitter)
        
        # Création de l'adaptateur LLM pour la génération de réponses
        self.llm_adapter = UniversityLLMAdapter(
            api_key=self.api_key,
            api_url=self.api_url,
            max_tokens=1024,
            temperature=0.1
        )

    def index_from_json(self, json_path: str) -> int:
        """
        Indexe des documents à partir d'un fichier JSON
        
        Args:
            json_path: Chemin vers le fichier JSON
            
        Returns:
            Nombre de documents indexés
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        documents = []
        for entry in data:
            document = Document(
                content=entry.get("text", "")
            )
            documents.append(document)
            print(document)

        if isinstance(documents, list):
            print(f"Documents est une liste avec {len(documents)} éléments.")
        else:
            print(f"Documents n'est pas une liste, c'est un {type(documents)}.")

        # Création de la pipeline d'indexation
        indexing_pipeline = create_indexing_pipeline(self.document_store, self.embedder_model)
        
        # Exécution de la pipeline
        indexing_pipeline.run({"doc_embedder": {"documents": documents}})

        # Vérification des embeddings
        indexed_docs = self.document_store.filter_documents()
        for doc in indexed_docs:
            print(f"Document ID: {doc.id}, Embedding: {doc.embedding is not None}")

        print(f"{len(documents)} documents indexés depuis {json_path}")
        return len(documents)

    def retrieve_with_hyde(self, question: str, top_k: int = 4) -> List[Document]:
        """
        Récupère les documents pertinents en utilisant HyDE
        
        Args:
            question: La question de l'utilisateur
            top_k: Nombre de documents à récupérer
            
        Returns:
            Liste des documents récupérés
        """

        print("debut de retrieve with hyde")
        # Génération et embedding du document hypothétique avec HyDE
        hyde_output = self.hyde_pipeline.run({
            "prompt_builder": {"question": question},
            "generator": {"n_generations": 3}  # Génère 3 documents hypothétiques
        })
        
        # Log pour le debugging
        print("Clés disponibles dans hyde_output:", list(hyde_output.keys()))
        for key in hyde_output:
            print(f"Structure de {key}:", list(hyde_output[key].keys()) if isinstance(hyde_output[key], dict) else "Non dictionnaire")
        
        # Récupération de l'embedding hypothétique moyen
        hyp_embedding = hyde_output["hyde"]["hypothetical_embedding"]
        
        # Récupération des documents avec l'embedding hypothétique
        retrieved_docs = self.retriever.run(
            query_embedding=hyp_embedding, 
            top_k=top_k,
        )["documents"]
        

        # Affichage des documents hypothétiques générés
        print("\nDocuments hypothétiques générés:")
        if "generator" in hyde_output and "documents" in hyde_output["generator"]:
            for i, doc in enumerate(hyde_output["generator"]["documents"]):
                print(f"Document {i+1}: {doc.content[:100]}...")
        
        return retrieved_docs
    
    def retrieve_standard(self, question: str, top_k: int = 4) -> List[Document]:
        """
        Récupère les documents pertinents avec la méthode standard
        
        Args:
            question: La question de l'utilisateur
            top_k: Nombre de documents à récupérer
            
        Returns:
            Liste des documents récupérés
        """
        query_embedding = self.text_embedder.run(text=question)["embedding"]
        retrieved_docs = self.retriever.run(
            query_embedding=query_embedding, 
            top_k=top_k
        )["documents"]
        
        return retrieved_docs
    
    def build_context(self, documents: List[Document], max_length: int = 6000) -> str:
        context_parts = []
        total_length = 0

        for i, doc in enumerate(documents):
            content = doc.content.strip()
            excerpt_header = f"EXTRAIT {i+1}:\n"
            available_length = max_length - total_length - len(excerpt_header)

            if available_length <= 0:
                break

            # Si le contenu est plus court que la limite dispo, pas besoin de tronquer
            if len(content) <= available_length:
                trimmed_content = content
            else:
                # Essaye de couper à la dernière phrase complète
                sentences = re.split(r'(?<=[.!?])\s+', content)
                trimmed_content = ''
                for sentence in sentences:
                    if len(trimmed_content) + len(sentence) + 1 > available_length:
                        break
                    trimmed_content += sentence + ' '
                trimmed_content = trimmed_content.strip()

                # Si aucune phrase complète ne rentre, on coupe à la fin du dernier mot complet
                if not trimmed_content:
                    cutoff = content[:available_length]
                    last_space = cutoff.rfind(' ')
                    trimmed_content = cutoff[:last_space] if last_space != -1 else cutoff

            excerpt = excerpt_header + trimmed_content
            context_parts.append(excerpt)
            total_length += len(excerpt)

            if total_length >= max_length:
                break

        print("CONTEXTE ----------")
        print("\n\n".join(context_parts))

        return "\n\n".join(context_parts)
    
    def _build_prompt(self, question: str, context: str, definition: str = "") -> str:
        """
        Construit le prompt pour le LLM
        
        Args:
            question: La question de l'utilisateur
            context: Le contexte extrait des documents
            definition: Éventuellement une définition à inclure
            
        Returns:
            Le prompt complet
        """
        seven_sins_definitions = """
        Voici les définitions des Péchés de l'Échantillonnage d'ADN Non-Invasif :
        1. Mauvaise classification des fèces comme échantillons d'ADN non invasifs : Considérer automatiquement la collecte de fèces comme non invasive sans tenir compte du contexte de la collecte (par exemple, capture de l'animal pour obtenir les fèces, utilisation d'aéronefs pouvant stresser les animaux, impact sur le marquage territorial).
        2. Appâtage des pièges à ADN : Utiliser des appâts ou des leurres pour augmenter le rendement des pièges à ADN, ce qui modifie le comportement naturel des animaux et ne correspond pas à une méthode entièrement non invasive.
        3. Un oiseau dans la main vaut mieux que deux dans la nature (Capture ou manipulation d'animaux) : Capturer et/ou manipuler des animaux sauvages pour obtenir des échantillons d'ADN, ce qui cause du stress et potentiellement des blessures, contrairement à la définition de l'échantillonnage non invasif.
        4. Tout ou rien (Manque de reconnaissance des approches minimalement invasives) : Ne pas reconnaître ou utiliser le terme "minimalement invasif" pour des méthodes qui réduisent l'impact sur l'animal, car la définition stricte de "non invasif" ne laisse pas de place au milieu.
        5. Équivalence entre une procédure non invasive et un échantillonnage d'ADN non invasif : Utiliser la définition médicale ou vétérinaire d'une procédure non invasive (qui n'implique pas de perforation de la peau) pour classer l'échantillonnage d'ADN, sans tenir compte de l'impact comportemental ou du bien-être animal.
        """

        return f"""
        Tu es un assistant scientifique expert dans l'analyse des méthodes d'échantillonnage d'ADN. Ta tâche est d'examiner les protocoles décrits dans les extraits ci-dessous et de déterminer s'ils contreviennent aux "Sept Péchés de l'Échantillonnage d'ADN Non-Invasif".

        ##################################################
        # DÉFINITIONS ET CONTEXTE
        ##################################################

        Définition de l'échantillonnage non-invasif :
            {definition}

        Définitions des Sept Péchés :
        {seven_sins_definitions}

                 Important :
            1. Toute action qui modifie le comportement de l'animal (fuite, appât, stress...) doit être considérée comme invasive.
            2. Tout contact direct avec l'animal (manipulation, toucher, prélèvement de salive/sang/poils...) est TOUJOURS considéré comme invasif.
            3. Toute capture, même momentanée, est TOUJOURS considérée comme invasive.
            4. Un échantillonnage n'est non invasif QUE s'il est effectué sans aucun contact avec l'animal (ex: collecte de poils/plumes tombés naturellement, fèces trouvées dans l'environnement sans perturber l'animal).
            5. Un prélèvement de fèces est considéré comme invasif si l'animal est perturbé (ex: en utilisant un aéronef pour les collecter, ou s'il s'agit d'une espèce qui marque son territoire avec ses fèces).

        ##################################################
        # CONTEXTE À ANALYSER
        ##################################################

        {context}

        ##################################################
        # INSTRUCTIONS D'ANALYSE
        ##################################################

        1. Identifie UNIQUEMENT les protocoles d'échantillonnage d'ADN distincts mentionnés dans le texte.
        2. Pour chaque protocole, analyse :
        - La méthode d'obtention de l'ADN et la partie de l'animal concernée.
        - La présence ou l'absence de manipulation, capture ou perturbation directe de l'animal.
        - Les impacts potentiels sur le comportement ou le bien-être animal.
        - Le niveau d'invasivité selon la définition fournie (Non invasif/Invasif).
        - Les numéros des péchés transgressés.
        - Si de nouveaux péchés sont identifiés, indique "Oui" ou "Non" et formule brièvement la justification.

        3. Priorise les protocoles invasifs. Si un protocole non invasif est trouvé en premier, continue à chercher un protocole invasif.
        4. Fournis un seul protocole par article. Si aucun protocole invasif n'est trouvé, affiche le protocole non invasif.
        5. Analyse UNIQUEMENT les protocoles d'échantillonnage d'ADN. Ignore les autres méthodes (ex: PCR, séquençage, etc.).

        ##################################################
        # FORMAT DE RÉPONSE
        ##################################################

        Pour chaque protocole identifié, présente l'analyse dans ce format JSON, un protocole = un objet JSON :

        ```json
        {{
            "protocole": "[Nom concis du protocole]",
            "extrait_pertinent": ["Texte exact de chaque extrait utilisé, entre guillemets."],
            "impacts_potentiels": "[Comportement, stress, douleur, si mentionné]",
            "evaluation_invasivite": "[Non invasif / Invasif]",
            "peches_identifies": ["1", "2", "7"],
            "nouveaux_peches": "[Oui / Non – Si oui, formuler brièvement]"
        }}
        ```

        ##################################################
        VÉRIFICATION DU JSON

        ##################################################

        Avant de soumettre ton JSON, vérifie :

            Qu'il n'y a pas de clés dupliquées.
            Que toutes les accolades et crochets sont bien fermés.
            Que toutes les virgules sont correctement placées.
            Que le JSON est parfaitement valide et pourrait être parsé sans erreur.
            Que SI tu souhaite faire une liste, elle commence par [ et se termine par ].

        RÈGLES CRUCIALES POUR LE FORMAT JSON :

            Chaque clé doit apparaître UNE SEULE FOIS dans l'objet JSON.
            Les noms de clés doivent utiliser des underscores simples (_).
            Toute liste doit commencer par [ et se terminer par ].
            L'objet justification_peches doit commencer par {{ et se terminer par }}.
            Chaque élément dans un objet ou une liste doit être séparé par une virgule.
            N'utilise pas d'accents dans les noms de clés.
            Assure-toi que chaque valeur est du bon type : chaînes entre guillemets, listes entre crochets, objets entre accolades.
            Assure-toi que le champ extrait_pertinent contient toujours une liste de chaînes, même s'il n'y a qu'un seul élément. 


        IMPORTANT : 
            JE VEUX QUE TU AFFICHES UNIQUEMENT LES PROTOCOLES D'ECHANTILLONNAGE d'ADN.
            TU NE DOIS PAS ECRIRE PLUSIEURS FOIS LE MÊME PROTOCOLE.

        """

    def generate_answer(self, question: str, context: str, definition: str = "") -> str:
        """
        Génère une réponse à partir du contexte et de la question
        
        Args:
            question: La question de l'utilisateur
            context: Le contexte extrait des documents
            definition: Éventuellement une définition à inclure
            
        Returns:
            La réponse générée
        """
        prompt = self._build_prompt(question, context, definition)

        return self.llm_adapter.generate_answer(prompt)
    

    def answer_question(self, question: str, definition: str = "", use_hyde: bool = True, top_k: int = 8) -> str:
        """
        Processus complet pour répondre à une question (sans reranker)
        """
        if use_hyde:
            print("--- Méthode avec HyDE ---")
            retrieved_docs = self.retrieve_with_hyde(question=question, top_k=top_k)
            print("\n--- DOCUMENTS RÉCUPÉRÉS (HYDE) ---")
            for doc in retrieved_docs:
                print(f"Contenu: {doc.content[:200]}...")
            print("--- FIN DES DOCUMENTS RÉCUPÉRÉS (HYDE) ---\n")

        else:
            print("--- Méthode standard ---")
            results = self.rag_pipeline.run(data={"text_embedder": {"text": question}, "retriever": {"top_k": top_k}})
            print(f"\n--- RÉSULTATS DE RAG PIPELINE (STANDARD) ---")
            print(results)
            print("--- FIN DES RÉSULTATS DE RAG PIPELINE (STANDARD) ---\n")
            chunked_docs = results["splitter"]["documents"] # Les documents sont déjà chunkés ici
            print(f"Chunks récupérés avec la méthode standard: {len(chunked_docs)}")
            retrieved_docs = chunked_docs # Pour que le reste de la fonction fonctionne sans modification

        # Affichage des documents récupérés (qui sont maintenant les chunks pour la méthode standard)
        print("\n--- DOCUMENTS RÉCUPÉRÉS (OU CHUNKS) ---")
        for i, doc in enumerate(retrieved_docs):
            print(f"\nDocument/Chunk {i+1}:")
            print(f"Score: {doc.score if hasattr(doc, 'score') else 'N/A'}")

            print(f"Contenu: {doc.content[:100]}...") #  aperçu

        # Construction du contexte
        context = self.build_context(retrieved_docs) 

        # Génération de la réponse
        answer = self.generate_answer(question, context, definition)

        return answer


expected_keys = {
    "protocole", "extrait_pertinent",
    "impacts_potentiels", "evaluation_invasivite",
    "peches_identifies", "nouveaux_peches"
}

def is_valid_entry(entry):
    """Vérifie que toutes les clés attendues sont présentes dans l'entrée"""
    return isinstance(entry, dict) and expected_keys.issubset(entry.keys())

def extract_json_blocks(text):
    """Extrait tous les blocs JSON entourés de balises ```json ... ```"""
    matches = re.findall(r"```json(.*?)```", text, re.DOTALL)
    return [m.strip() for m in matches]

def safe_load_json(text):
    """Charge un bloc JSON, filtre les objets malformés, conserve uniquement les clés attendues"""
    try:
        obj = json.loads(text)
        result = []

        if isinstance(obj, dict):
            if is_valid_entry(obj):
                result.append({k: v for k, v in obj.items() if k in expected_keys})

        elif isinstance(obj, list):
            for entry in obj:
                if is_valid_entry(entry):
                    result.append({k: v for k, v in entry.items() if k in expected_keys})

        return result
    except json.JSONDecodeError:
        return []

def parse_json(response_text, filename):
    """Parse le texte de réponse en extrayant les blocs JSON valides"""
    json_blocks = extract_json_blocks(response_text)
    parsed_blocks = []

    for block in json_blocks:
        cleaned_jsons = safe_load_json(block)
        parsed_blocks.extend(cleaned_jsons)

    if not parsed_blocks:
        return pd.DataFrame()

    df = pd.DataFrame([
        {
            "Filename": filename,
            "Protocole": item["protocole"],
            "Extrait pertinent": item["extrait_pertinent"],
            "Évaluation d'invasivité": item["evaluation_invasivite"],
            "Péchés identifiés": item["peches_identifies"],
            "Nouveaux péchés": item["nouveaux_peches"]
        }
        for item in parsed_blocks
    ])

    return df

    
def print_markdown_table(tableau):
    if not tableau:
        print("Aucun tableau récapitulatif trouvé.")
        return

    # Imprimer l'en-tête du tableau
    print("| Protocole | Extrait | Statut | Péchés | Nouveaux Péchés |")
    print("|-----------|---------|--------|--------|----------------|")

    # Imprimer chaque ligne du tableau
    for entry in tableau:
        print(f"| {entry['protocole']} | {entry['extrait']} | {entry['statut']} | {entry['peches']} | {entry['nouveaux_peches']} |")
