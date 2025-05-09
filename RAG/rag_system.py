"""
Système RAG principal
"""

import os
import json
from typing import List, Dict, Any, Optional
import pandas as pd
import re
import numpy as np

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
            api_key: Clé API pour le LLM 
            api_url: URL de l'API LLM 
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
            max_tokens=10000,
            temperature=0.1
        )

    def generate_answer_with_override(self, question: str, context: str, definition: str = "", prompt_override: str = None) -> str:
        """
        Génère une réponse à partir du contexte et de la question
        """
        if prompt_override:
            prompt = prompt_override
        else:
            prompt = self._build_prompt(question, context, definition)

        return self.llm_adapter.generate_answer(prompt)

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
    
    def build_context(self, documents: List[Document], max_length: int = 3000) -> str:
        context_parts = []
        total_length = 0

        for i, doc in enumerate(documents): # itération sur les documents
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

        print("DEBUT  CONTEXTE ----------")
        print("\n\n".join(context_parts)[:200])
        print("FIN  CONTEXTE ----------")
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
        Tu es un assistant scientifique expert dans l'analyse des méthodes d'échantillonnage d'ADN. Ta tâche est d'examiner les protocoles décrits dans les extraits pour déterminer s'ils contreviennent aux "Sept Péchés de l'Échantillonnage d'ADN Non-Invasif".

        Tu dois suivre strictement le format REACT (Reasoning and Acting) en séparant clairement chaque étape :

        ## Thought: [Réflexion sur le protocole identifié]
        ## Act: [Analyse détaillée du protocole]
        ## Obs: [Citations exactes des extraits]

        ##################################################
        # DÉFINITIONS ET CONTEXTE
        ##################################################

        Définition de l'échantillonnage non-invasif :
        {definition}

        definition des péchés :
        {seven_sins_definitions}

        Important :
        1. Toute action modifiant le comportement animal = invasive
        2. Tout contact direct avec l'animal = TOUJOURS invasif
        3. Toute capture = TOUJOURS invasive
        4. Échantillonnage non invasif = UNIQUEMENT sans contact avec l'animal
        5. Prélèvement de fèces = invasif SEULEMENT si perturbation de l'animal

        ##################################################
        # RÈGLES SUPPLÉMENTAIRES D'INTERPRÉTATION
        ##################################################

        1. **Radio-suivi** :
        - Si l'animal est localisé grâce à un marquage radio **réalisé avant le protocole**, ce n'est **pas invasif**.
        - Si le marquage est **effectué dans le cadre du protocole**, il est **invasif**.

        2. **Prélèvement de fèces de mammifères** :
        - Si **toutes les fèces** sont prélevées : **invasif** (marquage territorial).
        - Si **seule une partie** est prélevée : **non invasif**.
        - Si aucune précision n'est donnée (ex : "fèces collectées après le passage de l'animal") :
            → `"evaluation_invasivite": "Inconnu"`
        - Si le contexte de la collecte des fèces implique une perturbation de l'animal (e.g., utilisation d'aéronefs, capture de l'animal), même sans précision sur la quantité, alors :
            → `"evaluation_invasivite": "Invasif"`

        3. **Autres prélèvements (poils, salive, urine...)** :
        - Si les informations sont **insuffisantes** (aucune mention de capture, de manipulation, etc.) :
            → `"evaluation_invasivite": "présumé non invasif"`
            avec justification adaptée.

        ##################################################
        # CONTEXTE À ANALYSER
        ##################################################

        {context}

        ##################################################
        # INSTRUCTIONS D'ANALYSE — FORMAT REACT
        ##################################################

        1. Identifie UNIQUEMENT les protocoles d'échantillonnage d'ADN distincts.
        2. **Regroupe IMPÉRATIVEMENT TOUS les protocoles qui partagent des extraits similaires ou qui décrivent des méthodes d'échantillonnage essentiellement identiques, même s'ils sont mentionnés dans différentes parties du texte. Considère TOUTES les variations mineures (e.g., "poils" vs. "poils collectés dans la nature", "fèces" vs. "scat") comme faisant partie du même protocole général. Ne crée PAS de protocoles distincts pour ces variations. Si un échantillon est collecté à la fois en captivité et dans la nature, considère cela comme UN SEUL protocole et indique "mixte" pour l'échantillon. PRIORISE le regroupement au détriment de la séparation.**
        3. Pour chaque protocole regroupé :
        - **Thought** : réfléchis à ce que tu observes, identifie une action ou une méthode liée à l'échantillonnage.
        - **Act** : Si pertinent, analyse :
            - Méthode d'obtention de l'ADN et partie de l'animal concernée.
            - Présence ou absence de manipulation, capture, ou perturbation.
            - Impacts potentiels (stress, comportement, douleur).
            - Invasivité : `"Non invasif"` / `"Invasif"` / `"Inconnu"` / `"présumé non invasif"`.
            - Numéros des péchés concernés (si applicables).
            - Nouveaux péchés : `"Oui"` ou `"Non"` + justification.
        - **Obs** : Cite TOUS les extraits pertinents utilisés pour ton raisonnement (même s'il n'y en a qu'un seul, liste obligatoire).

        4. Ne traite qu'un **seul protocole par article**, en suivant ces priorités :
        - Si un protocole est clairement plus invasif que les autres, choisis celui-ci.
        - Sinon, si plusieurs protocoles ont un niveau d'invasivité similaire, décris-les **TOUS** dans l'analyse du protocole unique.
        - Si tous les protocoles sont non-invasifs ou présumés non-invasifs, choisis celui qui est décrit avec le plus de détails.
        5. Ignore toute méthode ne concernant **pas** l'échantillonnage d'ADN (ex : PCR, séquençage...).

        ##################################################
        # FORMAT DE RÉPONSE
        ##################################################

        Pour chaque protocole identifié, présente ton analyse en suivant strictement les étapes REACT (Thought, Act, Obs), puis fournis un JSON final selon ce format:

        ```json
        {{
        "protocole": "Nom concis du protocole",
        "extrait_pertinent": ["Texte exact de l'extrait 1", "Texte exact de l'extrait 2"],
        "echantillon": "Type d'échantillon (poils/sang/fèces/urine/salive/mixte)",
        "impacts_potentiels": ["impact1", "impact2"],
        "evaluation_invasivite": "Non invasif / Invasif / Inconnu / présumé non invasif",
        "peches_identifies": ["1", "2", "5"],
        "nouveaux_peches": "Oui / Non"
        }}
        ##################################################
        VÉRIFICATION DU JSON
        ##################################################
        Avant de soumettre ton JSON, vérifie STRICTEMENT:

        Qu'il n'y a pas de clés dupliquées.
        Que toutes les accolades et crochets sont bien fermés.
        Que toutes les virgules sont correctement placées.
        Que le JSON est parfaitement valide et pourrait être parsé sans erreur.
        Que SI tu souhaites faire une liste, elle commence par [ et se termine par ].

        RÈGLES CRUCIALES POUR LE FORMAT JSON :

        Chaque clé doit apparaître UNE SEULE FOIS dans l'objet JSON.
        Les noms de clés doivent utiliser des underscores simples (_) et NE PAS contenir d'accents.
        Toute liste doit commencer par [ et se terminer par ].
        Chaque élément dans un objet ou une liste doit être séparé par une virgule.
        Assure-toi que chaque valeur est du bon type : chaînes entre guillemets, listes entre crochets.
        Assure-toi que le champ extrait_pertinent contient toujours une liste de chaînes, même s'il n'y a qu'un seul élément.

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
    "protocole", "extrait_pertinent", "echantillon",
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
            "echantillon": item["echantillon"],
            "Extrait pertinent": item["extrait_pertinent"],
            "Évaluation d'invasivité": item["evaluation_invasivite"],
            "Péchés identifiés": item["peches_identifies"],
            "Nouveaux péchés": item["nouveaux_peches"]
        }
        for item in parsed_blocks
    ])

    return df

    
