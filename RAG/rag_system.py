"""
Système RAG principal
"""

import os
import json
from typing import List, Dict, Any, Optional

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
    
    def build_context(self, documents: List[Document],max_length: int = 6000) -> str:
        """
        Construit le contexte à partir des documents récupérés
        
        Args:
            documents: Liste des documents
            
        Returns:
            Contexte concaténé
        """
        context_parts = []
        total_length = 0

        for i, doc in enumerate(documents):
            content = doc.content.strip()
            excerpt = f"EXTRAIT {i+1}:\n{content}"
            if total_length + len(excerpt) > max_length:
                break
            context_parts.append(excerpt)
            total_length += len(excerpt)
        print ("CONTEXTE ----------")
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

        return f"""Tu es un assistant scientifique expert dans l'analyse des méthodes d'échantillonnage d'ADN. Ta tâche est d'examiner attentivement les protocoles décrits dans les extraits ci-dessous et de déterminer s'ils contreviennent aux "Sept Péchés de l'Échantillonnage d'ADN Non-Invasif".

        ##################################################
        # DÉFINITIONS ET CONTEXTE
        ##################################################

        Définitions des Sept Péchés :
        {seven_sins_definitions}

        Définition de l'échantillonnage non-invasif :
        {definition}

        ##################################################
        # CONTEXTE À ANALYSER
        ##################################################

        Chaque extrait est précédé de 'EXTRAIT n:'. Analyse-les un par un.

        {context}

        ##################################################
        # QUESTION
        ##################################################

        {question}

        ##################################################
        # INSTRUCTIONS D'ANALYSE PRÉCISES ET STRICTES
        ##################################################

        Si aucun protocole clair n'est identifiable dans le contexte (pas de partie du corps utilisée, pas de méthode de prélèvement), NE FOURNIS AUCUNE ANALYSE. Indique simplement : "Aucun protocole identifiable".

        Analyse chaque protocole d'échantillonnage d'ADN identifié dans le contexte. Ta réponse FINALE doit ABSOLUMENT et EXCLUSIVEMENT respecter le format indiqué ci-dessous. AUCUNE information supplémentaire, répétition ou reformulation ne sera tolérée. Sois EXTRÊMEMENT concis et précis dans ton analyse.

        MÉTHODE D'ANALYSE (pour ton raisonnement interne) :
        1. Identifie UNIQUEMENT les protocoles d'échantillonnage d'ADN DISTINCTS mentionnés dans le texte. Ne répète pas l'analyse pour des mentions similaires.
        2. Pour CHAQUE protocole DISTINCT, analyse de manière ULTRA-CONCISE :
        - La méthode d'obtention de l'ADN et la partie de l'animal concernée (en quelques mots).
        - La présence ou l'absence de manipulation, capture ou perturbation DIRECTE de l'animal (Oui/Non et brève description).
        - Les impacts potentiels sur le comportement ou le bien-être animal (très brièvement).
        - Le niveau d'invasivité selon la définition fournie (Non invasif/Minimalement invasif/Invasif).
        - Les numéros des péchés transgressés (uniquement les numéros).
        - La justification DIRECTE et ULTRA-CONCISE de chaque péché transgressé, en lien STRICT avec sa définition (une phrase courte par péché).
        - La proposition d'un NOUVEAU péché UNIQUEMENT si le protocole est clairement invasif et ne correspond à AUCUN des cinq péchés existants (formulation très brève du nouveau péché).
        3. **Assure-toi que l' "EXTRAIT PERTINENT" que tu sélectionnes pour chaque protocole est le même dans les deux sections de ta réponse (Analyse Détaillée et Tableau Récapitulatif) et qu'il conserve la langue d'origine.**

        ##################################################
        # FORMAT DE RÉPONSE OBLIGATOIRE ET UNIQUE
        ##################################################

        Ta réponse doit contenir STRICTEMENT les DEUX sections suivantes, dans cet ordre PRÉCIS :

        1. ANALYSE DÉTAILLÉE PAR PROTOCOLE

        Pour chaque protocole DISTINCT identifié, présente l'analyse EXACTEMENT dans ce format :

        PROTOCOLE: [Nom concis du protocole]
        EXTRAIT PERTINENT: "[Citation ASSEZ courte du texte clé EN LANGUE D'ORIGINE]"
        DESCRIPTION: [Description ULTRA-CONCISE de la méthode]
        MANIPULATION: [Oui/Non] - [Brève indication de la manipulation]
        IMPACTS POTENTIELS: [Très brève indication des impacts]
        ÉVALUATION D'INVASIVITÉ: [Non invasif/Minimalement invasif/Invasif]
        JUSTIFICATION: [Justification ULTRA-CONCISE basée sur la définition]
        PÉCHÉS IDENTIFIÉS: [Liste des numéros des péchés]
        JUSTIFICATION DES PÉCHÉS: [Justification ULTRA-CONCISE pour chaque péché]
        NOUVEAUX PÉCHÉS?: [Formulation ULTRA-CONCISE du nouveau péché si applicable]

        2. SYNTHÈSE ET TABLEAU RÉCAPITULATIF

        | Protocole                 | Extrait pertinent | Statut              | Péchés (N°) | Nouveaux péchés (si applicable) |
        |---------------------------|---------------------------------|---------------------|-------------|---------------------------------|
        | [Nom concis Protocole 1]  | "[Citation du texte EN LANGUE D'ORIGINE]"          | [Statut]            | [Numéros]   | [Nouveau péché concis]          |
        | [Nom concis Protocole 2]  | "[Citation du texte EN LANGUE D'ORIGINE]"          | [Statut]            | [Numéros]   | [Nouveau péché concis]          |
        | ...                       | ...                             | ...                 | ...         | ...                             |

        CONTRAINTES ABSOLUES :
        - RÉPONDS UNIQUEMENT dans ce format EXACT. AUCUN texte introductif, conclusif, explication supplémentaire, note personnelle ou reformulation ne sera toléré.
        - LIMITE-TOI STRICTEMENT à ces DEUX sections dans l'ordre indiqué.
        - NE PRÉSENTE AUCUNE répétition d'analyse pour des protocoles similaires. Analyse chaque protocole DISTINCT une seule fois.
        - SOIS EXTRÊMEMENT CONCIS dans TOUTES les parties de ta réponse.
        - RESPECTE À LA LETTRE les titres de champs fournis.
        - Les extraits pertinents dans les DEUX sections doivent conserver la langue d'origine.
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


    def parse_response(self, response: str):
        import re

        protocols = []
        tableau = []

        print("Raw response to parse:", response)

        # Parser les protocoles
        protocol_pattern = r'PROTOCOLE:\s*(.*?)\s*EXTRAIT PERTINENT:\s*["\']?(.*?)["\']?[\s\S]*?ÉVALUATION D\'INVASIVITÉ:\s*(.*?)\s*JUSTIFICATION:[\s\S]*?PÉCHÉS IDENTIFIÉS:\s*(.*?)\s*JUSTIFICATION DES PÉCHÉS:[\s\S]*?NOUVEAUX PÉCHÉS\?:\s*(.*?)(?=\nPROTOCOLE:|### SYNTHÈSE ET TABLEAU RÉCAPITULATIF|$)'
        protocol_blocks = re.findall(protocol_pattern, response, re.DOTALL)

        print("Protocol blocks found:", protocol_blocks)

        for block in protocol_blocks:
            nom = block[0].strip()
            extrait = block[1].strip()
            invasivite = block[2].strip()
            peches_str = block[3].strip()
            nouveaux_peches = block[4].strip()

            print(f"Parsing protocol block: {nom}")
            print(f"Extrait: {extrait}")
            print(f"Invasivité: {invasivite}")
            print(f"Péchés: {peches_str}")
            print(f"Nouveaux péchés: {nouveaux_peches}")

            peches = [p.strip() for p in peches_str.split(',') if p.strip() and p.lower() != "aucun"]
            nouveaux_peches = "" if nouveaux_peches.lower() in ["non", "aucun", "n/a"] else nouveaux_peches.strip()

            if nom and extrait:
                protocols.append({
                    'nom': nom,
                    'extrait': extrait,
                    'invasivite': invasivite,
                    'peches': peches,
                    'nouveaux_peches': nouveaux_peches
                })

        print("Parsed protocols:", protocols)

        # Parser le tableau récapitulatif en ignorant l'entête et le séparateur)
        tableau_pattern = r'\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|'
        tableau_blocks = re.findall(tableau_pattern, response, re.DOTALL)

        print("Tableau blocks found:", tableau_blocks)

        header_skipped = False
        separator_skipped = False

        for block in tableau_blocks:
            protocole = block[0].strip()
            extrait = block[1].strip()
            statut = block[2].strip()
            peches_str = block[3].strip()
            nouveaux_peches = block[4].strip()

            if not header_skipped and protocole.lower() == "protocole":
                header_skipped = True
                continue
            elif not separator_skipped and protocole.startswith("---"):
                separator_skipped = True
                continue
            elif header_skipped and separator_skipped and protocole:
                peches = [p.strip() for p in peches_str.split(',') if p.strip() and p.lower() != "aucun"]
                tableau.append({
                    'protocole': protocole,
                    'extrait': extrait,
                    'statut': statut,
                    'peches': peches,
                    'nouveaux_peches': nouveaux_peches.strip()
                })

        return {'protocols': protocols, 'tableau': tableau}


    
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
