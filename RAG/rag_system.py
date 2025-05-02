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
        
        # Initialisation des pipelines
        self.hyde_pipeline = create_hyde_pipeline(self.api_key, self.api_url, self.embedder_model)
        self.rag_pipeline = create_rag_pipeline(self.text_embedder, self.retriever, self.reranker)
        
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
    
    def build_context(self, documents: List[Document]) -> str:
        """
        Construit le contexte à partir des documents récupérés
        
        Args:
            documents: Liste des documents
            
        Returns:
            Contexte concaténé
        """
        if isinstance(documents, list):
            if all(hasattr(doc, "content") for doc in documents):
                return "\n\n".join([doc.content for doc in documents])
            else:
                return "\n\n".join([str(doc) for doc in documents])
        else:
            return str(documents)
    
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

        return f"""Tu es un assistant scientifique spécialisé dans l'analyse des méthodes d'échantillonnage d'ADN. Ta mission est d'examiner les protocoles décrits dans les extraits ci-dessous et de juger s'ils contreviennent aux "Sept Péchés de l'Échantillonnage d'ADN Non-Invasif".

    Définitions des Sept Péchés :
    {seven_sins_definitions}

    Définition de l'échantillonnage non-invasif :
    {definition}

    Contexte :
    {context}

    Question :
    {question}

    Méthode à suivre :

    1. **Identifier les protocoles** d'échantillonnage d'ADN mentionnés.

    2. **Décrire brièvement** chaque protocole :
    - Quelle méthode ?
    - Comment et sur quelle partie de l'animal est prélevé l'ADN ?
    - Y a-t-il manipulation, capture ou perturbation ?
    - Quels effets possibles sur le comportement ou le bien-être ?

    3. **Évaluer l'invasivité** de chaque protocole :
    - Est-il non invasif, invasif selon la définition donnée, si le protocole est ne serait-ce qu'un petit peu invasif, alors il est invasif ?
    - Justifie clairement ton jugement.

    4. **Comparer aux Sept Péchés** :
    - Le protocole transgresse-t-il un ou plusieurs péchés ? Si oui, lesquels et pourquoi précise le numéro du péché ?
    - Si le protocole tu estime que le protocole est invasif mais qu'il ne rentre dans aucun péché, relève le et donne un nom à ce nouveau péché !

    5. **Synthèse finale** :
    - Résume clairement tes conclusions pour chaque protocole.
    - Présente un **tableau récapitulatif** :  
        | Protocole | Statut (Non/Minim./Invasif) | Péchés identifiés et avec le numéro |

    IMPORTANT : Ne te limite pas aux affirmations de l'article. Utilise la **définition** et les **péchés** comme cadre critique pour évaluer les pratiques décrites, même si elles sont qualifiées de "non invasives" dans le texte.
    Base-toi uniquement sur les informations contenues dans le contexte fourni.
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

    def answer_question(self, question: str, definition: str = "", use_hyde: bool = True, top_k: int = 4) -> str:
        """
        Processus complet pour répondre à une question
        
        Args:
            question: La question de l'utilisateur
            definition: Définition à inclure dans le prompt (optionnel)
            use_hyde: Utiliser HyDE ou la méthode standard
            top_k: Nombre de documents à récupérer
            
        Returns:
            La réponse finale
        """
        # Récupération des documents pertinents
        if use_hyde:
            retrieved_docs = self.retrieve_with_hyde(question, top_k)
            print(f"Documents récupérés avec HyDE: {len(retrieved_docs)}")
        else:
            retrieved_docs = self.retrieve_standard(question, top_k)
            print(f"Documents récupérés avec méthode standard: {len(retrieved_docs)}")

        # Affichage des documents récupérés pour le debugging
        print("\n--- DOCUMENTS RÉCUPÉRÉS ---")
        for i, doc in enumerate(retrieved_docs):
            print(f"\nDocument {i+1}:")
            print(f"Score: {doc.score if hasattr(doc, 'score') else 'N/A'}")
            print(f"Contenu: {doc.content}")
            print("-" * 80)

        # Construction du contexte
        context = self.build_context(retrieved_docs)
        
        # Génération de la réponse
        answer = self.generate_answer(question, context, definition)
        
        return answer
