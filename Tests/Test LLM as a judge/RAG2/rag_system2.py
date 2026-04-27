"""
Système RAG principal
"""

import os
import json
from typing import List, Dict, Any, Optional
import re

from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.components.preprocessors import DocumentSplitter

import sys

# Ajoute le dossier parent de RAG2/ (c’est-à-dire RAG/) au sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
        return f"""
    Tu es un assistant scientifique. Ton objectif est d'extraire **précisément** les informations suivantes à partir des extraits donnés :

    - animal : nom de l'animal sur lequel l'ADN a été prélevé
    - famille : famille biologique de cet animal
    - Invert/Vert : en fonction de l'animal déduit si il est vertébrés ou invertébrés
    - pays : où l'échantillon a été prélevé
    - source : extrait du texte qui mentionne le pays ou l'animal (en quelques phrases) NE PAS TRADUIRE 

    ⚠️ Ne devine rien. Si une info est absente, mets "inconnu". Donne UNE SEULE RÉPONSE par document. Je veux que tu ne me donnes ça et uniquement ça :
    Format UNIQUE de réponse souhaité :
    animal:<nom>
    famille:<famille>
    Invert/Vert:<Invert/Vert>
    pays:<pays>
    source:<extrait court>

    Extraits :
    {context}

    Question : {question}

    Réponse :
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


    import re

    def parse_animal_metadata_response(self, response: str) -> list[dict]:
        """
        Parse les informations à partir d'une réponse structurée en blocs "**EXTRAIT X :**".
        Gère les indentations et les champs avec ou sans guillemets.
        """

        def clean_value(val: str) -> str:
            return val.strip().strip('"') if val and val.lower() != "inconnu" else ""

        def unique_join(values: list[str], sep: str = ", ") -> str:
            return sep.join(dict.fromkeys(
                v for v in values if v and v.lower() != "inconnu"
            ))

        # Découper les blocs par EXTRAIT (ex : **EXTRAIT 1 :**)
        blocks = re.split(r"\*\*EXTRAIT\s*\d+\s*:\*\*", response, flags=re.IGNORECASE)

        entries = []
        for block in blocks:
            if not block.strip():
                continue

            animal_match = re.search(r"^\s*animal\s*:\s*(.+)", block, flags=re.IGNORECASE | re.MULTILINE)
            famille_match = re.search(r"^\s*famille\s*:\s*(.+)", block, flags=re.IGNORECASE | re.MULTILINE)
            vert_match =  re.search(r"^\s*Invert/Vert\s*:\s*(.+)", block, flags=re.IGNORECASE | re.MULTILINE)
            pays_match = re.search(r"^\s*pays\s*:\s*(.+)", block, flags=re.IGNORECASE | re.MULTILINE)
            source_match = re.search(r"^\s*source\s*:\s*(.+)", block, flags=re.IGNORECASE | re.MULTILINE)

            animal = clean_value(animal_match.group(1)) if animal_match else ""
            famille = clean_value(famille_match.group(1)) if famille_match else ""
            vert = clean_value(vert_match.group(1)) if vert_match else ""
            pays = clean_value(pays_match.group(1)) if pays_match else ""
            source = clean_value(source_match.group(1)) if source_match else ""

            if animal or famille or pays or source:
                entries.append({
                    "animal": animal,
                    "famille": famille,
                    "Invert/Vert" : vert,
                    "pays": pays,
                    "source": source
                })

        if not entries:
            return []

        combined_entry = {
            "animal": unique_join([e["animal"] for e in entries]),
            "famille": unique_join([e["famille"] for e in entries]),
            "Invert/Vert" : unique_join([e["Invert/Vert"]for e in entries]),
            "pays": unique_join([e["pays"] for e in entries]),
            "source": " ".join(dict.fromkeys(
                e["source"] for e in entries if e["source"]
            ))
        }

        return [combined_entry] if any(combined_entry.values()) else []











            
