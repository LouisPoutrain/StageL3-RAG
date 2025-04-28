import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from numpy import array, mean

from haystack import Pipeline, component, Document
from haystack.dataclasses import Document
from haystack.components.preprocessors import DocumentSplitter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.components.builders import PromptBuilder
from haystack.components.converters import OutputAdapter

from llama_cpp import Llama
from FewShotPrompting import FewShotPrompting
from TEIXMLToDocument import TEIXMLToDocument

#  embedding de documents hypothétiques
@component
class HypotheticalDocumentEmbedder:
    @component.output_types(hypothetical_embedding=List[float])
    def run(self, documents: List[Document]):
        # Collecte des embeddings des documents hypothétiques
        stacked_embeddings = array([doc.embedding for doc in documents])
        # Calcul de l'embedding moyen
        avg_embeddings = mean(stacked_embeddings, axis=0)
        # Mise en forme du vecteur HyDE
        hyde_vector = avg_embeddings.reshape((1, len(avg_embeddings)))
        return {"hypothetical_embedding": hyde_vector[0].tolist()}

# adapter la sortie du LLM local aux documents
@component
class LlamaOutputAdapter:
    def __init__(self, model_path, max_tokens=1024, temperature=0.1):
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    @component.output_types(documents=List[Document])
    def run(self, prompt: str, n_generations: int = 3):
        """
        Génère n_generations réponses avec Llama et les convertit en documents
        """
        try:
            llm = Llama(model_path=self.model_path, n_ctx=4096)
            documents = []
            
            for _ in range(n_generations):
                response = llm(
                    prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    repeat_penalty=1.2,
                    top_k=40,
                    top_p=0.95
                )
                text = response["choices"][0]["text"]
                documents.append(Document(content=text))
            
            return {"documents": documents}
        except Exception as e:
            return {"documents": [Document(content=f"Erreur: {str(e)}")]}
        finally:
            if 'llm' in locals():
                llm.close()

class RAGSystem:
    def __init__(self, model_path="./model/mistral-7b-instruct-v0.1.Q4_K_M.gguf"): # modifier le chemin 
        """
        Initialise le système RAG avec tous ses composants
        """
        self.model_path = model_path
        self.document_store = InMemoryDocumentStore()
        
        # Initialiser les embedders 
        self.embedder_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.text_embedder = SentenceTransformersTextEmbedder(model=self.embedder_model)
        self.text_embedder.warm_up()
        
        # retriever et writer
        self.retriever = InMemoryEmbeddingRetriever(document_store=self.document_store)
        self.writer = DocumentWriter(document_store=self.document_store)
        
        # Reranker
        self.reranker = TransformersSimilarityRanker(model="cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.reranker.warm_up()
        
        # initialiser les pipelines
        self._setup_indexing_pipeline()
        self._setup_hyde_pipeline()
        self._setup_rag_pipeline()

    def _setup_indexing_pipeline(self):
        """Configure la pipeline d'indexation des documents TEI"""
        self.indexing_pipeline = Pipeline()
        self.indexing_pipeline.add_component("tei_converter", TEIXMLToDocument())
        self.indexing_pipeline.add_component("splitter", DocumentSplitter(
            split_length=300, 
            split_overlap=20, 
            split_by="word"
        ))
        self.indexing_pipeline.add_component("doc_embedder", 
                                            SentenceTransformersDocumentEmbedder(model=self.embedder_model))
        self.indexing_pipeline.add_component("writer", self.writer)

        # Connexions
        self.indexing_pipeline.connect("tei_converter.documents", "splitter.documents")
        self.indexing_pipeline.connect("splitter.documents", "doc_embedder.documents")
        self.indexing_pipeline.connect("doc_embedder.documents", "writer.documents")

    def _setup_hyde_pipeline(self):
        """
        Configure la pipeline HyDE selon la documentation de Haystack
        """
        self.hyde_pipeline = Pipeline()
        
        # pour la génération de documents hypothétiques
        self.hyde_pipeline.add_component(
            "prompt_builder", 
            PromptBuilder(
                template="""Étant donné une question, génère un paragraphe de texte qui répond à la question.
                Question: {{question}}
                Paragraphe:""",
                required_variables=["question"]  
            )
        )
        

        self.hyde_pipeline.add_component(
            "generator",
            LlamaOutputAdapter(
                model_path=self.model_path,
                max_tokens=400,
                temperature=0.75
            )
        )
        
        self.hyde_pipeline.add_component("embedder", 
                                    SentenceTransformersDocumentEmbedder(model=self.embedder_model))
        
        # calculer l'embedding moyen
        self.hyde_pipeline.add_component("hyde", HypotheticalDocumentEmbedder())
        
        # Connexions
        self.hyde_pipeline.connect("prompt_builder.prompt", "generator.prompt")
        self.hyde_pipeline.connect("generator.documents", "embedder.documents")
        self.hyde_pipeline.connect("embedder.documents", "hyde.documents")
    
    def _setup_rag_pipeline(self):
        """
        Configure la pipeline RAG standard
        """
        self.rag_pipeline = Pipeline()
        self.rag_pipeline.add_component("text_embedder", self.text_embedder)
        self.rag_pipeline.add_component("retriever", self.retriever)
        self.rag_pipeline.add_component("reranker", self.reranker)
    
    def index_documents(self, tei_folder: str = "tei") -> int:
        """
        Indexe tous les documents TEI d'un dossier
        
        Args:
            tei_folder: Chemin vers le dossier contenant les fichiers TEI.XML
            
        Returns:
            Le nombre de documents indexés
        """
        tei_path = Path(tei_folder)
        tei_files = [str(tei_path / f) for f in os.listdir(tei_path) if f.endswith(".xml")]
        
        if not tei_files:
            print("Aucun fichier TEI trouvé dans le dossier spécifié.")
            return 0
            
        self.indexing_pipeline.run({
            "tei_converter": {"sources": tei_files}
        })
        
        indexed_docs = self.document_store.filter_documents()
        num_docs = len(indexed_docs)
        
        print(f"Nombre de documents indexés : {num_docs}")
        if num_docs > 0:
            print(f"Premier document : {indexed_docs[0].content[:100]}...")
            print(f"Embedding présent : {indexed_docs[0].embedding is not None}")
            
        return num_docs
    
    def retrieve_with_hyde(self, question: str, top_k: int = 4) -> List[Document]:
        """
        Récupère les documents pertinents en utilisant HyDE
        
        Args:
            question: La question de l'utilisateur
            top_k: Nombre de documents à récupérer
            
        Returns:
            Liste des documents récupérés
        """
        # Générer et encoder le document hypothétique avec la pipeline HyDE
        hyde_output = self.hyde_pipeline.run({
            "prompt_builder": {"question": question},
            "generator": {"n_generations": 3}  # Générer 3 documents hypothétiques
        })
        
        # Vérifier la structure des données retournées
        print("Clés disponibles dans hyde_output:", list(hyde_output.keys()))
        # Afficher la structure pour chaque composant
        for key in hyde_output:
            print(f"Structure de {key}:", list(hyde_output[key].keys()) if isinstance(hyde_output[key], dict) else "Non dictionnaire")
        
        # Récupérer l'embedding hypothétique moyen
        hyp_embedding = hyde_output["hyde"]["hypothetical_embedding"]
        
        # Utiliser cet embedding pour rechercher des documents similaires
        retrieved_docs = self.retriever.run(
            query_embedding=hyp_embedding, 
            top_k=top_k
        )["documents"]
        
        # Afficher les documents hypothétiques générés pour inspection
        print("\nDocuments hypothétiques générés:")
        
        # Accéder correctement aux documents générés
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
    
    def rerank_documents(self, question: str, documents: List[Document]) -> List[Document]:
        """
        Réordonne les documents selon leur pertinence
        
        Args:
            question: La question de l'utilisateur
            documents: Liste des documents à réordonner
            
        Returns:
            Liste des documents réordonnés
        """
        reranker_output = self.reranker.run(query=question, documents=documents)
        
        if isinstance(reranker_output, dict) and "documents" in reranker_output:
            return reranker_output["documents"]
        else:

            print("Avertissement: Le reranker n'a pas renvoyé un format attendu.")
            return documents
    
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
    
    def generate_answer(self, question: str, context: str, definition: str = "") -> str:
        """
        Génère une réponse à partir du modèle local
        
        Args:
            question: La question de l'utilisateur
            context: Le contexte extrait des documents
            definition: Éventuellement une définition à inclure dans le prompt
            
        Returns:
            La réponse générée
        """
        prompt = self._build_prompt(question, context, definition)
        
        try:
            llm = Llama(model_path=self.model_path, n_ctx=4096)
            response = llm(
                prompt,
                max_tokens=1024,
                temperature=0.2,
                repeat_penalty=1.2,
                top_k=40,
                top_p=0.95,
                stop=["</answer>", "\n\n\n"]
            )
            answer = response["choices"][0]["text"]
            return answer
        except ValueError as e:
            return f"Erreur lors de la génération de la réponse: {e}"
        finally:
            if 'llm' in locals():
                llm.close()
    
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
        definition_part = f"\nVoici la définition de l'échantillonnage non-invasif :\n{definition}\n" if definition else ""
        
        return f"""Tu es un assistant scientifique rigoureux, spécialisé dans l'analyse des méthodes d'échantillonnage d'ADN.

        Ta mission est de répondre à la question suivante en t'appuyant **strictement et uniquement** sur les extraits de texte ci-dessous.
            Ta réponse doit :
            - Être précise, complète et factuelle.
            - Évaluer l'ensemble du protocole expérimental (capture, maintien, exposition, sacrifice) par rapport à la définition de l'échantillonnage non-invasif.
            - Identifier si certaines parties du protocole peuvent être considérées non-invasives ou non, et conclure globalement.
            - Justifier ton raisonnement avec des extraits ou éléments du contexte (citer clairement).

        {definition_part}
        Contexte :
        {context}

        Question :
        {question}

        Réponds en suivant ce format :
        - Réponse courte (Oui / Non / Impossible de conclure).
        - Justification détaillée : Analyse du protocole complet et argumentation basée sur le contexte extrait (citer les éléments précis qui justifient ta réponse).
        Réponse :
        """

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
        # Récupérer les documents
        if use_hyde:
            retrieved_docs = self.retrieve_with_hyde(question, top_k)
            print(f"Documents récupérés avec HyDE: {len(retrieved_docs)}")
        else:
            retrieved_docs = self.retrieve_standard(question, top_k)
            print(f"Documents récupérés avec méthode standard: {len(retrieved_docs)}")
        
        # reranker les documents
        reranked_docs = self.rerank_documents(question, retrieved_docs)
        
        # Construire le contexte
        context = self.build_context(reranked_docs)
        
        # Générer la réponse
        answer = self.generate_answer(question, context, definition)
        
        return answer


# Exemple d'utilisation
if __name__ == "__main__":
    rag_system = RAGSystem()
    
    # Indexer les documents
    num_docs = rag_system.index_documents("tei")
    
    if num_docs > 0:
        # Question et définition
        question = "La méthode proposée pour estimer le ratio de sexe des populations de tétras des armoises utilise-t-elle réellement des échantillons génétiques non-invasifs au sens strict du terme scientifique définit par Taberlet et Al. ?"
        definition = "Selon Taberlet et al. (1999), l’échantillonnage d’ADN non invasif désigne toute méthode permettant d'obtenir du matériel génétique sans avoir à capturer, blesser, ni perturber significativement l'animal. Cela inclut, par exemple, l’analyse d’échantillons laissés dans l’environnement comme les poils, les plumes, les fèces, l’urine, ou encore la salive."
        
        print("\n--- Méthode avec HyDE ---")
        hyde_answer = rag_system.answer_question(question, definition, use_hyde=True)
        print("\nRéponse avec HyDE:")
        print(hyde_answer)
        
        print("\n--- Méthode standard ---")
        standard_answer = rag_system.answer_question(question, definition, use_hyde=False)
        print("\nRéponse standard:")
        print(standard_answer)