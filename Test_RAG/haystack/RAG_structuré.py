import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from numpy import array, mean
import requests

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
from UniversityLLMAdapter import UniversityLLMAdapter

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
    def __init__(self, api_key=None, api_url=None):
        """
        Initialise le système RAG avec tous ses composants
        """
        self.api_key = api_key or os.environ.get("UNIVERSITY_LLM_API_KEY")
        self.api_url = api_url or os.environ.get("UNIVERSITY_LLM_API_URL")
        
        if not self.api_key or not self.api_url:
            raise ValueError("API key and URL must be provided or set as environment variables")
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
                template="""Imagine que tu es un article scientifique qui contient l'information permettant de répondre à cette question. 
                Génère un court extrait de ce document qui contiendrait l'information pertinente, sans directement répondre à la question.
                
                Question: {{question}}
                
                Extrait de document scientifique:""",
                required_variables=["question"]  
            )
        )
        

        self.hyde_pipeline.add_component(
            "generator",
            UniversityLLMAdapter(
                api_key=self.api_key,
                api_url=self.api_url,
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
       # tei_files = [str(tei_path / f) for f in os.listdir(tei_path) if f.endswith(".xml")]
       # Solution temporaire
        tei_files = Path("tei/012017-jfwm-007.grobid.tei.xml")
        self.indexing_pipeline.run({
            "tei_converter": {"sources": [tei_files]}
        })

        
        if not tei_files:
            print("Aucun fichier TEI trouvé dans le dossier spécifié.")
            return 0
            
        '''self.indexing_pipeline.run({
            "tei_converter": {"sources": tei_files}
        })'''
        
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
            top_k=top_k,
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
        Génère une réponse à partir d'une API compatible OpenAI
        
        Args:
            question: La question de l'utilisateur
            context: Le contexte extrait des documents
            definition: Éventuellement une définition à inclure dans le prompt
            
        Returns:
            La réponse générée
        """
        prompt = self._build_prompt(question, context, definition)
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "phi3:14b",
                "messages": [
                    {"role": "system", "content": "Tu es un assistant scientifique rigoureux."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1024,
                "temperature": 0.2
            }
        
            
            response = requests.post(self.api_url, headers=headers, json=payload)
            
            content_type = response.headers.get('Content-Type', '')
            
            try:
                result = response.json()
            except Exception as e:
                return f"Erreur de parsing JSON: {str(e)}. Réponse: {response.text[:300]}"
            # Extraction des réponses selon différents formats possibles
            if "choices" in result and len(result["choices"]) > 0:
                if "message" in result["choices"][0]:
                    return result["choices"][0]["message"].get("content", "")
                elif "text" in result["choices"][0]:
                    return result["choices"][0]["text"]
            elif "output" in result:
                return result["output"]
            elif "generated_text" in result:
                return result["generated_text"]
            elif "completion" in result:
                return result["completion"]
            else:
                import json
                return f"Structure de réponse non reconnue: {json.dumps(result, indent=2)}"
                
        except Exception as e:
            error_msg = f"Erreur lors de la génération de la réponse: {str(e)}"
            if 'response' in locals():
                if hasattr(response, 'status_code'):
                    error_msg += f"\nStatus: {response.status_code}"
                if hasattr(response, 'text'):
                    error_msg += f"\nRéponse: {response.text[:300]}"
            return error_msg
        

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

            return f"""Tu es un assistant scientifique rigoureux, spécialisé dans l'analyse des méthodes d'échantillonnage d'ADN.
                Ta mission est d'analyser les protocoles d'échantillonnage d'ADN décrits dans les extraits de texte ci-dessous pour déterminer s'ils contreviennent aux "Sept Péchés de l'Échantillonnage d'ADN Non-Invasif".
                
                Voici les définitions des Sept Péchés :
                {seven_sins_definitions}
                
                De plus, voici la définition de l'échantillonnage non-invasif :
                {definition}
                
                Contexte :
                {context}
                
                Question :
                {question}
                
                INSTRUCTIONS: Pour répondre à cette question, tu dois procéder méthodiquement étape par étape comme suit :
                
                Étape 1: Identifie tous les protocoles d'échantillonnage d'ADN mentionnés dans le contexte fourni. Liste-les clairement.
                
                Étape 2: Pour chaque protocole identifié, réponds aux questions suivantes :
                a) En quoi consiste exactement ce protocole d'échantillonnage ? Décris les méthodes utilisées en détail.
                b) Comment l'échantillon d'ADN est-il prélevé ? Quelles parties de l'animal sont concernées ?
                c) Y a-t-il manipulation, capture ou perturbation de l'animal lors de l'échantillonnage ?
                d) Quels impacts potentiels cette méthode pourrait-elle avoir sur le comportement ou le bien-être de l'animal ?
                
                Étape 3: Détermine si chaque protocole est véritablement non invasif selon la définition fournie :
                a) Compare les caractéristiques du protocole à la définition de l'échantillonnage non-invasif.
                b) Identifie les points de convergence et de divergence.
                c) Formule un jugement justifié : le protocole est-il non invasif, minimalement invasif, ou invasif ?
                
                Étape 4: Évalue si le protocole contrevient à l'un des Sept Péchés :
                a) Examine chaque péché un par un et vérifie si le protocole y correspond.
                b) Si un péché est identifié, explique précisément pourquoi et comment le protocole y contrevient.
                c) Cite des éléments spécifiques du protocole qui illustrent la transgression du péché.
                
                Étape 5: Synthèse finale :
                a) Résume tes conclusions pour chaque protocole identifié.
                b) Présente un tableau récapitulatif des protocoles, leur statut (non invasif, minimalement invasif, invasif) et les péchés identifiés.
                
                IMPORTANT: Base-toi uniquement sur les informations présentes dans les documents. Ne mentionne pas de méthodes ou de conclusions qui ne sont pas explicitement décrites dans le contexte.
    
                Lorsque tu répondras à la question, tu dois suivre cette méthodologie étape par étape. 
                 N'essaie pas de répondre maintenant. Attends la question complète de l'utilisateur et applique ensuite cette méthode.
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

         # Afficher le contenu des documents récupérés
        print("\n--- DOCUMENTS RÉCUPÉRÉS ---")
        for i, doc in enumerate(retrieved_docs):
            print(f"\nDocument {i+1}:")
            print(f"Score: {doc.score if hasattr(doc, 'score') else 'N/A'}")
            print(f"Contenu: {doc.content}")
            print("-" * 80)

            # RERANKER À RÉACTIVER QUAND ON AURA UN MEILLEUR RETRIEVER
        '''  
        # reranker les documents
        reranked_docs = self.rerank_documents(question, retrieved_docs)

        # Afficher les documents après reranking
        print("\n--- DOCUMENTS APRÈS RERANKING ---")
        for i, doc in enumerate(reranked_docs):
            print(f"\nDocument {i+1}:")
            print(f"Score: {doc.score if hasattr(doc, 'score') else 'N/A'}")
            print(f"Contenu: {doc.content}")
            print("-" * 80)
        
        # Construire le contexte
        context = self.build_context(reranked_docs)'''
        context = self.build_context(retrieved_docs)
        
        # Générer la réponse
        answer = self.generate_answer(question, context, definition)
        
        return answer


# Exemple d'utilisation
if __name__ == "__main__":
    rag_system = RAGSystem(
         api_key="YOUR-API-KEY",
         api_url="http://gpu1.pedagogie.sandbox.univ-tours.fr:32800/api/chat/completions"
    )
    
    # Indexer les documents
    num_docs = rag_system.index_documents("tei")
    
    if num_docs > 0:
        # Question et définition
        question = "Dans l'article spécifique qui traite des chauves-souris, peux-tu me donner tous les protocoles d'échantillonnage d'ADN et si cet article les décrit comme invasif ou non-invasif ?"
        definition = "Selon Taberlet et al. (1999), l’échantillonnage d’ADN non invasif désigne toute méthode permettant d'obtenir du matériel génétique sans avoir à capturer, blesser, ni perturber significativement l'animal. Cela inclut, par exemple, l’analyse d’échantillons laissés dans l’environnement comme les poils, les plumes, les fèces, l’urine, ou encore la salive."
        
        print("\n--- Méthode avec HyDE ---")
        hyde_answer = rag_system.answer_question(question, definition, use_hyde=True)
        print("\nRéponse avec HyDE:")
        print(hyde_answer)
        
        print("\n--- Méthode standard ---")
        standard_answer = rag_system.answer_question(question, definition, use_hyde=False)
        print("\nRéponse standard:")
        print(standard_answer)
