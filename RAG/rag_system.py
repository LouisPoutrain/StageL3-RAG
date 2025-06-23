"""
Ce fichier définit les classes principales du système RAG.

- RAGSystem: Classe de base contenant les composants fondamentaux (document store,
  embedder, retriever) et les méthodes d'indexation et HyDE.
- RefineRAGSystem: Spécialisation de RAGSystem qui implémente une méthode
  d'analyse pour extraire et évaluer les protocoles scientifiques.
- RAGNonInvasiveDetection: Spécialisation de RAGSystem conçue pour analyser
  si un article scientifique présente un protocole comme étant non-invasif.
"""

import os
import json
from typing import List, Dict, Any, Optional
import pandas as pd
import re
import numpy as np
import gc
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.dataclasses import Document, ChatMessage
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
import nltk
from nltk.tokenize import sent_tokenize
from UniversityLLMAdapter import UniversityLLMAdapter
from pipelines import create_hyde_pipeline, create_indexing_pipeline
from components import DynamicThresholdAdapter
import prompts



def chunk_text(text, max_length=1000, overlap=200):
    """
    Découpe un texte en chunks de phrases ne dépassant pas max_length caractères,
    avec un chevauchement de 'overlap' caractères entre chaque chunk.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_length:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            # Appliquer le chevauchement
            if overlap > 0:
                # Prendre les derniers 'overlap' caractères du chunk précédent
                overlap_start = max(0, len(current_chunk) - overlap)
                current_chunk = current_chunk[overlap_start:] + " " + sentence
            else:
                current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


expected_keys = {
    "protocole", "extrait_pertinent", "echantillon",
    "impacts_potentiels", "evaluation_invasivite",
    "peches_identifies", "taux_de_confiance", "justification_invasivite"
}

expected_keys_invasive = {
    "annonce_invasivite", "justification_annonce_invasivite"
}

def is_valid_entry(entry):
    """Vérifie que toutes les clés attendues sont présentes dans l'entrée"""
    return isinstance(entry, dict) and (expected_keys.issubset(entry.keys()) or expected_keys_invasive.issubset(entry.keys()))

def extract_json_blocks(text):
    """Extrait tous les blocs JSON entourés de balises ```json ... ```"""
    matches = re.findall(r"```json(.*?)```", text, re.DOTALL)
    return [m.strip() for m in matches]

def safe_load_json(text):
    """Charge un bloc JSON, filtre les objets malformés, conserve uniquement les clés attendues"""
    try:
        obj = json.loads(text)
        result = []

        if isinstance(obj, dict): # Dans le cas où l'objet correspond à un dictionnaire 
            if is_valid_entry(obj):
                result.append({k: v for k, v in obj.items() if k in expected_keys or k in expected_keys_invasive})

        elif isinstance(obj, list) :# Dans le cas où l'objet correspond à une liste
            for entry in obj:
                if is_valid_entry(entry):
                    result.append({k: v for k, v in entry.items() if k in expected_keys or k in expected_keys_invasive})

        return result
    except json.JSONDecodeError as e:
        print(f"Erreur de décodage JSON : {str(e)}")
        return []

def parse_json(response_text, filename, invasive_detection: bool = False):
    """Parse le texte de réponse en extrayant les blocs JSON valides"""
    json_blocks = extract_json_blocks(response_text)
    parsed_blocks = []

    # 1. Essayer avec les balises ```json
    for block in json_blocks:
        cleaned_jsons = safe_load_json(block)
        parsed_blocks.extend(cleaned_jsons)

    # 2. si aucun bloc trouvé ou valide, essayer tout le texte
    if not parsed_blocks:
        cleaned_jsons = safe_load_json(response_text)
        parsed_blocks.extend(cleaned_jsons)

    # 3. Retour pour mode invasif
    if invasive_detection:
        return cleaned_jsons 

    # 4. Si rien à parser
    if not parsed_blocks:
        return pd.DataFrame()

    # 5. Conversion DataFrame
    df = pd.DataFrame([
        {
            "Filename": filename,
            "Protocole": item["protocole"],
            "echantillon": item["echantillon"],
            "Impacts potentiels": item["impacts_potentiels"],
            "Extrait pertinent": item["extrait_pertinent"],
            "Évaluation d'invasivité": item["evaluation_invasivite"],
            "Justification d'invasivité": item.get("justification_invasivite", ""),
            "Péchés identifiés": item["peches_identifies"],
            "Taux de confiance": item.get("taux_de_confiance", 0),
        }
        for item in parsed_blocks
    ])

    return df


class RAGSystem:
    """
    Classe de base pour le système RAG. Gère l'initialisation des composants
    Haystack (store, embedder, retriever), l'indexation des documents
    et la recherche sémantique HyDE.
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

        # Variables pour suivre les impacts
        self.impacts_energy = []
        self.impacts_co2 = []
        
        if not self.api_key or not self.api_url:
            raise ValueError("API key and URL must be provided or set as environment variables")
        
        # Cache pour les embeddings
        self.embedding_cache = {}
        
        # Optimisation de l'utilisation de la mémoire
        self._clean_memory()
   
        
        # Initialisation des embedders 
        self.embedder_model = "sentence-transformers/allenai-specter"  #BAAI/bge-large-en-v1.5
        embedding_dim = 1024
        try:
            # Initialisation du document store en mémoire, avec une fonction de similarité cosinus
            self.document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
            # Initialisation du modèle d'embedding
            self.text_embedder = SentenceTransformersTextEmbedder(model=self.embedder_model)
            self.text_embedder.warm_up()
        except Exception as e:
            print(f"Avertissement lors de l'initialisation de l'embedder: {str(e)}")
            # Un modèle plus léger si nécessaire
            self.embedder_model = "sentence-transformers/all-MiniLM-L6-v2"
            print(f"Utilisation du modèle d'embedding de fallback: {self.embedder_model}")
            self.text_embedder = SentenceTransformersTextEmbedder(model=self.embedder_model)
            self.text_embedder.warm_up()
            
            # Réinitialisation avec InMemoryDocumentStore
            self.document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
        
        # Initialisation du retriever et du writer
        self.retriever = InMemoryEmbeddingRetriever(document_store=self.document_store)
        self.writer = DocumentWriter(document_store=self.document_store)
        
        # Initialisation des pipelines
        self.hyde_pipeline = create_hyde_pipeline(self.api_key, self.api_url, self.embedder_model, invasive_detection=False)
        self.hyde_pipeline_invasive_detection = create_hyde_pipeline(self.api_key, self.api_url, self.embedder_model, invasive_detection=True)
        
        # Création de l'adaptateur LLM pour la génération de réponses
        self.llm_adapter = UniversityLLMAdapter(
            api_key=self.api_key,
            api_url=self.api_url,
            max_tokens=20000,
            temperature=0
        )
        
        # Nettoyage final
        self._clean_memory()

    def _clean_memory(self):
        """Nettoie la mémoire pour éviter les problèmes d'OOM"""
        import psutil
        if psutil.virtual_memory().percent > 80:
            gc.collect()
        
            # Import de torch dans la portée locale pour éviter l'erreur UnboundLocalError
            import torch
        
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                # Pour les systèmes macOS 
                try:
                    import torch.mps
                    torch.mps.empty_cache()
                except:
                    pass





    def index_from_json(self, json_path: str) -> int:
        """
        Indexe les documents à partir d'un fichier JSON.
        Le texte est découpé en chunks et traité par lots pour optimiser la mémoire.
        """
        try:
            # Lecture du fichier JSON contenant les documents à indexer
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            documents = []
            # Pour chaque entrée du JSON, découpe le texte en morceaux (chunks)
            for i, entry in enumerate(data):
                chunks = chunk_text(entry.get("text", ""), max_length=4000)
                for j, chunk in enumerate(chunks):
                    doc_id = f"doc_{i}_{j}"
                    document = Document(id=doc_id, content=chunk)
                    documents.append(document)            

            # Création de la pipeline d'indexation
            indexing_pipeline = create_indexing_pipeline(self.document_store, self.embedder_model)
            
            # Optimisation de la taille des lots pour ne pas surcharger la mémoire
            batch_size = 10  
            total_docs = len(documents)
            indexed_count = 0
            
            # Traitement des documents par lots
            for i in range(0, total_docs, batch_size):
                batch_end = min(i + batch_size, total_docs)
                print(f"Traitement du lot {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size} (documents {i+1}-{batch_end})")
                
                try:
                    # Indexation du lot courant
                    batch_docs = documents[i:batch_end]
                    indexing_pipeline.run({"doc_embedder": {"documents": batch_docs}})
                    indexed_count += len(batch_docs)
                    print(f"Lot {i//batch_size + 1} indexé avec succès: {len(batch_docs)} documents")
                except Exception as e:
                    print(f"Erreur lors de l'indexation du lot {i//batch_size + 1}: {str(e)}")
                    
                    # Si une erreur survient, on tente d'indexer chaque document du lot individuellement
                    for j in range(i, batch_end):
                        try:
                            indexing_pipeline.run({"doc_embedder": {"documents": [documents[j]]}})
                            indexed_count += 1
                            print(f"Document {j+1} indexé individuellement")
                        except Exception as doc_e:
                            print(f"Impossible d'indexer le document {j+1}: {str(doc_e)}")
                
                # Nettoyage de la mémoire entre les lots
                self._clean_memory()

            print(f"{indexed_count}/{total_docs} documents indexés depuis {json_path}")
            return indexed_count
        except Exception as e:
            import traceback
            print(f"Erreur lors de l'indexation du fichier {json_path}:")
            print(traceback.format_exc())
            return 0

    def retrieve_with_hyde(self, question: str, top_k: int = 6, invasive_detection: bool = False, title: str = "") -> List[Document]:
        """
        Récupère les documents en utilisant la méthode HyDE (Hypothetical Document Embeddings).
        Génère une réponse hypothétique pour améliorer la pertinence de la recherche.
        Applique ensuite un seuil de similarité dynamique pour filtrer les résultats
        et ne garder que les documents les plus pertinents.
        """
        # Générer le document hypothétique avec HyDE
        if invasive_detection:
            hyde_output = self.hyde_pipeline_invasive_detection.run({
                "prompt_builder": {"question": question, "title": title},
                "generator": {"n_generations": 3}
            })
        else:
            hyde_output = self.hyde_pipeline.run({
            "prompt_builder": {"question": question, "title": title},
            "generator": {"n_generations": 3}
        })
                    
        # Récupérer l'embedding hypothétique moyen
        hyp_embedding = hyde_output["hyde"]["hypothetical_embedding"]
        
        # Récupérer plus de documents que nécessaire pour avoir une meilleure distribution
        initial_k = min(top_k * 2, 10)  
        retrieval_output = self.retriever.run(
            query_embedding=hyp_embedding,
            top_k=initial_k,
        )
        
        # Extraire les scores de similarité
        scores = [doc.score for doc in retrieval_output["documents"]]
        
        # Afficher les statistiques des scores avant filtrage
        print("\n=== Statistiques des scores de similarité ===")
        print(f"Score minimum initial: {min(scores):.4f}")
        print(f"Score maximum initial: {max(scores):.4f}")
        print(f"Score moyen initial: {np.mean(scores):.4f}")
        
        # Appliquer le filtrage dynamique
        threshold_adapter = DynamicThresholdAdapter(min_threshold=0.3, percentile=50)
        filtered_output = threshold_adapter.run(
            documents=retrieval_output["documents"],
            scores=scores
        )
        
        # Extraire les scores des documents filtrés
        filtered_scores = [doc.score for doc in filtered_output["filtered_documents"]]
        if filtered_scores:
            print("\n=== Statistiques après filtrage dynamique ===")
            print(f"Seuil dynamique appliqué: {max(0.3, np.percentile(scores, 60)):.4f}")
            print(f"Score minimum conservé: {min(filtered_scores):.4f}")
            print(f"Score maximum conservé: {max(filtered_scores):.4f}")
            print(f"Score moyen conservé: {np.mean(filtered_scores):.4f}")
            print(f"Nombre de documents conservés: {len(filtered_scores)}")
        else:
            print("\nAucun document n'a passé le seuil de filtrage")
        
        # Limiter au nombre demandé
        return filtered_output["filtered_documents"][:top_k]
    

class RefineRAGSystem(RAGSystem):
    """
    Implémente une pipeline d'analyse multi-étapes ("refine") pour une
    extraction et une évaluation détaillées des protocoles.
    """
    
    def __init__(self, api_key=None, api_url=None):
        super().__init__(api_key, api_url)

    def build_protocol_detection_prompt(self, chunk: str) -> str:
        """
        Construit le prompt pour le premier niveau de raffinement
        qui détecte si le chunk contient un protocole d'échantillonnage ADN
        """
        return prompts.PROTOCOL_DETECTION_PROMPT.format(chunk=chunk)

    def build_fusion_extrait_prompt(self, extracted_chunks: List[str]) -> str:
        """
        Construit le prompt pour fusionner les extraits similaires (même protocole) dans un seul json afin de faire qu'une analyse par protocoles.
        """
        return prompts.FUSION_EXTRAIT_PROMPT.format(extracted_chunks=extracted_chunks)


    
    def build_invasivity_analysis_prompt(self, protocol: str, definition: str) -> str:
        """
        Construit le prompt pour le deuxième niveau de raffinement
        qui analyse l'invasivité du protocole
        """
        return prompts.INVASIVITY_ANALYSIS_PROMPT.format(definition=definition, protocol=protocol)

    
    def build_fusion_prompt(self, analyses: List[Dict]) -> str:
        """
        Construit le prompt pour le troisième niveau de raffinement
        qui fusionne les analyses en un tableau JSON final
        """
        return prompts.FUSION_PROMPT.format(analyses=analyses)

    
    def refine_analysis(self, question: str, definition: str = "", top_k: int = 4, title: str = "") -> str:
        """
        Applique une méthode d'analyse et de raffinement en plusieurs étapes :
        1.  Récupère les chunks de texte pertinents via HyDE.
        2.  Détecte les protocoles potentiels dans chaque chunk.
        3.  Fusionne les extraits décrivant un même protocole.
        4.  Analyse l'invasivité de chaque protocole identifié.
        5.  Fusionne les analyses finales en un seul JSON consolidé.
        """
        # Récupération des chunks pertinents
        documents = self.retrieve_with_hyde(question, top_k, title=title)
        chunks = [doc.content for doc in documents]
        print(f"Documents récupérés : {len(chunks)}")

        #  Détection des protocoles
        protocols = []
        for chunk in chunks:
            prompt = self.build_protocol_detection_prompt(chunk)
            response, impact = self.llm_adapter.generate_answer_with_impact(prompt)
            self.impacts_energy.append(impact["energy_kwh"])
            self.impacts_co2.append(impact["co2_g"])
            print(f"Chunk : {chunk}")
            print(f"Réponse du modèle : {response}")
            protocols.append(response)

        if not protocols:
            print("Aucun protocole détecté.")
            return "[]"

        print(f"Protocoles détectés : {len(protocols)}")

        # Fusion des extraits similaires
        fusion_prompt = self.build_fusion_extrait_prompt(protocols)
        fused_protocols_response, impact = self.llm_adapter.generate_answer_with_impact(fusion_prompt)
        self.impacts_energy.append(impact["energy_kwh"])
        self.impacts_co2.append(impact["co2_g"])
        print(f"Protocoles fusionnés : {fused_protocols_response}")

        # Parsing des protocoles fusionnés
        fused_protocols = []
        try:
            # Extraction des blocs JSON
            json_blocks = extract_json_blocks(fused_protocols_response)
            for block in json_blocks:
                try:
                    protocol = json.loads(block)
                    if protocol.get("contient_protocole", False):
                        fused_protocols.append(protocol)
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            print(f"Erreur lors du parsing des protocoles fusionnés : {str(e)}")
            fused_protocols = protocols 

        print(f"Nombre de protocoles après fusion : {len(fused_protocols)}")
 
        #  Analyse de l'invasivité
        analyses = []
        for protocol in fused_protocols:
            # Vérifier que le protocole contient des extraits pertinents
            if not protocol.get("extrait_pertinent"):
                continue
                
            protocol_text = "\n".join(protocol["extrait_pertinent"])
            
            prompt = self.build_invasivity_analysis_prompt(protocol_text, definition)
            response, impact = self.llm_adapter.generate_answer_with_impact(prompt)
            self.impacts_energy.append(impact["energy_kwh"])
            self.impacts_co2.append(impact["co2_g"])
            print(f"Analyse du protocole avec les extraits : {protocol_text}")
            print(f"Réponse du modèle : {response}")
            analyses.append(response)

        if not analyses:
            print("Aucune analyse générée.")
            return "[]"

        print(f"Analyses générées : {len(analyses)}")

        #  Fusion des analyses
        prompt = self.build_fusion_prompt(analyses)
        final_response, impact = self.llm_adapter.generate_answer_with_impact(prompt)
        self.impacts_energy.append(impact["energy_kwh"])
        self.impacts_co2.append(impact["co2_g"])
        print(f"Réponse finale : {final_response}")

        return final_response




class RAGNonInvasiveDetection(RAGSystem):
    """
    Spécialisation du RAG conçue pour une tâche précise : vérifier si les
    auteurs présente un protocole (déjà identifié comme invasif) comme
    étant "non-invasif", afin de détecter des contradictions.
    """
    def __init__(self, api_key=None, api_url=None):
        super().__init__(api_key, api_url)

    def build_prompt(self, Chunks : List[str], protocole : str, title : str):
        """
        Construit le prompt
        """
        return prompts.NON_INVASIVE_DETECTION_PROMPT.format(Chunks=Chunks, protocole=protocole, title=title)
    
    def detect_non_invasive_level(self, Chunks: List[str], protocole: str, title: str, impacts_energy: List[float] = None, impacts_co2: List[float] = None):
        """
        Détecte l'annonce de l'invasivité du protocole faite par les auteurs
        """
        if impacts_energy is None:
            impacts_energy = []
        if impacts_co2 is None:
            impacts_co2 = []
            
        if not Chunks:
            return "Aucun extrait disponible pour l'analyse"
        
        try:
            prompt = self.build_prompt(Chunks, protocole, title)
            print(f"Prompt : {prompt}")
            response, impact = self.llm_adapter.generate_answer_with_impact(prompt)
            impacts_energy.append(impact["energy_kwh"])
            impacts_co2.append(impact["co2_g"])
            return response
        except Exception as e:
            print(f"Erreur lors de la détection du niveau non invasif : {str(e)}")
            return f"Erreur lors de l'analyse : {str(e)}"
    
    def answer_question(self, question : str, protocole : str = "", top_k : int = 8, title : str = "") -> str:
        """
        Répond à la requête
        """
        documents = self.retrieve_with_hyde(question, top_k, invasive_detection=True, title=title)

        chunks = [doc.content for doc in documents]
        return self.detect_non_invasive_level(chunks, protocole, title)

