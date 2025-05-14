"""
Système RAG principal
"""

import os
import json
from typing import List, Dict, Any, Optional
import pandas as pd
import re
import numpy as np
import gc

from haystack.dataclasses import Document, ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.components.preprocessors import DocumentSplitter

from UniversityLLMAdapter import UniversityLLMAdapter
from pipelines import create_hyde_pipeline, create_rag_pipeline, create_indexing_pipeline, PROTOCOL_SCHEMA


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
        
        # Optimisation de l'utilisation de la mémoire
        self._clean_memory()
            
        # Initialisation du store de documents
        self.document_store = InMemoryDocumentStore()
        
        # Initialisation des embedders 
        # Utilisation d'un modèle d'embedding plus performant
        self.embedder_model = "BAAI/bge-large-en-v1.5"
        try:
            self.text_embedder = SentenceTransformersTextEmbedder(model=self.embedder_model)
            self.text_embedder.warm_up()
        except Exception as e:
            print(f"Avertissement lors de l'initialisation de l'embedder: {str(e)}")
            # Fallback à un modèle plus léger si nécessaire
            self.embedder_model = "sentence-transformers/all-MiniLM-L6-v2"
            print(f"Utilisation du modèle d'embedding de fallback: {self.embedder_model}")
            self.text_embedder = SentenceTransformersTextEmbedder(model=self.embedder_model)
            self.text_embedder.warm_up()
        
        # Initialisation du retriever et du writer
        self.retriever = InMemoryEmbeddingRetriever(document_store=self.document_store)
        self.writer = DocumentWriter(document_store=self.document_store)
        
        # Initialisation et activation du reranker
        try:
            self.reranker = TransformersSimilarityRanker(model="cross-encoder/ms-marco-MiniLM-L-6-v2")
            self.reranker.warm_up()
        except Exception as e:
            print(f"Erreur lors de l'initialisation du reranker: {str(e)}")
            self.reranker = None
            print("Le reranker a été désactivé pour économiser de la mémoire")

        # Amélioration du splitter pour une meilleure segmentation
        # Augmentation de la taille du split et du chevauchement pour un meilleur contexte
        self.splitter = DocumentSplitter(split_by="word", split_length=800, split_overlap=150)

        # Initialisation des pipelines
        self.hyde_pipeline = create_hyde_pipeline(self.api_key, self.api_url, self.embedder_model)
        self.rag_pipeline = create_rag_pipeline(self.text_embedder, self.retriever, self.splitter)
        
        # Création de l'adaptateur LLM pour la génération de réponses
        self.llm_adapter = UniversityLLMAdapter(
            api_key=self.api_key,
            api_url=self.api_url,
            max_tokens=14000,
            temperature=0
        )
        
        # Nettoyage final
        self._clean_memory()

    def _clean_memory(self):
        """Nettoie la mémoire pour éviter les problèmes d'OOM"""
        gc.collect()
        
        # Import de torch dans la portée locale pour éviter l'erreur UnboundLocalError
        import torch
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Pour les systèmes macOS avec Apple Silicon
            try:
                import torch.mps
                torch.mps.empty_cache()
            except:
                pass

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
        
        # Traitement par lots pour éviter les problèmes de mémoire
        batch_size = 10  # Taille de lot réduite pour éviter les problèmes de mémoire
        total_docs = len(documents)
        
        for i in range(0, total_docs, batch_size):
            batch_end = min(i + batch_size, total_docs)
            print(f"Traitement du lot {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size} (documents {i+1}-{batch_end})")
            
            try:
                # Exécution de la pipeline sur un lot
                batch_docs = documents[i:batch_end]
                indexing_pipeline.run({"doc_embedder": {"documents": batch_docs}})
            except Exception as e:
                print(f"Erreur lors de l'indexation du lot {i//batch_size + 1}: {str(e)}")
                
                # Si l'erreur est liée à la mémoire, réduire encore la taille du lot
                if "out of memory" in str(e).lower():
                    sub_batch_size = max(1, batch_size // 2)
                    print(f"Tentative avec une taille de lot réduite: {sub_batch_size}")
                    
                    for j in range(i, batch_end, sub_batch_size):
                        sub_batch_end = min(j + sub_batch_size, batch_end)
                        sub_batch_docs = documents[j:sub_batch_end]
                        
                        try:
                            indexing_pipeline.run({"doc_embedder": {"documents": sub_batch_docs}})
                            print(f"  Sous-lot {j-i+1}-{sub_batch_end-i} traité avec succès")
                        except Exception as sub_e:
                            print(f"  Erreur sur le sous-lot: {str(sub_e)}")
                            # En dernier recours, traiter un par un
                            for k in range(j, sub_batch_end):
                                try:
                                    indexing_pipeline.run({"doc_embedder": {"documents": [documents[k]]}})
                                    print(f"    Document {k+1} traité individuellement")
                                except Exception as doc_e:
                                    print(f"    Impossible d'indexer le document {k+1}: {str(doc_e)}")

        # Vérification des embeddings
        indexed_docs = self.document_store.filter_documents()
        print(f"{len(indexed_docs)}/{len(documents)} documents indexés depuis {json_path}")
        
        return len(indexed_docs)

    def retrieve_with_hyde(self, question: str, top_k: int = 6) -> List[Document]:
        """
        Récupère les documents pertinents en utilisant HyDE
        
        Args:
            question: La question de l'utilisateur
            top_k: Nombre de documents à récupérer
            
        Returns:
            Liste des documents récupérés
        """
        # Nettoyage préventif de la mémoire
        self._clean_memory()

        print("debut de retrieve with hyde")
        try:
            # Génération et embedding du document hypothétique avec HyDE
            hyde_output = self.hyde_pipeline.run({
                "prompt_builder": {"question": question},
                "generator": {"n_generations": 2}  # Réduction du nombre pour éviter les problèmes de mémoire
            })
            
            # Log pour le debugging
            print("Clés disponibles dans hyde_output:", list(hyde_output.keys()))
            
            # Vérification de la présence des clés nécessaires
            if "hyde" not in hyde_output or "hypothetical_embedding" not in hyde_output["hyde"]:
                print("Erreur: Embedding hypothétique non trouvé dans la sortie HyDE")
                # Fallback vers la méthode standard
                return self.retrieve_standard(question, top_k)
            
            # Récupération de l'embedding hypothétique moyen
            hyp_embedding = hyde_output["hyde"]["hypothetical_embedding"]
            
            # Nettoyage intermédiaire
            self._clean_memory()
            
            # Récupération des documents avec l'embedding hypothétique
            retrieved_docs = self.retriever.run(
                query_embedding=hyp_embedding, 
                top_k=top_k
            )["documents"]
            
            # Affichage des documents hypothétiques générés
            print("\nDocuments hypothétiques générés:")
            if "generator" in hyde_output and "documents" in hyde_output["generator"]:
                for i, doc in enumerate(hyde_output["generator"]["documents"]):
                    print(f"Document {i+1}: {doc.content[:100]}...")
            
            return retrieved_docs
            
        except Exception as e:
            print(f"Erreur lors de la récupération avec HyDE: {str(e)}")
            print("Fallback vers la méthode de récupération standard")
            self._clean_memory()
            return self.retrieve_standard(question, top_k)
    
    def retrieve_standard(self, question: str, top_k: int = 6) -> List[Document]:
        """
        Récupère les documents pertinents avec la méthode standard
        
        Args:
            question: La question de l'utilisateur
            top_k: Nombre de documents à récupérer
            
        Returns:
            Liste des documents récupérés
        """
        try:
            results = self.rag_pipeline.run(
                data={
                    "text_embedder": {"text": question}, 
                    "retriever": {"top_k": top_k*2}
                }
            )
            
            if "splitter" in results and "documents" in results["splitter"]:
                retrieved_docs = results["splitter"]["documents"]
                return retrieved_docs[:top_k]  # Limiter aux top_k documents
            else:
                print("Avertissement: Aucun document récupéré par la méthode standard")
                return []
        except Exception as e:
            print(f"Erreur lors de la récupération standard: {str(e)}")
            # Fallback vers la méthode directe
            query_embedding = self.text_embedder.run(text=question)["embedding"]
            retrieved_docs = self.retriever.run(
                query_embedding=query_embedding, 
                top_k=top_k
            )["documents"]
            return retrieved_docs
    
    def build_context(self, documents: List[Document], max_length: int = 4000) -> str:
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
        ### >>> INSTRUCTIONS POUR LE MODELE SEULEMENT <<<  
        # NE PAS INTERPRÉTER CE QUI SUIT COMME UN EXTRAIT À ANALYSER.
        # TU DOIS ATTENDRE LE CONTEXTE RÉEL DANS LA SECTION : CONTEXTE À ANALYSER
        # TU NE DOIS TRAITER QUE LE TEXTE FOURNI DANS LA VARIABLE {{context}}
        # IGNORE LE PRÉSENT PROMPT POUR L'ÉTAPE REACT.

        Tu es un assistant scientifique expert dans l'analyse des méthodes d'échantillonnage d'ADN. Ta tâche est d'examiner les protocoles décrits dans les extraits pour déterminer s'ils contreviennent aux " Péchés de l'Échantillonnage d'ADN Non-Invasif".

        Le but est de produire une analyse précise et un JSON valide pour chaque protocole d'échantillonnage d'ADN selon la structure demandée.

        Tu dois suivre strictement le format REACT (Reasoning and Acting) en séparant clairement chaque étape :

        ## Thought: [Réflexion approfondie sur le protocole identifié]  
        ## Act: [Analyse détaillée du protocole avec référence spécifique aux péchés]  
        ## Obs: [Citations exactes des extraits avec numéros d'extraits correspondants]

        ##################################################
        # DÉFINITIONS ET CONTEXTE
        ##################################################

        Définition de l'échantillonnage non-invasif :  
        {definition}

        Définitions des péchés :  
        {seven_sins_definitions}

        Important :  
        1. Toute action modifiant le comportement animal = invasive  
        2. Tout contact direct avec l'animal = TOUJOURS invasif  
        3. Toute capture = TOUJOURS invasive  
        4. Échantillonnage non invasif = UNIQUEMENT sans contact avec l'animal  
        5. Prélèvement de fèces = invasif SEULEMENT si perturbation de l'animal ou du marquage territorial

        ##################################################
        # RÈGLES SUPPLÉMENTAIRES D'INTERPRÉTATION
        ##################################################

        1. Radio-suivi :  
        - Si l'animal est localisé grâce à un marquage radio **réalisé avant le protocole**, ce n'est **pas invasif**.  
        - Si le marquage est **effectué dans le cadre du protocole**, il est **invasif**.

        2. Prélèvement de fèces de mammifères :  
        - Si toutes les fèces sont prélevées : invasif (perturbation du marquage territorial).  
        - Si seule une partie est prélevée : non invasif.  
        - Si aucune précision n'est donnée (ex : "fèces collectées après le passage de l'animal") :  
            ALORS "evaluation_invasivite": "présumé non invasif"  
        - Si le contexte de la collecte des fèces implique une perturbation de l'animal (utilisation d'aéronefs, capture), alors :  
            "evaluation_invasivite": "Invasif"

        3. **Autres prélèvements (poils, salive, urine...)** :  
        - Si les informations sont insuffisantes (pas de mention de capture ou manipulation) :  
            "evaluation_invasivite": "présumé non invasif"  
            avec justification adaptée.

        ##################################################
        # CONTEXTE À ANALYSER
        ##################################################

        {context}

        ##################################################
        # INSTRUCTIONS D'ANALYSE — FORMAT REACT
        ##################################################

        1. Identifie avec précision chaque protocole d'échantillonnage d'ADN distinct.
        2. Regroupe IMPÉRATIVEMENT TOUS les protocoles qui décrivent des méthodes d'échantillonnage similaires.
        3. Pour chaque protocole, réalise une analyse en trois parties :
        - Thought : formule une réflexion approfondie sur la méthode et ses implications
        - Act : analyse rigoureusement la méthode, ses impacts sur l'animal, et identifie les péchés spécifiques
        - Obs : cite EXACTEMENT les extraits pertinents avec leurs numéros d'extraits correspondants

        4. Ne traite qu'un SEUL protocole par article, selon les priorités décrites.

        5. Ignore toute méthode ne concernant pas directement l'échantillonnage d'ADN (ex : PCR, séquençage, photographie, etc.).

        ##################################################
        # FORMAT DE RÉPONSE
        ##################################################

        ```json
        {{
            "protocole": "Nom concis du protocole",
            "extrait_pertinent": ["Texte exact de l'extrait 1", "Texte exact de l'extrait 2"],
            "echantillon": "Type précis d'échantillon (poils/sang/fèces/urine/salive/mixte)",
            "impacts_potentiels": ["impact1", "impact2"],
            "evaluation_invasivite": "Non invasif / Invasif / présumé non invasif",
            "peches_identifies": ["1", "2", "5"]
        }}
        ```

        ##################################################
        VÉRIFICATION DU JSON
        ##################################################
        Avant de soumettre ton JSON, vérifie STRICTEMENT:

        - Absence de clés dupliquées
        - Fermeture correcte des accolades et crochets
        - Placement approprié des virgules
        - Validité parfaite du format JSON
        - Format correct des listes pour les champs "extrait_pertinent", "impacts_potentiels" et "peches_identifies"

        IMPORTANT:
        - Utiliser "evaluation_invasivite" (sans accent)
        - Utiliser "echantillon" (sans accent)
        - Utiliser "peches_identifies" (sans accent)
        - Les extraits pertinents doivent toujours être une liste, même pour un seul élément

        ### Exemple :

        ## Thought: Le protocole indique que les fèces ont été prélevées en totalité à proximité du terrier.
        ## Act: Le prélèvement complet des fèces perturbe le marquage territorial. Cette méthode, bien que passive, a un impact comportemental. Elle est donc considérée comme invasive. Péché 1 est applicable.
        ## Obs: "Toutes les fèces ont été ramassées à la sortie du terrier."

        ```json
            {{
            "protocole": "Prélèvement total de fèces au terrier",
            "extrait_pertinent": ["Toutes les fèces ont été ramassées à la sortie du terrier."],
            "echantillon": "fèces",
            "impacts_potentiels": ["perturbation du marquage territorial"],
            "evaluation_invasivite": "Invasif",
            "peches_identifies": ["1"]
            }}
        ```

        Donne uniquement le JSON bien formé, sans texte explicatif autour.
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
    




    

    def answer_question(self, question: str, definition: str = "", use_hyde: bool = True, top_k: int = 8, max_retries: int = 3) -> str:
        """
        Processus complet pour répondre à une question avec validation JSON
        
        Args:
            question: La question à traiter
            definition: Définition optionnelle à inclure dans le prompt
            use_hyde: Utiliser la méthode HyDE ou standard
            top_k: Nombre de documents à récupérer
            max_retries: Nombre maximum de tentatives en cas d'échec de validation
            
        Returns:
            La réponse validée ou une erreur après max_retries tentatives
        """
        # Nettoyage de la mémoire avant de commencer
        self._clean_memory()
        
        for attempt in range(max_retries):
            if use_hyde:
                print("--- Méthode avec HyDE ---")
                retrieved_docs = self.retrieve_with_hyde(question=question, top_k=top_k)
                print("\n--- DOCUMENTS RÉCUPÉRÉS (HYDE) ---")
                for doc in retrieved_docs:
                    print(f"Contenu: {doc.content[:200]}...")
                print("--- FIN DES DOCUMENTS RÉCUPÉRÉS (HYDE) ---\n")
            else:
                print("--- Méthode standard ---")
                retrieved_docs = self.retrieve_standard(question=question, top_k=top_k)
                print(f"\n--- DOCUMENTS RÉCUPÉRÉS (STANDARD) ---")
                print(f"Nombre de documents récupérés: {len(retrieved_docs)}")
                for doc in retrieved_docs:
                    print(f"Contenu: {doc.content[:200]}...")
                print("--- FIN DES DOCUMENTS RÉCUPÉRÉS (STANDARD) ---\n")

            # Nettoyage intermédiaire
            self._clean_memory()

            # Construction du contexte
            context = self.build_context(retrieved_docs)
            
            # Génération de la réponse
            answer = self.generate_answer(question, context, definition)
            
            # Validation du JSON
            try:
                # Extraction des blocs JSON
                json_blocks = extract_json_blocks(answer)
                if not json_blocks:
                    print(f"Tentative {attempt + 1}/{max_retries}: Aucun bloc JSON trouvé")
                    continue
                    
                # Validation de chaque bloc JSON
                for block in json_blocks:
                    try:
                        json_obj = json.loads(block)
                        # Vérification des champs requis
                        if all(key in json_obj for key in ["protocole", "extrait_pertinent", "echantillon", 
                                                         "impacts_potentiels", "evaluation_invasivite", 
                                                         "peches_identifies"]):
                            # Vérification des types
                            if (isinstance(json_obj["extrait_pertinent"], list) and 
                                isinstance(json_obj["impacts_potentiels"], list) and 
                                isinstance(json_obj["peches_identifies"], list) and
                                json_obj["evaluation_invasivite"] in ["Non invasif", "Invasif", "présumé non invasif"]):
                                self._clean_memory()  # Nettoyage final avant de retourner la réponse
                                return answer
                    except json.JSONDecodeError:
                        continue
                        
                print(f"Tentative {attempt + 1}/{max_retries}: JSON invalide")
                
            except Exception as e:
                print(f"Tentative {attempt + 1}/{max_retries}: Erreur de validation - {str(e)}")
                
            if attempt < max_retries - 1:
                print("Nouvelle tentative avec un prompt modifié...")
                # Nettoyage avant la nouvelle tentative
                self._clean_memory()
                # Modification du prompt pour être plus strict sur le format
                answer = self.generate_answer_with_override(
                    question, 
                    context, 
                    definition,
                    prompt_override=self._build_prompt(question, context, definition) + "\nIMPORTANT: Assurez-vous que le JSON est parfaitement valide et respecte le schéma suivant:\n" + json.dumps(PROTOCOL_SCHEMA, indent=2)
                )
        
        self._clean_memory()  # Nettoyage final
        return "Erreur: Impossible de générer une réponse valide après plusieurs tentatives"



expected_keys = {
    "protocole", "extrait_pertinent", "echantillon",
    "impacts_potentiels", "evaluation_invasivite",
    "peches_identifies"
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
            "Impacts potentiels": item["impacts_potentiels"],
            "Extrait pertinent": item["extrait_pertinent"],
            "Évaluation d'invasivité": item["evaluation_invasivite"],
            "Péchés identifiés": item["peches_identifies"]
        }
        for item in parsed_blocks
    ])

    return df

    
