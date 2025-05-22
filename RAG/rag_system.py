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
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.dataclasses import Document, ChatMessage
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.components.preprocessors import DocumentSplitter
import tqdm

from UniversityLLMAdapter import UniversityLLMAdapter
from pipelines import create_hyde_pipeline, create_rag_pipeline, create_indexing_pipeline, PROTOCOL_SCHEMA
from components import DynamicThresholdAdapter


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
        
        # Cache pour les embeddings
        self.embedding_cache = {}
        
        # Optimisation de l'utilisation de la mémoire
        self._clean_memory()
   
        
        # Initialisation des embedders 
        self.embedder_model = "sentence-transformers/allenai-specter"  #BAAI/bge-large-en-v1.5
        embedding_dim = 1024
        try:
            self.document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
            
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
        

        # Splitter pour la segmentation des documents
        self.splitter = DocumentSplitter(split_by="word", split_length=400, split_overlap=60)

        # Initialisation des pipelines
        self.hyde_pipeline = create_hyde_pipeline(self.api_key, self.api_url, self.embedder_model)
        self.rag_pipeline = create_rag_pipeline(self.text_embedder, self.retriever, self.splitter)
        
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
        Indexe des documents à partir d'un fichier JSON avec optimisation des lots
        """
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            documents = []
            for i, entry in enumerate(data):
                document = Document(
                    id=f"doc_{i}",
                    content=entry.get("text", "")[:3000]
                )
                documents.append(document)

            # Création de la pipeline d'indexation
            indexing_pipeline = create_indexing_pipeline(self.document_store, self.embedder_model)
            
            # Optimisation de la taille des lots
            batch_size = 10  
            total_docs = len(documents)
            indexed_count = 0
            
            for i in range(0, total_docs, batch_size):
                batch_end = min(i + batch_size, total_docs)
                print(f"Traitement du lot {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size} (documents {i+1}-{batch_end})")
                
                try:
                    batch_docs = documents[i:batch_end]
                    indexing_pipeline.run({"doc_embedder": {"documents": batch_docs}})
                    indexed_count += len(batch_docs)
                    print(f"Lot {i//batch_size + 1} indexé avec succès: {len(batch_docs)} documents")
                except Exception as e:
                    print(f"Erreur lors de l'indexation du lot {i//batch_size + 1}: {str(e)}")
                    
                    # Traitement document par document en cas d'erreur
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

    def retrieve_with_hyde(self, question: str, top_k: int = 6) -> List[Document]:
        """
        Récupère les documents pertinents en utilisant HyDE avec un seuil dynamique
        
        Args:
            question: La question de l'utilisateur
            top_k: Nombre maximum de documents à récupérer
            
        Returns:
            Liste des documents récupérés filtrés par le seuil dynamique
        """
        # Générer le document hypothétique avec HyDE
        hyde_output = self.hyde_pipeline.run({
            "prompt_builder": {"question": question},
            "generator": {"n_generations": 3}
        })
        
        # Récupérer l'embedding hypothétique moyen
        hyp_embedding = hyde_output["hyde"]["hypothetical_embedding"]
        
        # Récupérer plus de documents que nécessaire pour avoir une meilleure distribution
        initial_k = min(top_k * 2, 8)  
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
        threshold_adapter = DynamicThresholdAdapter(min_threshold=0.3, percentile=60)
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
            query_embedding = self.text_embedder.run(text=question)["embedding"]
            retrieved_docs = self.retriever.run(
                query_embedding=query_embedding, 
                top_k=top_k
            )["documents"]
            return retrieved_docs
    
    def build_context(self, documents: List[Document], max_length: int = 8000) -> str:
        """
        Construit le contexte à partir des documents récupérés
        """
        context_parts = []
        total_length = 0

        for i, doc in enumerate(documents):
            content = doc.content.strip()
            excerpt_header = f"EXTRAIT {i+1}:\n"
            available_length = max_length - total_length - len(excerpt_header)

            if available_length <= 0:
                break

            # Si le contenu est plus court que la limite dispo, on le garde en entier
            if len(content) <= available_length:
                trimmed_content = content
            else:
                # On essaie de couper à la dernière phrase complète
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

        return "\n\n".join(context_parts)
    
    def _build_prompt(self, question: str, context: str, definition: str = "") -> str:
        """
        Construit le prompt pour le LLM
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

        Tu es un expert en analyse de protocoles d'échantillonnage d'ADN. Ta mission est d'analyser les protocoles décrits dans les extraits fournis pour déterminer s'ils sont invasifs et quels "péchés d'échantillonnage" ils commettent.
        Définitions

        Échantillonnage d'ADN non-invasif:

        {definition}

        Les sept péchés de l'échantillonnage d'ADN:

        {seven_sins_definitions}
        Critères d'invasivité

        Un protocole est invasif si au moins UN des critères suivants est rempli:

            Contact direct avec l'animal vivant
            Capture ou manipulation de l'animal
            Modification du comportement naturel
            Perturbation significative de l'habitat
            Utilisation d'appâts ou leurres modifiant le comportement

        Un protocole est non-invasif UNIQUEMENT si TOUS ces critères sont remplis:

            Aucun contact avec l'animal vivant
            Aucune perturbation du comportement naturel
            Utilisation d'échantillons déjà abandonnés naturellement
            Aucune modification de l'habitat ou du territoire

        Cas particuliers CRITIQUES:

            Radio-suivi: Non invasif si marquage antérieur, invasif si marquage dans le protocole
            Fèces de mammifères: Invasif si collecte totale/quantité importante/non précisée (perturbation marquage territorial)
            ANIMAUX MORTS: TOUJOURS NON INVASIF si mort naturelle sans intervention humaine (aucun contact possible avec animal vivant)
            Échantillons post-mortem: Prélèvements sur cadavres = NON INVASIF par définition (animal déjà mort)
            Nécropsies: Si l'animal est mort naturellement, tous les prélèvements sont NON INVASIFS

        Extraits à analyser

        {context}

        SI extrait NE contient PAS de description explicite d'une méthode de collecte ou de traitement d'un échantillon

        ALORS rejeter le chunk comme "non exploitable"
        Processus d'analyse avec ReAct

        Pour chaque protocole d'échantillonnage distinct mentionné dans les extraits:

            Identifier clairement le protocole distinct dans les extraits
            Appliquer la méthode ReAct:

            Thought: [Réflexion approfondie sur le protocole identifié]
            Act: [Analyse détaillée du protocole avec référence spécifique aux péchés]
            Obs: [Citations exactes des extraits avec numéros d'extraits correspondants]

            Structurer l'analyse finale au format JSON

        SI tu ne trouves pas de protocole dans les extraits, tu dois renvoyer un tableau vide.
        Erreurs fréquentes à ÉVITER ABSOLUMENT

        ATTENTION - Règles d'évaluation critiques:

            ANIMAUX MORTS

            Non invasif si mort naturelle;
            Cadavres, nécropsies, organes (ex. intestins) : non invasif
            Aucun contact avec animal vivant, aucune modification de comportement

            Cas invasifs:

        Animal tué pour l'étude = invasif
        Mort causée par l'humain (ex. chasse, pollution, capture) = invasif

            Péchés:

        Aucun péché si mort naturelle
        Péché si mort liée à une intervention humaine ou provoquée par l'étude

            Regroupement des protocoles:

            NE PAS créer 3 protocoles différents si c'est le même type de prélèvement
            Exemple: "fèces pré-mortem", "fèces post-mortem", "fèces nécropsie" = UN SEUL protocole "Collecte de fèces"

            Péchés sur les fèces pour les mammifères :

            Si l'animal est un mammifère, il est invasif si la collecte est totale ou non précisée.
            En revanche si seulement une partie des fèces est prélevée, il est non invasif.

        Règles importantes pour l'analyse

            REGROUPE les protocoles identiques ou similaires (même type d'échantillon, même méthode)
            Citer les extraits exacts dans la langue originale avec assez de précision pour retrouver l'extrait dans le texte
            N'inventer JAMAIS de protocoles non mentionnés dans les extraits
            Ignore toute méthode ne concernant pas directement l'échantillonnage d'ADN (ex : PCR, séquençage, photographie, etc.).
            RAPPEL: Échantillons post-mortem = NON INVASIF par défaut

        Important : Pour chaque protocole, effectue mentalement la méthode ReAct (Thought, Act, Obs) afin de bien analyser le protocole.

        Mais dans ta réponse finale, n'affiche pas ces étapes.

        Affiche uniquement le tableau JSON final complet demandé, sans aucun autre texte ni raisonnement intermédiaire.

        Attention si le protocole d'échantillonnage d'ADN est la collecte de fèces chez les mammifères et qu'il est dit qu'ils prélèvent une partie des fèces, il est non invasif.

        Ne jamais supposer qu'un animal est mort s'il n'est pas explicitement mentionné comme mort dans l'extrait.

        Ne pas créer de protocole à partir d'un extrait s'il ne décrit pas explicitement une méthode de collecte d'échantillon.

        Autrement dit, ignorer tout extrait qui ne fournit aucune information sur le type d'échantillon, la manière dont il a été collecté, ou les conditions de cette collecte (ex. : simple mention d'une approbation éthique ou d'un résultat sans protocole = à ignorer)
        Structure du JSON final

        Fournir un tableau JSON contenant tous les protocoles identifiés:

        ```json
        [
        {{
            "protocole": "Nom précis et descriptif du protocole",
            "extrait_pertinent": ["Citation exacte de l'extrait"],
            "echantillon": "Type d'échantillon prélevé",
            "impacts_potentiels": ["Impact 1", "Impact 2"],
            "evaluation_invasivite": "Invasif" ou "Non invasif",
            "peches_identifies": ["1", "2", "5"]
        }}
        ]
        ```

        Instructions finales

        Ta réponse DOIT contenir:

            Uniquement le tableau JSON final complet
            Toutes les citations dans "extrait_pertinent" DOIVENT être copiées exactement depuis le texte original. Ne jamais traduire ni reformuler. Les citations doivent rester dans la langue d'origine des extraits

        Le JSON final DOIT:

            Être strictement valide (parsable sans erreur)
            Contenir tous les protocoles identifiés
            Utiliser les clés sans accents: "evaluation_invasivite", "echantillon", "peches_identifies"

        Limites strictes:

            Utiliser UNIQUEMENT les informations des extraits fournis
            La citation exacte doit avoir assez de contexte pour comprendre le protocole
        """

    def generate_answer(self, question: str, context: str, definition: str = "") -> str:
        """
        Génère une réponse à partir du contexte et de la question
        """

        
        prompt = self._build_prompt(question, context, definition)
        return self.llm_adapter.generate_answer(prompt)


    def answer_question(self, question: str, definition: str = "", use_hyde: bool = True, top_k: int = 8, max_retries: int = 2) -> str:
        """
        Processus complet pour répondre à une question avec validation JSON
        """
        self._clean_memory()
        
        for attempt in range(max_retries):
            if use_hyde:
                print("--- Méthode avec HyDE ---")
                retrieved_docs = self.retrieve_with_hyde(question=question, top_k=top_k)
                print("\n--- DOCUMENTS RÉCUPÉRÉS (HYDE) ---")
                for doc in retrieved_docs:
                    print(f"Contenu: {doc.content[:500]}...")
                print("--- FIN DES DOCUMENTS RÉCUPÉRÉS (HYDE) ---\n")
            else:
                print("--- Méthode standard ---")
                retrieved_docs = self.retrieve_standard(question=question, top_k=top_k)
                print(f"\n--- DOCUMENTS RÉCUPÉRÉS (STANDARD) ---")
                print(f"Nombre de documents récupérés: {len(retrieved_docs)}")
                for doc in retrieved_docs:
                    print(f"Contenu: {doc.content[:500]}...")
                print("--- FIN DES DOCUMENTS RÉCUPÉRÉS (STANDARD) ---\n")

            context = self.build_context(retrieved_docs)
            
            prompt = self._build_prompt(question, context, definition) 
            answer = self.llm_adapter.generate_answer(prompt)
            print(f"Réponse générée: {answer}")
                        
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
                        # Vérification de la structure
                        if isinstance(json_obj, list):
                            for item in json_obj:
                                if not isinstance(item, dict):
                                    continue
                                # Vérification des clés 
                                required_keys = {"protocole", "extrait_pertinent", "echantillon", 
                                              "impacts_potentiels", "evaluation_invasivite", 
                                              "peches_identifies"}
                                if not required_keys.issubset(item.keys()):
                                    continue
                                # Vérification des types
                                if not (isinstance(item["extrait_pertinent"], list) and 
                                      isinstance(item["impacts_potentiels"], list) and 
                                      isinstance(item["peches_identifies"], list) and
                                      isinstance(item["protocole"], str) and
                                      isinstance(item["echantillon"], str) and
                                      isinstance(item["evaluation_invasivite"], str)):
                                    continue
                                return answer
                    except json.JSONDecodeError:
                        continue
                    
                print(f"Tentative {attempt + 1}/{max_retries}: JSON invalide")
                
            except Exception as e:
                print(f"Tentative {attempt + 1}/{max_retries}: Erreur de validation - {str(e)}")
                
            if attempt < max_retries - 1:
                promptStrict = """
            ## Validation stricte requise

            - Si tu n'es pas certain de pouvoir produire un JSON conforme, NE DONNE AUCUNE RÉPONSE.
            - Le tableau JSON final **doit être parfaitement parsable** (aucune erreur de syntaxe, guillemets fermés, virgules bien placées, etc.).
            - Chaque champ doit correspondre **exactement** aux types suivants :
            - `protocole`: string
            - `extrait_pertinent`: array of strings
            - `echantillon`: string
            - `impacts_potentiels`: array of strings
            - `evaluation_invasivite`: string 
            - `peches_identifies`: array of strings représentant les numéros des péchés identifiés

            - Ne fournis **aucune autre sortie** après le tableau JSON.
            - N'ajoute **aucun commentaire, balise Markdown, ou explication** après le JSON final.
            - Le JSON doit être isolé, sans texte avant ou après (sauf l'analyse ReAct).

            """
                prompt = self._build_prompt(question, context, definition) + promptStrict
                answer = self.llm_adapter.generate_answer(prompt)
                print("Nouvelle tentative avec un prompt modifié...")

                
        
        return "Erreur: Impossible de générer une réponse valide après plusieurs tentatives"


    def validate_json(self, answer: str) -> bool:
        """
        Valide si une réponse contient un JSON conforme à la structure attendue.
        """
        try:
            json_blocks = extract_json_blocks(answer)
            if not json_blocks:
                print("[VALIDATE_JSON] Aucun bloc JSON trouvé.")
                return False

            for block in json_blocks:
                try:
                    json_obj = json.loads(block)
                    if isinstance(json_obj, list):
                        for item in json_obj:
                            if not isinstance(item, dict):
                                continue
                            required_keys = {
                                "protocole", "extrait_pertinent", "echantillon",
                                "impacts_potentiels", "evaluation_invasivite", "peches_identifies"
                            }
                            if not required_keys.issubset(item.keys()):
                                continue
                            if not (
                                isinstance(item["extrait_pertinent"], list) and
                                isinstance(item["impacts_potentiels"], list) and
                                isinstance(item["peches_identifies"], list) and
                                isinstance(item["protocole"], str) and
                                isinstance(item["echantillon"], str) and
                                isinstance(item["evaluation_invasivite"], str)
                            ):
                                continue
                            return True
                except json.JSONDecodeError:
                    continue

            print("[VALIDATE_JSON] Aucun bloc JSON valide trouvé.")
            return False

        except Exception as e:
            print(f"[VALIDATE_JSON] Erreur inattendue: {e}")
            return False
    
        
    def analyse_par_chunk(self, question: str, definition: str = "", top_k: int = 4) -> str:
        documents = self.retrieve_with_hyde(question, top_k)
        chunks = [doc.content for doc in documents]
        analyses = []
        for i, chunk in enumerate(chunks):
            prompt = self._build_prompt(question, chunk, definition)
            reponse = self.llm_adapter.generate_answer(prompt)
            print(f"Chunks {i+1} : {chunk}")
            print(f"Réponse {i+1} : {reponse}")
            analyses.append(reponse)
        return analyses
    
    def fusionne_analyses(self, analyses: List[str],max_retries: int = 2) -> str:
        """
        Fusionne les analyses en un seul bloc JSON
        """
        prompt_fusion = """
        Tu es un expert en protocoles d'échantillonnage d'ADN en biologie de la conservation.

        Ta tâche : fusionner et structurer des descriptions de protocoles d'échantillonnage d'ADN à partir d'analyses textuelles, **sans jamais inventer** d'information. Tu dois te baser uniquement sur les analyses fournies ci-dessous.

        ## Étapes à suivre (ReAct implicite) :

        1. **Filtrage**
        - Garde uniquement les protocoles liés à l'échantillonnage d'ADN (exclure ARN, séquençage, analyses biochimiques non liées au prélèvement).
        - Ignore tout ce qui ne décrit pas explicitement une procédure d'échantillonnage.

        2. **Fusion**
        - Si deux protocoles ont :
            - des noms de protocole qui décrivent clairement la même procédure même si formulés différemment
            - OU exactement le même échantillon ET les mêmes impacts
            - ET des extraits qui décrivent clairement la même procédure même si formulés différemment
            - Considère comme équivalents les extraits qui décrivent exactement le même protocole (même outil, même geste, même but), même si certains extraits sont plus détaillés que d'autres.
            - Ne te base pas uniquement sur la formulation littérale : si un extrait est un résumé fidèle d'un autre, fusionner.
        ALORS fusionner. Sinon, conserver séparés.
        - Conserve **toutes les variantes d'extraits** associées.
        - Regroupe les cas mentionnés comme équivalents (ex. « prélèvements de fèces » groupés ou individuels).
        -Important : si l'extrait ne précise pas explicitement si l'animal est vivant ou mort, tu dois considérer que cette information est inconnue. Ne déduis jamais qu'un animal est mort simplement parce que cela n'est pas dit.

        3. **Évaluation**
        - Uniformise les noms d'échantillons très proches. Exemples :
        - "ADN d'estomac" et "contenu stomacal" ALORS regroupe sous : `"Contenu stomacal"`
        - "ADN fécal" et "fèces" et "excréments" ALORS regroupe sous : `"Fèces"`
        - "ADN salivaire", "salive", etc. ALORS regroupe sous : `"Salive"`

        4. **Résolution de conflits**
        - Si un protocole est appliqué **sur des animaux déjà morts de cause naturelle, collision routière ou chasse tierce**, alors :
        - Les **impacts doivent être considérés comme nuls** (l'animal ne subit pas de stress ou de capture).
        - L'évaluation de l'invasivité doit être : `"Non invasif"`.
        - Les péchés doivent être : `[]`.
        - Ne jamais dupliquer un protocole à cause d'une contradiction sur l'impact si l'extrait indique clairement que l'animal est déjà mort.

        5. **Format de sortie**
        - Génère un **seul tableau JSON**, strictement conforme au format suivant.  
        - Ne produis **aucun texte avant ou après** le JSON.

        ```json
        [
            {{
                "protocole": "Nom précis et descriptif du protocole",
                "extrait_pertinent": ["Citation exacte de l'extrait 1", "Citation exacte 2"],
                "echantillon": "Type d'échantillon prélevé",
                "impacts_potentiels": ["Impact 1", "Impact 2"],
                "evaluation_invasivite": "Invasif" ou "Non invasif",
                "peches_identifies": ["1", "2", "5"]
            }}
        ]
        ```
        N'ajoute aucun texte avant ou après le JSON.
        Ne génère aucun protocole, nom ou info qui ne soit pas explicitement mentionné dans les extraits.

        ATTENTION — Règles strictes anti-hallucination :

        - Tu n'as PAS le droit d'inférer une information absente, même si elle semble logique.
        - Tu ne peux PAS supposer que deux extraits sont équivalents si leur contenu diffère légèrement.
        - Tu NE PEUX PAS modifier l'évaluation d'invasivité sauf si un des extraits le justifie explicitement.
        - Si deux protocoles ont des évaluations contradictoires (Invasif / Non invasif), NE PAS fusionner.
        Ne créez jamais de nouveau protocole ou extrait
        Voici les analyses à fusionner :
        {analyses}
        """.format(analyses="\n\n".join(analyses))

        for attempt in range(max_retries):
            answer = self.llm_adapter.generate_answer(prompt_fusion)
            print(f"Réponse fusionnée {attempt+1} : {answer}")
            if self.validate_json(answer):
                return answer
            else:
                print(f" Fusion invalide (tentative {attempt+1}/{max_retries})")
            if attempt < max_retries - 1:
                promptStrict = """
                ## Validation stricte requise

                - Si tu n'es pas certain de pouvoir produire un JSON conforme, NE DONNE AUCUNE RÉPONSE.
                - Le tableau JSON final **doit être parfaitement parsable** (aucune erreur de syntaxe, guillemets fermés, virgules bien placées, etc.).
                - Chaque champ doit correspondre **exactement** aux types suivants :
                - `protocole`: string
                - `extrait_pertinent`: array of strings
                - `echantillon`: string
                - `impacts_potentiels`: array of strings
                - `evaluation_invasivite`: string 
                - `peches_identifies`: array of strings représentant les numéros des péchés identifiés

                """
                prompt_fusion += promptStrict
        return "Erreur: Impossible de générer un JSON valide après plusieurs tentatives"




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



# Revoie la réponse sans les balises <think> et ce qui est entre elles
def filter_response_think(response: str) -> str:
    """
    Renvoie la réponse sans les balises <think>
    """
    return re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL) 


class RefineRAGSystem(RAGSystem):
    """
    Système RAG avec méthode de raffinement en trois niveaux
    """
    
    def __init__(self, api_key=None, api_url=None):
        super().__init__(api_key, api_url)
        
    def build_protocol_detection_prompt(self, chunk: str) -> str:
        """
        Construit le prompt pour le premier niveau de raffinement
        qui détecte si le chunk contient un protocole d'échantillonnage ADN
        """
        return f"""
        Tu es un expert en analyse de protocoles d'échantillonnage d'ADN.

        Ta tâche est de déterminer si l'extrait suivant contient une description de protocole d'échantillonnage d'ADN.

        Un protocole d'échantillonnage d'ADN doit décrire :
        - Le type d'échantillon prélevé (fèces, sang, poils, tissus, carcasses, estomac, etc.)
        - La méthode de collecte (par exemple, utilisation de kits de prélèvement, collecte manuelle, piégeage, etc.)
        - Le contexte de la collecte (par exemple, environnement naturel, laboratoire, piégeage récreatif, etc.)
        - Les outils ou équipements utilisés pour la collecte (par exemple, tubes stériles, gants, sacs en plastique, etc.)
        - Les conditions de stockage des échantillons (par exemple, température, durée, etc.)
        - Les méthodes de traitement des échantillons (par exemple, extraction d'ADN, concentration, etc.)

        Exemples de descriptions de protocoles :
        - "Des échantillons de sang ont été prélevés."
        - "Les échantillons de fèces ont été collectés."
        - "Les poils ont été prélevés."
        - "Blow sampling of three adult, clinically healthy aquarium belugas"
        - "Carcasses were collected as a result of recreational trapping, nuisance removal, and as opportunistic roadkill."
        - "Scat samples were collected."
        - "The 10mL stomach subsample was then thawed completely and further sub-sampled by removing 400μL with a wide-bore pipette tip."
        - "From the 86 otter stomachs sampled, 75 (87%) contained at least one identifiable prey item."

        Si l'extrait contient un ou plusieurs protocoles, renvoie-les au format JSON suivant :
        ```json
        {{
            "contient_protocole": true,
            "extraits_pertinents": ["Citation exacte de l'extrait 1 décrivant le protocole", "Citation exacte de l'extrait 2 décrivant le protocole"]
        }}
        ```

        Si l'extrait ne contient pas de protocole, renvoie :
        ```json
        {{
            "contient_protocole": false,
            "extraits_pertinents": []
        }}
        ```

        Tu dois Uniquement renvoyer le JSON, aucune autre explication, aucun commentaire, aucune phrase.

        Extrait à analyser :
        {chunk}
        """




        
    def build_invasivity_analysis_prompt(self, protocol: str, definition: str) -> str:
        """
        Construit le prompt pour le deuxième niveau de raffinement
        qui analyse l'invasivité du protocole
        """
        return f"""
        Tu es un expert en analyse de protocoles d'échantillonnage d'ADN.

        Ta tâche est d'analyser le protocole suivant pour déterminer s'il est invasif selon la définition donnée.

        Définition de l'échantillonnage non-invasif :
        {definition}

        Critères d'invasivité :
        Un protocole est invasif si au moins UN des critères suivants est rempli:
        - Contact direct avec l'animal vivant
        - Capture ou manipulation de l'animal
        - Modification du comportement naturel
        - Perturbation significative de l'habitat
        - Utilisation d'appâts ou leurres modifiant le comportement

        Un protocole est non-invasif UNIQUEMENT si TOUS ces critères sont remplis:
        - Aucun contact avec l'animal vivant
        - Aucune perturbation du comportement naturel
        - Utilisation d'échantillons déjà abandonnés naturellement
        - Aucune modification de l'habitat ou du territoire

        Processus d'analyse avec ReAct :

        1. **Réflexion (Thought)** :
        - Identifier les éléments clés du protocole.
        - Comparer ces éléments avec les critères d'invasivité.

        2. **Action (Act)** :
        - Évaluer chaque critère d'invasivité en fonction des éléments identifiés.
        - Déterminer si le protocole est invasif ou non.

        3. **Observation (Obs)** :
        - Citer les extraits pertinents du protocole qui justifient l'évaluation.
        - Identifier les impacts potentiels et les péchés d'échantillonnage.

        Protocole à analyser :
        {protocol}

        Renvoie ton analyse au format JSON suivant :
        ```json
        {{
            "extraits_pertinents": ["Citation exacte de l'extrait 1 décrivant le protocole", "Citation exacte de l'extrait 2 décrivant le protocole"],
            "evaluation_invasivite": "Invasif" ou "Non invasif",
            "impacts_potentiels": ["Impact 1", "Impact 2"],
            "peches_identifies": ["1", "2", "3", "4", "5"]
        }}
        ```

        Tu dois Uniquement renvoyer le JSON, aucune autre explication, aucun commentaire, aucune phrase.
        """


    
    def build_fusion_prompt(self, analyses: List[Dict]) -> str:
        """
        Construit le prompt pour le troisième niveau de raffinement
        qui fusionne les analyses en un tableau JSON final
        """
        return f"""
        Tu es un expert en protocoles d'échantillonnage d'ADN.

        Ta tâche est de fusionner les analyses suivantes en un tableau JSON final.

        Processus de fusion avec ReAct :

        1. **Filtrage (Filtering)** :
        - Garde uniquement les protocoles liés à l'échantillonnage d'ADN.
        - Ignore tout ce qui ne décrit pas explicitement une procédure d'échantillonnage.

        2. **Fusion (Fusion)** :
        - Si deux protocoles décrivent la même procédure, les fusionner.
        - Conserver tous les extraits pertinents.
        - Uniformiser les noms d'échantillons similaires.
        - Ne pas inventer d'informations.

        3. **Évaluation (Evaluation)** :
        - Uniformise les noms d'échantillons très proches.
        - Résoudre les conflits en fonction des règles strictes.

        4. **Résolution de conflits (Conflict Resolution)** :
        - Si un protocole est appliqué sur des animaux déjà morts de cause naturelle, les impacts doivent être considérés comme nuls.
        - L'évaluation de l'invasivité doit être "Non invasif".
        - Les péchés doivent être une liste vide.

        5. **Format de sortie (Output Format)** :
        - Génère un seul tableau JSON, strictement conforme au format suivant.
        - Ne produis aucun texte avant ou après le JSON.

        Règles de fusion :
        1. Si deux protocoles décrivent la même procédure, les fusionner.
        2. Conserver tous les extraits pertinents.
        3. Uniformiser les noms d'échantillons similaires.
        4. Ne pas inventer d'informations.
        5. Retire les protocoles inconnus avec aucune citation.

        Analyses à fusionner :
        {analyses}

        Renvoie le résultat au format JSON suivant :
        ```json
        [
            {{
                "protocole": "Nom précis du protocole",
                "extrait_pertinent": ["Citation 1", "Citation 2"],
                "echantillon": "Type d'échantillon",
                "impacts_potentiels": ["Impact 1", "Impact 2"],
                "evaluation_invasivite": "Invasif" ou "Non invasif",
                "peches_identifies": ["1", "2", "3", "4", "5"]
            }}
        ]
        ```

        Tu dois Uniquement renvoyer le JSON, aucune autre explication, aucun commentaire, aucune phrase.
        """

    
    def refine_analysis(self, question: str, definition: str = "", top_k: int = 4) -> str:
        """
        Applique la méthode de raffinement en trois niveaux
        """
        # Récupération des chunks pertinents
        documents = self.retrieve_with_hyde(question, top_k)
        chunks = [doc.content for doc in documents]
        print(f"Documents récupérés : {len(chunks)}")

        #  Détection des protocoles
        protocols = []
        for chunk in chunks:
            prompt = self.build_protocol_detection_prompt(chunk)
            response = self.llm_adapter.generate_answer(prompt)
            print(f"Chunk : {chunk}")
            print(f"Réponse du modèle : {response}")
            protocols.append(response)

        if not protocols:
            print("Aucun protocole détecté.")
            return "[]"

        print(f"Protocoles détectés : {len(protocols)}")

        #  Analyse de l'invasivité
        analyses = []
        for protocol in protocols:
            prompt = self.build_invasivity_analysis_prompt(protocol, definition)
            response = self.llm_adapter.generate_answer(prompt)
            print(f"Réponse du modèle : {response}")
            analyses.append(response)

        if not analyses:
            print("Aucune analyse générée.")
            return "[]"

        print(f"Analyses générées : {len(analyses)}")

        #  Fusion des analyses
        prompt = self.build_fusion_prompt(analyses)
        final_response = self.llm_adapter.generate_answer(prompt)
        print(f"Réponse finale : {final_response}")

        return final_response
