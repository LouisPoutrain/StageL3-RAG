"""
Ce fichier définit les fonctions pour construire les différentes pipelines Haystack
utilisées par le système RAG.
"""

from typing import List, Dict
from haystack import Pipeline, component
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.dataclasses import Document
from components import HypotheticalDocumentEmbedder
from UniversityLLMAdapter import UniversityLLMAdapter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy


@component
class PrintDocumentsComponent:
    """
    Composant qui affiche le contenu des documents qui lui sont passés.
    """
    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        print("\n=== Documents hypothétiques générés ===")
        for i, doc in enumerate(documents, 1):
            print(f"\nDocument hypothétique {i}:")
            print("-" * 50)
            print(f"{doc.content}")
            print("-" * 50)
        print("\n=== Fin des documents hypothétiques ===\n")
        return {"documents": documents}

def create_hyde_pipeline(api_key: str, api_url: str, embedder_model: str, invasive_detection: bool = False) -> Pipeline:
    """
    Crée et configure une pipeline HyDE (Hypothetical Document Embeddings)
    
    Args:
        api_key: Clé API pour le modèle LLM
        api_url: URL de l'API pour le modèle LLM
        embedder_model: Modèle à utiliser pour les embeddings
        
    Returns:
        Pipeline Haystack configurée pour HyDE
    """
    hyde_pipeline = Pipeline()

    # Le template du prompt varie si la pipeline est utilisée pour la détection d'invasivité
    # ou pour l'analyse générale des protocoles.
    if invasive_detection:
         hyde_pipeline.add_component(
            "prompt_builder", 
            PromptBuilder(
                template= """Imagine que tu es l'auteur d'un article scientifique décrivant des protocoles d'échantillonnage utilisés pour l'analyse génétique d'animaux sauvages.

            Ta tâche est de rédiger un extrait réaliste de cet article, dans un style scientifique, qui **décrit précisément un protocole donné** et **justifie clairement son niveau d'invasivité (non invasif, minimalement invasif ou invasif)**.

            Tu dois :

            - Ne pas répondre directement à la question posée, mais construire un **paragraphe descriptif** crédible (75 à 100 mots),
            - Justifier pourquoi le protocole est considéré comme (non/minimalement) invasif, en mentionnant éventuellement les conditions d'échantillonnage, la capture ou non des animaux, l'usage d'anesthésie, etc.
            - Utiliser un vocabulaire scientifique clair et nuancé.

            Voici le protocole à décrire sous forme de question :
            Question : {{question}}

            Voici un axe de recherche (titre de l'article) qui peut guider ton style ou ton contexte :
            Titre : {{title}}

            Commence directement par l'extrait (pas d'introduction), dans un style scientifique, à la troisième personne :
            Extrait de document scientifique :""",
                            required_variables=["question", "title"]  
                        )
                    )
    
    else :
        # Création du prompt pour générer des documents hypothétiques
        hyde_pipeline.add_component(
            "prompt_builder", 
            PromptBuilder(
                template="""Imagine que tu es l'auteur d'un article scientifique portant sur les méthodes d'échantillonnage d'ADN utilisées pour l'analyse génétique d'animaux sauvages.

            Ta tâche est de générer un extrait crédible (75 à 100 mots) issu de la section "Méthodes" d'un tel article. Cet extrait doit décrire **un ou plusieurs protocoles d'échantillonnage** en respectant les points suivants :

            1. Type de prélèvement : Décris précisément le type d'échantillon collecté (sang, tissu, poils, fèces, urine, salive…)
            2. Méthode de collecte : Explique si l'animal est capturé, manipulé ou non contacté (collecte passive)
            3. Équipement utilisé : Mentionne les outils, pièges, appâts, dispositifs de stockage
            4. Protocole d'échantillonnage : Décris les étapes du protocole (fréquence, quantité, durée)
            5. Considérations éthiques et impact : Indique si des précautions sont prises pour minimiser le stress, les blessures ou la perturbation des animaux

            L'objectif est de simuler un extrait scientifique détaillé qui permette d'évaluer le niveau d'invasivité selon la définition de Taberlet, même si le mot "invasif" n'est pas utilisé directement.

            Ne réponds pas à la question ci-dessous : utilise-la uniquement comme contexte pour t'inspirer.

            Question : {{question}}

            Voici le titre de l'article à prendre comme guide stylistique ou thématique :
            Titre : {{title}}

            Commence directement par l'extrait, dans un style scientifique réaliste, à la troisième personne :

            Extrait de document scientifique :
            """,
                        required_variables=["question", "title"]  
                    )
                )
    
    # Le générateur utilise l'adaptateur personnalisé pour appeler le LLM.
    hyde_pipeline.add_component(
        "generator",
        UniversityLLMAdapter(
            api_key=api_key,
            api_url=api_url,
            max_tokens=500,
            temperature=0.7
        )
    )

    # Afficher les documents générés
    hyde_pipeline.add_component(
        "print_generated_documents",
        PrintDocumentsComponent()
    )

    # L'embedder transforme les documents textuels générés en vecteurs numériques.
    hyde_pipeline.add_component(
        "embedder", 
        SentenceTransformersDocumentEmbedder(model=embedder_model)
    )
    
    # Ce composant calcule l'embedding moyen de tous les documents hypothétiques générés.
    hyde_pipeline.add_component("hyde", HypotheticalDocumentEmbedder())
    
    
    # Connexion des composants pour former le flux de données de la pipeline.
    hyde_pipeline.connect("prompt_builder.prompt", "generator.prompt")
    hyde_pipeline.connect("generator", "print_generated_documents")
    hyde_pipeline.connect("print_generated_documents", "embedder.documents")
    hyde_pipeline.connect("embedder.documents", "hyde.documents")

    return hyde_pipeline



def create_indexing_pipeline(document_store, embedder_model: str) -> Pipeline:
    """
    Crée une pipeline pour l'indexation de documents
    
    Args:
        document_store: Store de documents pour l'indexation
        embedder_model: Modèle à utiliser pour les embeddings
        
    Returns:
        Pipeline Haystack pour l'indexation
    """
    
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component("doc_embedder", 
                                   SentenceTransformersDocumentEmbedder(model=embedder_model))
    indexing_pipeline.add_component("writer", DocumentWriter(document_store=document_store, policy=DuplicatePolicy.OVERWRITE))
    indexing_pipeline.connect("doc_embedder.documents", "writer.documents")
    
    return indexing_pipeline
