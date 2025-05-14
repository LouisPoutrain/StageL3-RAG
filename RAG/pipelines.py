"""
Pipelines pour le système RAG
"""

from typing import List, Dict
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.dataclasses import Document
from components import HypotheticalDocumentEmbedder
from UniversityLLMAdapter import UniversityLLMAdapter
from haystack.components.writers import DocumentWriter
from haystack.components.validators import JsonSchemaValidator

# Schéma JSON pour la validation des réponses
PROTOCOL_SCHEMA = {
    "type": "object",
    "required": ["protocole", "extrait_pertinent", "echantillon", "impacts_potentiels", "evaluation_invasivite", "peches_identifies"],
    "properties": {
        "protocole": {"type": "string"},
        "extrait_pertinent": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1
        },
        "echantillon": {"type": "string"},
        "impacts_potentiels": {
            "type": "array",
            "items": {"type": "string"}
        },
        "evaluation_invasivite": {
            "type": "string",
            "enum": ["Non invasif", "Invasif", "présumé non invasif"]
        },
        "peches_identifies": {
            "type": "array",
            "items": {"type": "string"}
        }
    }
}

def create_hyde_pipeline(api_key: str, api_url: str, embedder_model: str) -> Pipeline:
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
    
    # Création du prompt pour générer des documents hypothétiques
    hyde_pipeline.add_component(
        "prompt_builder", 
        PromptBuilder(
            template="""Imagine que tu es un article scientifique qui décrit des protocoles d'échantillonnage pour l'analyse génétique d'animaux. 

Ta tâche est de générer un extrait détaillé (75-100 mots) décrivant les **méthodes expérimentales** spécifiques utilisées pour l'échantillonnage d'ADN, avec une attention particulière aux aspects suivants :

1. **Type de prélèvement**: Décris précisément la nature du prélèvement (sang, tissu, poils, fèces, urine, salive)
2. **Méthode de collecte**: Explique si l'animal est manipulé, capturé ou si le prélèvement est fait sans contact
3. **Équipement utilisé**: Mentionne les pièges, appâts, dispositifs de stockage ou instruments
4. **Protocole d'échantillonnage**: Décris le processus étape par étape, incluant fréquence et quantité
5. **Considérations éthiques**: Indique si des mesures sont prises pour minimiser l'impact sur l'animal

Ne réponds pas directement à la question, mais fournis un **extrait descriptif et précis** qui pourrait provenir d'un article scientifique réel sur le sujet.

Question: {{question}}

Extrait de document scientifique:""",
            required_variables=["question"]  
        )
    )
    
    # Générateur de texte (LLM)
    hyde_pipeline.add_component(
        "generator",
        UniversityLLMAdapter(
            api_key=api_key,
            api_url=api_url,
            max_tokens=500,
            temperature=0.7
        )
    )

    
    
    # Embedder pour les documents générés
    hyde_pipeline.add_component(
        "embedder", 
        SentenceTransformersDocumentEmbedder(model=embedder_model)
    )
    
    # Calcul de l'embedding moyen
    hyde_pipeline.add_component("hyde", HypotheticalDocumentEmbedder())
    
    
    # Connexions entre les composants
    hyde_pipeline.connect("prompt_builder.prompt", "generator.prompt")
    hyde_pipeline.connect("generator.documents", "embedder.documents")
    hyde_pipeline.connect("embedder.documents", "hyde.documents")
    
    return hyde_pipeline


def create_rag_pipeline(text_embedder, retriever, splitter) -> Pipeline:
    """
    Crée une pipeline RAG standard
    
    Args:
        text_embedder: Composant pour les embeddings de texte
        retriever: Composant pour la récupération de documents
        splitter: Composant pour le découpage des documents
        
    Returns:
        Pipeline Haystack pour le RAG standard
    """
    rag_pipeline = Pipeline()
    rag_pipeline.add_component("text_embedder", text_embedder)
    rag_pipeline.add_component("retriever", retriever)
    rag_pipeline.add_component("splitter", splitter)
    
    # Connexions entre les composants
    rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    rag_pipeline.connect("retriever.documents", "splitter.documents")
    
    return rag_pipeline


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
    indexing_pipeline.add_component("writer", DocumentWriter(document_store=document_store, policy="overwrite"))
    indexing_pipeline.connect("doc_embedder.documents", "writer.documents")
    
    return indexing_pipeline
