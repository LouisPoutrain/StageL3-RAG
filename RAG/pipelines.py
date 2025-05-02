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
            template="""Imagine que tu es un article scientifique qui contient l'information permettant de répondre à cette question. 
            Génère un court extrait de ce document qui contiendrait l'information pertinente, sans directement répondre à la question.
            
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
            max_tokens=400,
            temperature=0.75
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


def create_rag_pipeline(text_embedder, retriever, reranker) -> Pipeline:
    """
    Crée une pipeline RAG standard
    
    Args:
        text_embedder: Composant pour les embeddings de texte
        retriever: Composant pour la récupération de documents
        reranker: Composant pour le re-classement des documents
        
    Returns:
        Pipeline Haystack pour le RAG standard
    """
    rag_pipeline = Pipeline()
    rag_pipeline.add_component("text_embedder", text_embedder)
    rag_pipeline.add_component("retriever", retriever)
    rag_pipeline.add_component("reranker", reranker)
    
    # Connexions entre les composants
    rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    rag_pipeline.connect("retriever.documents", "reranker.documents")
    
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
    from haystack.components.writers import DocumentWriter
    from haystack.components.embedders import SentenceTransformersDocumentEmbedder
    
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component("doc_embedder", 
                                   SentenceTransformersDocumentEmbedder(model=embedder_model))
    indexing_pipeline.add_component("writer", DocumentWriter(document_store=document_store))
    indexing_pipeline.connect("doc_embedder.documents", "writer.documents")
    
    return indexing_pipeline
