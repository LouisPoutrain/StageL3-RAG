"""
Composants personnalisés pour le système RAG
"""

from typing import List, Dict, Any, Optional
import numpy as np
from numpy import array, mean
from haystack import component
from haystack.dataclasses import Document

from llama_cpp import Llama
from typing import Dict, List, Tuple


@component
class HypotheticalDocumentEmbedder:
    """
    Composant qui calcule l'embedding moyen à partir de documents hypothétiques générés.
    Utilisé dans la méthode HyDE (Hypothetical Document Embeddings).
    """
    @component.output_types(hypothetical_embedding=List[float])
    def run(self, documents: List[Document]):
        # Collecte des embeddings des documents hypothétiques
        stacked_embeddings = array([doc.embedding for doc in documents])
        # Calcul l'embedding moyen
        avg_embeddings = mean(stacked_embeddings, axis=0)
        # Puis on met forme le vecteur 
        hyde_vector = avg_embeddings.reshape((1, len(avg_embeddings)))
        return {"hypothetical_embedding": hyde_vector[0].tolist()}

@component
class DynamicThresholdAdapter:
    """
    Composant qui ajuste dynamiquement le seuil de similarité cosinus
    en fonction de la distribution des scores, sauf si peu de documents.
    """
    def __init__(self, min_threshold: float = 0.3, percentile: float = 75, min_docs: int = 5):
        self.min_threshold = min_threshold
        self.percentile = percentile
        self.min_docs = min_docs  

    @component.output_types(filtered_documents=List[Document])
    def run(self, documents: List[Document], scores: List[float]):
        if not documents or not scores:
            return {"filtered_documents": []}

        if len(documents) < self.min_docs:
            # Si peu de documents, on garde tout
            return {"filtered_documents": documents}

        # Calcul du seuil dynamique
        dynamic_threshold = max(
            self.min_threshold,
            np.percentile(scores, self.percentile)
        )

        # Filtrage des documents
        filtered_docs = [
            doc for doc, score in zip(documents, scores)
            if score >= dynamic_threshold
        ]

        return {"filtered_documents": filtered_docs}
    



