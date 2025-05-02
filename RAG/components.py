"""
Composants personnalisés pour le système RAG
"""

from typing import List, Dict, Any, Optional
import numpy as np
from numpy import array, mean
from haystack import component
from haystack.dataclasses import Document

from llama_cpp import Llama


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
        # Calcul de l'embedding moyen
        avg_embeddings = mean(stacked_embeddings, axis=0)
        # Mise en forme du vecteur HyDE
        hyde_vector = avg_embeddings.reshape((1, len(avg_embeddings)))
        return {"hypothetical_embedding": hyde_vector[0].tolist()}


@component
class LlamaOutputAdapter:
    """
    Composant qui utilise un modèle Llama local pour générer des réponses 
    et les convertir en documents Haystack.
    """
    def __init__(self, model_path, max_tokens=4096, temperature=0.1):
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    @component.output_types(documents=List[Document])
    def run(self, prompt: str, n_generations: int = 3):
        """
        Génère n_generations réponses avec Llama et les convertit en documents
        
        Args:
            prompt: Le prompt à envoyer au modèle
            n_generations: Nombre de documents à générer
            
        Returns:
            Liste des documents générés
        """
        try:
            llm = Llama(model_path=self.model_path, n_ctx=6096)
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
