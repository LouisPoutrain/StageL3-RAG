from typing import List, Dict, Any, Optional
from haystack import component
from haystack.dataclasses import Document
import requests

@component
class UniversityLLMAdapter:
    """
    Adaptateur pour interagir avec un API LLM compatible OpenAI
    """
    
    def __init__(
        self,
        api_key: str,
        api_url: str,
        model: str = "phi3:14b",
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ):
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    @component.output_types(documents=List[Document])
    def run(self, prompt: str, n_generations: int = 1):
        """
        Génère des réponses avec l'API LLM compatible OpenAI
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        documents = []
        
        try:
            payload_chat = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "Tu es un assistant scientifique rigoureux."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "n": n_generations
            }
            
            response = requests.post(self.api_url, headers=headers, json=payload_chat)

            
            response.raise_for_status()
            result = response.json()
            
            # Extraction des réponses selon différents formats possibles
            if "choices" in result:
                for choice in result["choices"]:
                    if "message" in choice and "content" in choice["message"]:
                        content = choice["message"]["content"]
                        documents.append(Document(content=content))
                    elif "text" in choice:
                        content = choice["text"]
                        documents.append(Document(content=content))
            
     
                
            return {"documents": documents}
            
        except Exception as e:
            error_msg = f"Erreur lors de la génération: {str(e)}"
            if 'response' in locals():
                if hasattr(response, 'status_code'):
                    error_msg += f"\nStatus: {response.status_code}"
                if hasattr(response, 'text'):
                    error_msg += f"\nRéponse: {response.text[:300]}"
            
            return {"documents": [Document(content=error_msg)]}
