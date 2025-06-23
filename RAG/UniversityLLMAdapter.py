"""
Adaptateur pour interagir avec une API LLM 
"""

from typing import List, Dict, Any, Optional
from haystack import component
from haystack.dataclasses import Document
import requests
import pprint
from ecologits import EcoLogits
import time

# === Initialisation globale ===
EcoLogits.init(electricity_mix_zone="FRA")


# === Fonction utilitaire pour calculer l'impact ===
def impact_from_manual_values(model_name: str, duration_s: float, energy_kwh: float) -> dict:
    gco2_per_kwh = 53  # facteur moyen pour la France
    return {
        "model": model_name,
        "duration_s": duration_s,
        "energy_kwh": energy_kwh,
        "co2_g": energy_kwh * gco2_per_kwh
    }

@component
class UniversityLLMAdapter:
    """
    Cet adaptateur est un composant Haystack personnalisé qui sert de pont entre
    la pipeline Haystack et une API de Grand Modèle de Langage (LLM) compatible
    avec le format d'OpenAI.

    Il est responsable de :
    - Formater la requête (prompt) selon le schéma attendu par l'API.
    - Envoyer la requête HTTP avec les en-têtes d'authentification.
    - Recevoir la réponse du LLM.
    - Extraire le contenu textuel généré et le transformer en objets `Document` Haystack,
      qui peuvent ensuite être utilisés par d'autres composants de la pipeline.
    """
    def __init__(
        self,
        api_key: str,
        api_url: str,
        model: str = "mistral-small3.1:latest",   #mistral-small3.1:latest
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system_prompt: str = "Tu es un assistant scientifique rigoureux."
    ):
        """
        Initialise l'adaptateur pour l'API LLM
        
        Args:
            api_key: Clé API pour l'authentification
            api_url: URL du point d'API
            model: Identifiant du modèle à utiliser
            max_tokens: Nombre maximum de tokens à générer
            temperature: Température pour la génération 
            system_prompt: Message système pour guider le comportement du modèle
        """
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    @component.output_types(documents=List[Document])
    def run(self, prompt: str, n_generations: int = 1):
        """
        Génère des réponses avec l'API LLM compatible OpenAI
        
        Args:
            prompt: Le texte du prompt à envoyer au modèle
            n_generations: Nombre de variantes à générer
            
        Returns:
            Liste des documents générés
        """
        documents = []
        try:
            # Construction du payload pour l'API, en respectant le format OpenAI.
            payload_chat = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "n": n_generations,
                "logprobs": True, # Ne marche pas
                "top_logprobs": 2
            }
            
            # Envoi de la requête POST à l'API du LLM avec les en-têtes et le payload
            response = requests.post(self.api_url, headers=self.headers, json=payload_chat)
            # Lève une exception si la requête a échoué
            response.raise_for_status()
            result = response.json()
            
            # La réponse de l'API contient une liste "choices", on itère dessus pour extraire chaque génération
            if "choices" in result:
                for i, choice in enumerate(result["choices"]):
                    content = ""
                    logprobs = None

                    if "message" in choice and "content" in choice["message"]:
                        content = choice["message"]["content"]
                        logprobs = choice["message"].get("logprobs")
                    elif "text" in choice:
                        content = choice["text"]
                        logprobs = choice.get("logprobs")

                    print(f"resultats sortie llm: {result}")
                    # Le contenu généré est encapsulé dans un objet Document Haystack
                    documents.append(Document(content=content))
            return {"documents": documents}
            
        except Exception as e:
            # En cas d'erreur
            error_msg = f"Erreur lors de la génération: {str(e)}"
            if 'response' in locals():
                if hasattr(response, 'status_code'):
                    error_msg += f"\nStatus: {response.status_code}"
                if hasattr(response, 'text'):
                    error_msg += f"\nRéponse: {response.text[:300]}"
            # Pour garantir que la pipeline ne casse pas, on retourne l'erreur dans un objet Document
            return {"documents": [Document(content=error_msg)]}
        
    def generate_answer(self, prompt: str) -> str:
        """
        Version simple : retourne juste la réponse textuelle générée.
        """
        result = self.run(prompt)
        if "documents" in result and result["documents"]:
            return result["documents"][0].content
        return "Erreur: Aucune réponse générée"
    
    def generate_answer_with_impact(self, prompt: str) -> tuple[str, dict]:
        """
        Version avec mesure de l'impact environnemental.
        """
        start_time = time.time()
        result = self.run(prompt)
        end_time = time.time()

        duration_s = end_time - start_time
        energy_kwh = duration_s * 0.0005  # estimation approx.

        impact = impact_from_manual_values(
            model_name=self.model,
            duration_s=duration_s,
            energy_kwh=energy_kwh
        )

        if "documents" in result and result["documents"]:
            return result["documents"][0].content.strip(), impact

        return "Erreur: Aucune réponse générée", impact
