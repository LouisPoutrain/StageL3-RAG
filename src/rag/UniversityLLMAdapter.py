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
        api_url: str = "http://localhost:11434/api/chat",
        model: str = "mistral-nemo:latest",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system_prompt: str = "Tu es un assistant scientifique rigoureux.",
        timeout: int = 180,
        max_retries: int = 2
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
        self.timeout = timeout
        self.max_retries = max_retries

        self.headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
    
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
        documents: List[Document] = []
        errors: List[str] = []
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]

            payload_options: Dict[str, Any] = {
                "temperature": self.temperature
            }
            if self.max_tokens:
                payload_options["num_predict"] = self.max_tokens

            for gen_idx in range(max(1, n_generations)):
                payload_chat = {
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": payload_options
                }

                attempt = 0
                while attempt < self.max_retries:
                    try:
                        response = requests.post(
                            self.api_url,
                            headers=self.headers,
                            json=payload_chat,
                            timeout=self.timeout
                        )
                        response.raise_for_status()
                        result = response.json()

                        content = ""
                        if isinstance(result, dict):
                            if "message" in result and isinstance(result["message"], dict):
                                content = result["message"].get("content", "")
                            elif "messages" in result and isinstance(result["messages"], list):
                                assistant_msgs = [m.get("content", "") for m in result["messages"] if m.get("role") == "assistant"]
                                if assistant_msgs:
                                    content = assistant_msgs[-1]
                            elif "response" in result:
                                content = result["response"]
                            elif "choices" in result:
                                choices = result["choices"]
                                if isinstance(choices, list) and choices:
                                    first_choice = choices[0]
                                    if isinstance(first_choice, dict):
                                        if "message" in first_choice:
                                            content = first_choice["message"].get("content", "")
                                        elif "text" in first_choice:
                                            content = first_choice.get("text", "")

                        print(f"resultats sortie llm: {result}")
                        cleaned_content = content.strip()
                        if cleaned_content:
                            documents.append(Document(content=cleaned_content))
                        break
                    except requests.exceptions.Timeout as e:
                        attempt += 1
                        errors.append(f"Timeout lors de la génération {gen_idx + 1}: {str(e)}")
                        if attempt >= self.max_retries:
                            print(f"Timeout persistant pour la génération {gen_idx + 1}")
                    except requests.exceptions.RequestException as e:
                        attempt += 1
                        errors.append(f"Erreur HTTP lors de la génération {gen_idx + 1}: {str(e)}")
                        if attempt >= self.max_retries:
                            print(f"Erreur HTTP persistante pour la génération {gen_idx + 1}")
                    except ValueError as e:
                        attempt += 1
                        errors.append(f"Erreur de parsing JSON génération {gen_idx + 1}: {str(e)}")
                        if attempt >= self.max_retries:
                            print(f"Erreur de parsing persistante pour la génération {gen_idx + 1}")

            if documents:
                return {"documents": documents}
            
            error_details = " | ".join(errors) if errors else "Aucune génération réussie"
            fallback_message = f"Erreur lors de la génération: {error_details}"
            return {"documents": [Document(content=fallback_message)]}
            
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
