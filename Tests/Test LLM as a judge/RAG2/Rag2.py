import json
import faiss
import numpy as np
import requests
import os
import csv
from sentence_transformers import SentenceTransformer

# --- Configuration API ---
API_URL = "http://gpu1.pedagogie.sandbox.univ-tours.fr:32800/api/chat/completions"
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6ImRjZjZkYWE2LTRjYzMtNDYyOS05MjJiLTkyYzM1NGQzODYwMCJ9.04oTM2nVW8iQTCr8qrs4MknI13dGgqBp85Wq7t2jAeQ"

DOCS_FOLDER = "output2"
QUESTION = "Donne moi le nom de l'animal et sa famille, et le pays où l'ADN a été prélevé"
CSV_FILE = "Résultats/résultats.csv"

encoder = SentenceTransformer("allenai-specter")

def call_llm(prompt, temperature=0.7, max_tokens=4096, model="mistral-small3.1:latest"):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Erreur API : {e}"

def rewrite_query(question):
    prompt = f"""Tu es un assistant d'indexation scientifique. Ta tâche est de transformer une question en une phrase **descriptive et littérale**, en t'appuyant uniquement sur les mots-clés présents dans la question.

Ta reformulation doit respecter ces règles :
- Utilise uniquement les informations présentes dans la question (aucune interprétation, aucune inférence).
- Ne change pas de sujet, ne généralise pas, ne complète pas la question.
- Si certains mots sont ambigus, conserve-les tels quels.

Le but est de produire une phrase simple, factuelle.

Question : {question}
Phrase reformulée :"""
    return call_llm(prompt, temperature=0.4, max_tokens=120)

def search_query(query, chunks, index, k=5):
    query_embedding = encoder.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    return [chunks[i] for i in indices[0]]

def generate_structured_answer(question, retrieved_chunks):
    context = "\n\n".join([
        f"[{c['section']}]\n{c['text'][:200]}..." for c in retrieved_chunks
    ])
    prompt = f"""
Tu es un assistant scientifique. Ton objectif est d'extraire **précisément** les informations suivantes à partir des extraits donnés :

- animal : nom de l'animal sur lequel l'ADN a été prélevé
- famille : famille biologique de cet animal
- pays : où l'échantillon a été prélevé
- source : extrait du texte qui mentionne le pays ou l'animal (en quelques phrases)

⚠️ Ne devine rien. Si une info est absente, mets "inconnu". Donne une seule réponse par document.

Format de réponse souhaité :
animal:<nom>
famille:<famille>
pays:<pays>
source:<extrait court>

Extraits :
{context}

Question : {question}

Réponse :
"""
    return call_llm(prompt)

# --- Pipeline principal ---
if __name__ == "__main__":
    rewritten = rewrite_query(QUESTION)
    print("🔍 Phrase utilisée :", rewritten)

    with open(CSV_FILE, "w", encoding="utf-8", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["document", "animal", "famille", "pays", "source_extrait"])

        for file in os.listdir(DOCS_FOLDER):
            if file.endswith(".json"):
                base_name = file[:-5]
                json_path = os.path.join(DOCS_FOLDER, base_name + ".json")
                faiss_path = os.path.join(DOCS_FOLDER, base_name + ".faiss")

                if not os.path.exists(faiss_path):
                    print(f"⛔ Index FAISS manquant pour {base_name}, ignoré.")
                    continue

                print(f"\n📄 Traitement du document : {base_name}")

                with open(json_path, "r", encoding="utf-8") as f:
                    chunks = json.load(f)
                index = faiss.read_index(faiss_path)

                retrieved = search_query(rewritten, chunks, index, k=5)
                raw_answer = generate_structured_answer(QUESTION, retrieved)

                # Parsing du format structuré
                lines = raw_answer.strip().splitlines()
                data = {"animal": "inconnu", "famille": "inconnu", "pays": "inconnu", "source": ""}
                for line in lines:
                    if line.startswith("animal:"):
                        data["animal"] = line.replace("animal:", "").strip()
                    elif line.startswith("famille:"):
                        data["famille"] = line.replace("famille:", "").strip()
                    elif line.startswith("pays:"):
                        data["pays"] = line.replace("pays:", "").strip()
                    elif line.startswith("source:"):
                        data["source"] = line.replace("source:", "").strip()

                writer.writerow([base_name, data["animal"], data["famille"], data["pays"], data["source"]])
                print("✅ Réponse structurée enregistrée.")
