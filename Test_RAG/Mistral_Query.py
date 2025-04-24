import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

# --- Charger les chunks (texte + métadonnées) ---
with open("chunk_metadata.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# --- Charger l’index FAISS ---
index = faiss.read_index("article_index.faiss")

# --- Charger SentenceTransformer ---
encoder = SentenceTransformer("all-MiniLM-L6-v2")

# --- Initialiser le LLM local avec plus de contexte ---
llm = Llama(model_path="models/mistral-7b-instruct-v0.1.Q2_K.gguf", n_ctx=4096)

# --- Reformulation de la requête utilisateur ---
def rewrite_query(question):
    prompt = f"""Tu es un assistant d'indexation scientifique. Ta tâche est de transformer une question en une phrase **descriptive et littérale**, en t'appuyant uniquement sur les mots-clés présents dans la question.

Ta reformulation doit respecter ces règles :
- Utilise uniquement les informations présentes dans la question (aucune interprétation, aucune inférence).
- Ne change pas de sujet, ne généralise pas, ne complète pas la question.
- Si certains mots sont ambigus, conserve-les tels quels.

Le but est de produire une phrase simple, factuelle.

Question : {question}
Phrase reformulée :"""

    result = llm(prompt, max_tokens=120, temperature=0.4)
    return result["choices"][0]["text"].strip()


# --- Recherche contextuelle dans l’index FAISS ---
def search_query(query, k=5):
    query_embedding = encoder.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    return [chunks[i] for i in indices[0]]

# --- Génération avec sécurité contre réponses incomplètes ---
def safe_generate(prompt, llm, max_tokens=1024):
    full_response = ""
    for _ in range(3):  # autorise jusqu'à 3 reprises
        result = llm(prompt + full_response, max_tokens=max_tokens, temperature=0.7)
        new_text = result["choices"][0]["text"]
        full_response += new_text.strip()
        if full_response.endswith((".", "!", "?", "…")):
            break
    return full_response.strip()

# --- Génération finale avec le contexte ---
def generate_answer_local(question, retrieved_chunks):
    context = "\n\n".join([
        f"[{c['section']}]\n{c['text'][:500]}..."  # tronquer pour éviter surcharge
        for c in retrieved_chunks
    ])

    prompt = f"""Tu es un assistant scientifique rigoureux.

Réponds à la question suivante en t'appuyant **uniquement** sur les extraits de texte ci-dessous. Ta réponse doit être précise, factuelle et structurée. Si la réponse ne peut pas être déduite des extraits, indique-le explicitement.

Définition de référence à utiliser :
"Selon Taberlet et al. (1999), l'échantillonnage non-invasif est une méthode de collecte de matériel génétique d'un organisme sans recours à des techniques invasives comme l’anesthésie, la perforation de la peau, la destruction de tissus, ou tout acte susceptible d'altérer le comportement ou la survie de l’animal."

Extraits :
{context}

Question : {question}

Réponse :"""

    return safe_generate(prompt, llm, max_tokens=1024)

# --- Pipeline RAG complet ---
if __name__ == "__main__":
    question = input("La question : ").strip()

    # Étape 1 : reformulation
    print("\n🔧 Reformulation de la question....")
    rewritten = rewrite_query(question)
    print("🔍 Phrase utilisée pour l'index :", rewritten)

    # Étape 2 : recherche contextuelle
    retrieved = search_query(rewritten, k=6)

    # Étape 3 : génération de réponse
    answer = generate_answer_local(question, retrieved)

    print("\n Question :", question)
    print("\nRéponse générée :\n", answer)
