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
llm = Llama(model_path="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf", n_ctx=4096)

# --- Reformulation de la requête utilisateur ---
def rewrite_query(question):
    prompt = f"""Tu es un assistant spécialisé en reformulation de questions scientifiques. Reformule la question suivante en une phrase simple, explicite et précise, afin de faciliter la recherche d'extraits scientifiques pertinents.

Question originale : {question}

Phrase reformulée :"""
    result = llm(prompt, max_tokens=150, temperature=0.5)
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

    prompt = f"""Tu es un assistant scientifique. Réponds clairement à la question suivante en utilisant uniquement les extraits ci-dessous.

Extraits :
{context}

Question : {question}

Réponse :"""
    return safe_generate(prompt, llm, max_tokens=1024)

# --- Pipeline RAG complet ---
if __name__ == "__main__":
    question = input("La question : ").strip()

    # Étape 1 : reformulation
    print("\n🔧 Reformulation de la question...")
    rewritten = rewrite_query(question)
    print("🔍 Phrase utilisée pour l'index :", rewritten)

    # Étape 2 : recherche contextuelle
    retrieved = search_query(rewritten, k=6)

    # Étape 3 : génération de réponse
    answer = generate_answer_local(question, retrieved)

    print("\n Question :", question)
    print("\nRéponse générée :\n", answer)
