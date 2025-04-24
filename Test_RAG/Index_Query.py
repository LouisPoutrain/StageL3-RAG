import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

# --- 1. Charger l’index FAISS et les métadonnées ---
index = faiss.read_index("article_index.faiss")
with open("chunk_metadata.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# --- 2. Charger le même modèle utilisé pour l’indexation ---
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- 3. Fonction pour interroger l’index ---
def search_query(query, k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)

    results = []
    for i, dist in zip(indices[0], distances[0]):
        results.append({
            "section": chunks[i]["section"],
            "text": chunks[i]["text"],
            "distance": float(dist)
        })
    return results

# --- 4. Exemple d’interrogation ---
if __name__ == "__main__":
    user_query = input("La question: ").strip()
    top_k = 5
    results = search_query(user_query, k=top_k)

    print(f"\n🔎 Résultats pour la requête : '{user_query}'\n")
    for i, res in enumerate(results, 1):
        print(f"{i}. [Section : {res['section']}] (Distance : {res['distance']:.4f})\n{res['text']}\n")
