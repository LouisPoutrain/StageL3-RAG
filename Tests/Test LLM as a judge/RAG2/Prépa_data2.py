import os
import re
import json
import numpy as np

from sentence_transformers import SentenceTransformer

# --- 1. Parser le fichier texte et extraire les chunks ---
def extract_chunks_from_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Chaque section commence par "[SECTION] Nom de la section\n"
    pattern = re.compile(r"\[SECTION\](.*?)\n(.*?)(?=\n\[SECTION\]|$)", re.DOTALL)
    matches = pattern.findall(content)

    chunks = []
    for title, body in matches:
        title = title.strip()
        body = body.strip()
        if body:
            chunks.append({
                "section": title,
                "text": body
            })

    return chunks


# --- 2. Générer les embeddings avec SentenceTransformer ---
def create_embeddings(chunks, model):
    texts = [chunk['text'] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings, texts



# --- 4. Sauvegarder les métadonnées des chunks ---
def save_chunk_metadata(chunks, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

# --- 5. Traitement d'un fichier texte ---
def process_txt_file(file_path, output_dir, model):
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    chunks = extract_chunks_from_txt(file_path)
    if not chunks:
        print(f"⚠️ Aucun chunk trouvé dans {file_path}")
        return

    embeddings, texts = create_embeddings(chunks, model)
    

    # Sauvegarde
   
    save_chunk_metadata(chunks, os.path.join(output_dir, f"{base_name}.json"))
    print(f"✅ Traitement terminé pour : {base_name}")

# --- 6. Exécution complète sur un dossier ---
if __name__ == "__main__":
    input_dir = "./Division2"       # Dossier contenant les .txt
    output_dir = "./output2"         # Dossier de sortie pour .faiss et .json

    os.makedirs(output_dir, exist_ok=True)
    model = SentenceTransformer('allenai-specter')

    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_dir, filename)
            process_txt_file(file_path, output_dir, model)

    print("🎯 Tous les fichiers ont été traités.")
