import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json

# --- 1. Parser le fichier TEI et extraire les chunks ---
def extract_chunks_from_tei(file_path):
    ns = {"tei": "http://www.tei-c.org/ns/1.0"}
    tree = ET.parse(file_path)
    root = tree.getroot()

    chunks = []
    for div in root.findall(".//tei:div", ns):
        section_title_el = div.find("tei:head", ns)
        section_title = section_title_el.text if section_title_el is not None else "Unknown Section"

        for p in div.findall("tei:p", ns):
            if p.text and p.text.strip():
                chunks.append({
                    "section": section_title.strip(),
                    "text": p.text.strip()
                })
    return chunks

# --- 2. Générer les embeddings avec SentenceTransformer ---
def create_embeddings(chunks, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    texts = [chunk['text'] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings, texts

# --- 3. Créer et stocker un index FAISS ---
def build_faiss_index(embeddings):
    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index

# --- 4. Sauvegarder le mapping texte <-> index ---
def save_chunk_metadata(chunks, path="chunk_metadata.json"):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

# --- Exécution complète ---
if __name__ == "__main__":
    file_path = "documents/LefortEtAl_JournalFormatted.grobid.tei.xml"
    chunks = extract_chunks_from_tei(file_path)
    embeddings, texts = create_embeddings(chunks)
    index = build_faiss_index(embeddings)

    faiss.write_index(index, "article_index.faiss")
    save_chunk_metadata(chunks)

    print("Index vectoriel et métadonnées sauvegardés avec succès.")
