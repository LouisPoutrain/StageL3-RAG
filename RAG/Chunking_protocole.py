import os
from glob import glob
from lxml import etree

# Espace de nom pour le XML TEI
NAMESPACES = {'tei': 'http://www.tei-c.org/ns/1.0'}

def extract_abstract(root):
    abstract_el = root.find('.//tei:abstract', namespaces=NAMESPACES)
    if abstract_el is not None:
        text = ''.join(abstract_el.itertext()).strip()
        if text:
            return f"[SECTION] abstract\n{text}"
    return None

def extract_non_excluded_sections(xml_file, EXCLUDED_HEADS, invasive_detection: bool = False):
    try:
        tree = etree.parse(xml_file)
        root = tree.getroot()
        relevant_chunks = []
        if invasive_detection:
            # Titre
            title = root.find('.//tei:titleStmt/tei:title[@type="main"]', namespaces=NAMESPACES)
            title_text = title.text.strip() if title is not None and title.text else "N/A"
            relevant_chunks.append(f"[SECTION] title\n{title_text}")

            # Ajouter l'abstract comme chunk s'il existe
            abstract_chunk = extract_abstract(root)
            if abstract_chunk:
                relevant_chunks.append(abstract_chunk)


        body = root.find('.//tei:text/tei:body', namespaces=NAMESPACES)
        if body is None:
            return []

        

        for div in body.findall('.//tei:div', namespaces=NAMESPACES):
            head = div.find('tei:head', namespaces=NAMESPACES)
            head_text = head.text.strip().lower() if head is not None and head.text else ""

            # Exclure certaines sections
            if any(excl in head_text for excl in EXCLUDED_HEADS):
                continue

            section = f"[SECTION] {head_text or '(no title)'}\n"
            paragraphs = div.findall('.//tei:p', namespaces=NAMESPACES)

            for p in paragraphs:
                paragraph_text = ''.join(p.itertext()).strip()
                if paragraph_text:
                    section += paragraph_text + "\n"

            relevant_chunks.append(section.strip())

        return relevant_chunks
    except Exception as e:
        print(f"❌ Erreur avec {xml_file}: {e}")
        return []

def process_directory(folder_path, output_file, EXCLUDED_HEADS, invasive_detection: bool = False):
    all_chunks = []

    for xml_path in glob(os.path.join(folder_path, "*.xml")):
        chunks = extract_non_excluded_sections(xml_path, EXCLUDED_HEADS, invasive_detection)
        if chunks:
            entry = f"\n\n{'='*100}\nFILE: {os.path.basename(xml_path)}\n{'='*100}\n"
            entry += "\n\n".join(chunks)
            all_chunks.append(entry)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_chunks))

    print(f"✅ {len(all_chunks)} fichiers traités. Résultats enregistrés dans '{output_file}'")

# === Utilisation ===
if __name__ == "__main__":

    # Liste des titres de sections à exclure (en minuscules)
    EXCLUDED_HEADS = [
        "introduction", "background", "related work", "state of the art",
        "discussion", "results", "résultats", "conclusion", "conclusions"
    ]


    EXCLUDED_HEADS_INVASIVE_DETECTION = [
        "background", "related work", "state of the art"
    ]

    dossier_tei = "results"              # <-- Remplace par ton dossier contenant les fichiers .xml
    sortie_txt = "Chunks/chunking.txt"   # <-- Fichier de sortie
    process_directory(dossier_tei, sortie_txt, EXCLUDED_HEADS, invasive_detection=False)


    sortie_txt_invasive_detection = "Chunks/chunking_invasive_detection.txt"
    process_directory(dossier_tei, sortie_txt_invasive_detection, EXCLUDED_HEADS_INVASIVE_DETECTION, invasive_detection=True)
