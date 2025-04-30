import os
from glob import glob
from lxml import etree

# Espace de nom pour le XML TEI
NAMESPACES = {'tei': 'http://www.tei-c.org/ns/1.0'}

# Liste des titres de sections à exclure (en minuscules)
EXCLUDED_HEADS = [
    "introduction", "background", "related work", "state of the art",
    "discussion", "results", "résultats", "conclusion", "conclusions"
]

def extract_non_excluded_sections(xml_file):
    try:
        tree = etree.parse(xml_file)
        root = tree.getroot()
        body = root.find('.//tei:text/tei:body', namespaces=NAMESPACES)
        if body is None:
            return []

        relevant_chunks = []

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

def process_directory(folder_path, output_file):
    all_chunks = []

    for xml_path in glob(os.path.join(folder_path, "*.xml")):
        chunks = extract_non_excluded_sections(xml_path)
        if chunks:
            entry = f"\n\n{'='*100}\nFILE: {os.path.basename(xml_path)}\n{'='*100}\n"
            entry += "\n\n".join(chunks)
            all_chunks.append(entry)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_chunks))

    print(f"✅ {len(all_chunks)} fichiers traités. Résultats enregistrés dans '{output_file}'")

# === Utilisation ===
if __name__ == "__main__":
    dossier_tei = "results"              # <-- Remplace par ton dossier contenant les fichiers .xml
    sortie_txt = "Chunks/chunking.txt"   # <-- Fichier de sortie
    process_directory(dossier_tei, sortie_txt)
