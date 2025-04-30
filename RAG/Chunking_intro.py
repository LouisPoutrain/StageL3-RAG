import os
from glob import glob
from lxml import etree

NAMESPACES = {'tei': 'http://www.tei-c.org/ns/1.0'}

def extract_tei_info(xml_file):
    tree = etree.parse(xml_file)
    root = tree.getroot()

    # Titre
    title = root.find('.//tei:titleStmt/tei:title[@type="main"]', namespaces=NAMESPACES)
    title_text = title.text.strip() if title is not None and title.text else "N/A"

    # Date
    date = root.find('.//tei:publicationStmt/tei:date[@type="published"]', namespaces=NAMESPACES)
    date_text = date.get("when") if date is not None and date.get("when") else "N/A"

    # Abstract
    abstract_paras = root.findall('.//tei:abstract//tei:p', namespaces=NAMESPACES)
    abstract_text = ' '.join(p.text.strip() for p in abstract_paras if p.text) if abstract_paras else "N/A"

    # Auteurs + Pays
    authors = []
    countries = set()
    for author in root.findall('.//tei:sourceDesc//tei:author', namespaces=NAMESPACES):
        persName = author.find('.//tei:persName', namespaces=NAMESPACES)
        forename = persName.find('tei:forename', namespaces=NAMESPACES) if persName is not None else None
        surname = persName.find('tei:surname', namespaces=NAMESPACES) if persName is not None else None
        name = f"{forename.text if forename is not None else ''} {surname.text if surname is not None else ''}".strip()

        country_el = author.find('.//tei:affiliation//tei:country', namespaces=NAMESPACES)
        country = country_el.text.strip() if country_el is not None and country_el.text else "N/A"
        if country != "N/A":
            countries.add(country)

        authors.append(f"{name} ({country})")

    # Formatage texte
    chunk_text = f"""\
FILE: {os.path.basename(xml_file)}
TITLE: {title_text}
DATE: {date_text}
AUTHORS: {', '.join(authors)}
ABSTRACT:
{abstract_text}
{"="*80}
"""
    return chunk_text

# === Paramètres ===
input_folder = "results"  # <-- Remplace par le chemin vers ton dossier
output_file = "Intro Chunks/output_chunks.txt"

# === Traitement de tous les fichiers XML du dossier ===
all_chunks = []

for xml_file in glob(os.path.join(input_folder, "*.xml")):
    try:
        chunk = extract_tei_info(xml_file)
        all_chunks.append(chunk)
    except Exception as e:
        print(f"❌ Erreur dans le fichier {xml_file}: {e}")

# === Sauvegarde dans un seul fichier ===
with open(output_file, "w", encoding="utf-8") as f:
    f.write('\n\n'.join(all_chunks))

print(f"✅ {len(all_chunks)} fichiers traités et sauvegardés dans '{output_file}'")
