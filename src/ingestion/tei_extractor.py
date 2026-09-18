"""Module d'extraction des sections textuelles à partir de documents TEI XML générés par GROBID."""

import os
from glob import glob
from pathlib import Path
from typing import List, Dict, Optional
from lxml import etree

NAMESPACES = {"tei": "http://www.tei-c.org/ns/1.0"}

DEFAULT_EXCLUDED_HEADS = [
    "introduction", "background", "related work", "state of the art",
    "discussion", "results", "résultats", "conclusion", "conclusions"
]

INVASIVE_DETECTION_EXCLUDED_HEADS = [
    "background", "related work", "state of the art"
]


class TEIExtractor:
    """Extracteur de métadonnées et de sections textuelles depuis les fichiers TEI XML."""

    def __init__(self, namespaces: Optional[Dict[str, str]] = None):
        self.namespaces = namespaces or NAMESPACES

    def extract_header_info(self, xml_file: str) -> Dict[str, str]:
        """Extrait le titre, la date, les auteurs et l'abstract d'un fichier TEI XML."""
        tree = etree.parse(xml_file)
        root = tree.getroot()

        # Titre principal
        title_el = root.find('.//tei:titleStmt/tei:title[@type="main"]', namespaces=self.namespaces)
        title_text = title_el.text.strip() if title_el is not None and title_el.text else "N/A"

        # Date de publication
        date_el = root.find('.//tei:publicationStmt/tei:date[@type="published"]', namespaces=self.namespaces)
        date_text = date_el.get("when") if date_el is not None and date_el.get("when") else "N/A"

        # Abstract
        abstract_paras = root.findall('.//tei:abstract//tei:p', namespaces=self.namespaces)
        abstract_text = ' '.join(p.text.strip() for p in abstract_paras if p.text) if abstract_paras else "N/A"

        # Auteurs et affiliations
        authors = []
        for author in root.findall('.//tei:sourceDesc//tei:author', namespaces=self.namespaces):
            pers_name = author.find('.//tei:persName', namespaces=self.namespaces)
            forename = pers_name.find('tei:forename', namespaces=self.namespaces) if pers_name is not None else None
            surname = pers_name.find('tei:surname', namespaces=self.namespaces) if pers_name is not None else None
            name = f"{forename.text if forename is not None else ''} {surname.text if surname is not None else ''}".strip()

            country_el = author.find('.//tei:affiliation//tei:country', namespaces=self.namespaces)
            country = country_el.text.strip() if country_el is not None and country_el.text else ""
            affil = f" ({country})" if country else ""
            if name:
                authors.append(f"{name}{affil}")

        return {
            "filename": os.path.basename(xml_file),
            "title": title_text,
            "date": date_text,
            "authors": ", ".join(authors),
            "abstract": abstract_text
        }

    def extract_abstract(self, root) -> Optional[str]:
        """Extrait le bloc résumé sous forme de section formatée."""
        abstract_el = root.find('.//tei:abstract', namespaces=self.namespaces)
        if abstract_el is not None:
            text = ''.join(abstract_el.itertext()).strip()
            if text:
                return f"[SECTION] abstract\n{text}"
        return None

    def extract_sections(
        self,
        xml_file: str,
        excluded_heads: Optional[List[str]] = None,
        invasive_detection: bool = False
    ) -> List[str]:
        """Extrait les sections de corps de texte en filtrant selon les en-têtes."""
        excluded = excluded_heads if excluded_heads is not None else DEFAULT_EXCLUDED_HEADS
        try:
            tree = etree.parse(xml_file)
            root = tree.getroot()
            chunks: List[str] = []

            if invasive_detection:
                title_el = root.find('.//tei:titleStmt/tei:title[@type="main"]', namespaces=self.namespaces)
                title_text = title_el.text.strip() if title_el is not None and title_el.text else "N/A"
                chunks.append(f"[SECTION] title\n{title_text}")

                abstract_chunk = self.extract_abstract(root)
                if abstract_chunk:
                    chunks.append(abstract_chunk)

            body = root.find('.//tei:text/tei:body', namespaces=self.namespaces)
            if body is None:
                return chunks

            for div in body.findall('.//tei:div', namespaces=self.namespaces):
                head = div.find('tei:head', namespaces=self.namespaces)
                head_text = head.text.strip().lower() if head is not None and head.text else ""

                if any(excl in head_text for excl in excluded):
                    continue

                section = f"[SECTION] {head_text or '(no title)'}\n"
                paragraphs = div.findall('.//tei:p', namespaces=self.namespaces)
                para_texts = [''.join(p.itertext()).strip() for p in paragraphs]
                content = "\n".join(t for t in para_texts if t)

                if content:
                    section += content + "\n"
                    chunks.append(section.strip())

            return chunks
        except Exception as e:
            print(f"Erreur lors du traitement de {xml_file}: {e}")
            return []

    def process_directory(
        self,
        input_dir: str,
        output_file: str,
        excluded_heads: Optional[List[str]] = None,
        invasive_detection: bool = False
    ) -> int:
        """Traite tous les XML TEI d'un répertoire et écrit les chunks concaténés."""
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        xml_files = sorted(glob(os.path.join(input_dir, "*.xml")))
        all_chunks: List[str] = []

        for xml_path in xml_files:
            chunks = self.extract_sections(xml_path, excluded_heads, invasive_detection)
            if chunks:
                header = f"\n\n{'=' * 100}\nFILE: {os.path.basename(xml_path)}\n{'=' * 100}\n"
                all_chunks.append(header + "\n\n".join(chunks))

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n\n".join(all_chunks))

        print(f"Traitement termine : {len(all_chunks)} fichiers traites -> {output_file}")
        return len(all_chunks)
