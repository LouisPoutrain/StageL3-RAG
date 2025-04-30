import os
import re

def extract_chunks_from_txt(content):
    sections = re.split(r"\[SECTION\]", content)
    chunks = []

    for section in sections:
        section = section.strip()
        if not section:
            continue
        lines = section.splitlines()
        if lines:
            title = lines[0].strip()
            text = "\n".join(line.strip() for line in lines[1:] if line.strip())
            if text:
                chunks.append((title, text))
    return chunks

def save_chunks_to_txt_file(chunks, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for title, text in chunks:
            f.write(f"[SECTION] {title}\n{text}\n\n")

def process_chunking_txt_file(input_txt_path):
    with open(input_txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Découpe chaque bloc par fichier TEI
    file_blocks = re.split(r"={5,}\s*FILE: (.+?)\.grobid\.tei\.xml\s*={5,}", content)

    # Le premier élément est vide ou du contenu inutile (avant le premier fichier)
    for i in range(1, len(file_blocks), 2):
        filename = file_blocks[i].strip()
        file_content = file_blocks[i + 1].strip()

        # Extraire les chunks
        chunks = extract_chunks_from_txt(file_content)

        # Construire le nom de sortie
        output_filename = f"{filename}.txt"
        save_chunks_to_txt_file(chunks, output_filename)
        print(f"Fichier généré : {output_filename}")

if __name__ == "__main__":
    input_path = "chunking.txt"
    process_chunking_txt_file(input_path)
